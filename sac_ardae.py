import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.optim import Adam, RMSprop
from utils.sac import soft_update, hard_update
from model import StochasticPolicy, QNetwork
from model import safe_log
import models as net
from models.aux import aux_loss_for_grad

from utils import save_checkpoint, load_checkpoint
from utils import sample_gaussian, logprob_gaussian, get_covmat


def atanh(x):
    return 0.5*torch.log((1+x)/(1-x))

def safe_cosh(x, value=45.):
    return torch.cosh(torch.clamp(x, min=-value, max=value))

def logprob_gaussian(mu, logvar, z, do_unsqueeze=True, do_mean=True):
    '''
    Inputs:⋅
        z: b1 x nz
        mu, logvar: b2 x nz
    Outputs:
        prob: b1 x nz
    '''
    if do_unsqueeze:
        z = z.unsqueeze(1)
        mu = mu.unsqueeze(0)
        logvar = logvar.unsqueeze(0)

    neglogprob = (z - mu)**2 / logvar.exp() + logvar + math.log(2.*math.pi)
    logprob = - neglogprob*0.5

    if do_mean:
        assert do_unsqueeze
        logprob = torch.mean(logprob, dim=1)

    return logprob

class SAC(object):
    """
    SAC class from Haarnoja et al. (2018)
    We leave the option to use automatice_entropy_tuning to avoid selecting entropy rate alpha
    """
    def __init__(self, num_inputs, action_space, args):
        self.num_enc_layers = args.num_enc_layers
        self.num_fc_layers = args.num_fc_layers
        self.num_inputs = num_inputs
        num_actions = action_space.shape[0]
        self.num_actions = num_actions

        self.args=args

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        # reward critic
        self.critic = QNetwork(
                num_inputs, num_actions, args.hidden_size,
                tau=args.mean_sub_tau, update_method=args.mean_upd_method,
                ).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        self.critic_target = QNetwork(
                num_inputs, num_actions, args.hidden_size,
                tau=args.mean_sub_tau, update_method=args.mean_upd_method,
                ).to(self.device)
        hard_update(self.critic_target, self.critic)

        # estimate partition func
        self.use_ptfnc = args.use_ptfnc
        self.ptflogvar = args.ptflogvar
        self.mean_sub_method = args.mean_sub_method
        assert args.gqnet_num_layers == 1
        assert args.gqnet_nonlin == 'relu'

        if self.automatic_entropy_tuning:
            raise NotImplementedError

        # policy
        self.policy = StochasticPolicy(num_inputs, num_actions, hidden_dim=args.hidden_size,
                noise_dim=args.noise_size,
                num_enc_layers=args.num_enc_layers,
                num_fc_layers=args.num_fc_layers,
                args=args,
                nonlinearity=args.policy_nonlin,
                fc_type=args.policy_type,
                ).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

        # jac_clamping
        self.lmbd = args.lmbd
        self.nu = args.nu
        self.eta = args.eta
        self.num_pert_samples = args.num_pert_samples
        if args.jac_act == 'none':
            self.jac_act = None
        elif args.jac_act == 'tanh':
            self.jac_act = torch.tanh
        else:
            raise NotImplementedError

        # cdae
        self.num_cdae_updates = args.num_cdae_updates
        self.train_nz_cdae = args.train_nz_cdae
        self.train_nstd_cdae = args.train_nstd_cdae
        self.std_scale = args.std_scale
        self.delta = args.delta

        if args.dae_enc_ctx == 'true':
            enc_ctx, enc_input = 'deep', True
        elif args.dae_enc_ctx == 'part':
            enc_ctx, enc_input = 'shallow', False
        else:
            enc_ctx, enc_input = False, False
        self.dae_ctx_type = args.dae_ctx_type
        if self.dae_ctx_type == 'state':
            ctx_dim = num_inputs
        elif self.dae_ctx_type == 'hidden':
            ctx_dim = args.hidden_size
        self.cdae = net.MLPGradCARDAE(
                input_dim=num_actions,
                context_dim=ctx_dim, #num_actions, #num_inputs+num_actions,
                std=1.,#opt.std_fin,
                h_dim=args.hidden_size,
                num_hidden_layers=args.dae_num_layers,
                nonlinearity=args.dae_nonlin,
                noise_type='gaussian',
                enc_ctx=enc_ctx,
                enc_input=enc_input,
                ).to(self.device)

        if args.d_optimizer == 'adam':
            self.cdae_optim = Adam(self.cdae.parameters(), lr=args.d_lr, betas=(args.d_beta1, 0.999))
        elif args.d_optimizer == 'rmsprop':
            self.cdae_optim = RMSprop(self.cdae.parameters(), lr=args.d_lr, momentum=args.d_momentum)
        else:
            raise NotImplementedError

    def select_action(self, state, eval=False):
        """
        Select action for a state
        (Train) Sample an action from NF{N(mu(s),Sigma(s))}
        (Eval) Pass mu(s) through NF{}
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            self.policy.train()
            action, _, _, _ = self.policy.evaluate(state)
        else:
            self.policy.eval()
            action, _, _, _ = self.policy.evaluate(state,eval=True)

        action = action.detach().cpu().numpy()
        return action[0]

    def est_partition_func(self,
            sample_size=128,
            next_state_batch=None, mask_batch=None,
            memory=None, batch_size=None,
            ptflogvar=-2.,
            ):
        if memory is not None:
            assert batch_size is not None
            # sample
            _, _, _, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)
        else:
            assert next_state_batch is not None
            assert mask_batch is not None
            batch_size = next_state_batch.size(0)

        # context
        _, nxt_preact_mean, nxt_hidden, _ = self.policy.evaluate(next_state_batch, eval=True)
        nxt_preact_mean = nxt_preact_mean.view(batch_size, 1, -1).detach()
        if self.dae_ctx_type == 'state':
            nxt_context = next_state_batch.view(batch_size, 1, -1).detach()
        elif self.dae_ctx_type == 'hidden':
            nxt_context = nxt_hidden.view(batch_size, 1, -1).detach()

        # sample
        _nxt_preact_mean = nxt_preact_mean.expand(batch_size, sample_size, self.num_actions)
        _nxt_preact_logvar = ptflogvar*nxt_preact_mean.new_ones(_nxt_preact_mean.size())
        _newz = sample_gaussian(_nxt_preact_mean, _nxt_preact_logvar) # bsz x ssz x zdim

        # proposal distribution
        logproposal = logprob_gaussian(
                _nxt_preact_mean,
                _nxt_preact_logvar,
                _newz,
                do_unsqueeze=False,
                do_mean=False,
                ) # bsz x ssz x 1
        logproposal = torch.sum(logproposal, dim=2, keepdim=True) \
                    - self.num_actions * math.log(self.std_scale) # bsz x ssz x 1

        # unnormalized distribution
        newz = _newz-nxt_preact_mean
        scaled_newz = self.std_scale*newz
        stdmat = torch.zeros(batch_size, sample_size, 1, device=self.device).fill_(0)
        logp_ptfunc = (
                self.cdae.logprob(scaled_newz, nxt_context, std=stdmat, scale=self.std_scale).detach()
                - logproposal)

        logp_ptfunc_max, _ = torch.max(logp_ptfunc, dim=1, keepdim=True)
        rprob_ptfunc = (logp_ptfunc - logp_ptfunc_max).exp() # relative prob
        logp_ptfunc = torch.log(torch.mean(rprob_ptfunc, dim=1, keepdim=True) + 1e-12) + logp_ptfunc_max # bsz x 1

        return logp_ptfunc.detach()

    def update_parameters(self, memory, batch_size, updates):
        ''' update dae '''
        cdae_loss = torch.FloatTensor(1).zero_().sum().to(self.device)
        for i in range(self.num_cdae_updates):
            # zero grad
            self.cdae_optim.zero_grad()

            # sample
            state_batch, _, _, _, _ = memory.sample(batch_size=batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.device)

            # init
            batch_size = state_batch.size(0)

            # get mean
            action_mean, preact_mean, _, _ = self.policy.evaluate(state_batch, eval=True)
            action_mean = action_mean.view(batch_size, 1, -1).detach()
            preact_mean = preact_mean.view(batch_size, 1, -1).detach()

            # forward policy
            action, preact, hidden, _ = self.policy.evaluate(state_batch, num_samples=self.train_nz_cdae)
            action = action.view(batch_size, self.train_nz_cdae, -1).detach()
            preact = preact.view(batch_size, self.train_nz_cdae, -1).detach()

            # get scaled_preact_sub_mean
            scaled_action_sub_mean = self.std_scale*(action-action_mean)
            scaled_preact_sub_mean = self.std_scale*(preact-preact_mean)

            # set std
            std_preact = torch.std(scaled_preact_sub_mean, dim=1, keepdim=True) # bsz x 1 x dims
            std = self.delta*torch.mean(std_preact, dim=2, keepdim=True) # bsz x 1 x 1

            # set context and stdmat
            if self.dae_ctx_type == 'state':
                context = state_batch.view(batch_size, 1, -1).detach()
            elif self.dae_ctx_type == 'hidden':
                context = hidden.view(batch_size, 1, -1).detach()
            stdmat = std*torch.randn(batch_size, self.train_nz_cdae*self.train_nstd_cdae, 1, device=self.device)

            # forward cdae
            _scaled_preact_sub_mean = scaled_preact_sub_mean.unsqueeze(2).expand(
                    batch_size, self.train_nz_cdae, self.train_nstd_cdae, self.num_actions,
                    ).reshape(batch_size, self.train_nz_cdae*self.train_nstd_cdae, self.num_actions)
            _, cdae_loss = self.cdae(_scaled_preact_sub_mean, context, std=stdmat, scale=self.std_scale)

            # update
            cdae_loss.backward()
            self.cdae_optim.step()

            # msc
            stdmat = torch.zeros(batch_size, self.train_nz_cdae, 1, device=self.device)
            logprob = self.cdae.logprob(scaled_preact_sub_mean, context, std=stdmat, scale=self.std_scale)
            _action = action.cpu().detach()
            #_preact = preact.cpu().detach()
            #_logprob = logprob.cpu().detach()
            info = {
                 'action': _action,
                 #'preact': _preact,
                 #'logprob': _logprob,
            }

        ''' update critic / policy '''
        # sample
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # init
        batch_size = state_batch.size(0)

        ''' update critic '''
        # context
        _, nxt_preact_mean, nxt_hidden, _ = self.policy.evaluate(next_state_batch, eval=True)
        nxt_preact_mean = nxt_preact_mean.view(batch_size, 1, -1).detach()
        if self.dae_ctx_type == 'state':
            nxt_context = next_state_batch.view(batch_size, 1, -1).detach()
        elif self.dae_ctx_type == 'hidden':
            nxt_context = nxt_hidden.view(batch_size, 1, -1).detach()

        # sample state_action
        sample_size = 1 # self.train_nz_ent
        assert sample_size == 1
        _next_state_action, _nxt_preact, _, _ = self.policy.evaluate(next_state_batch, num_samples=sample_size)
        _next_state_action = _next_state_action.view(batch_size, sample_size, -1)
        _nxt_preact = _nxt_preact.view(batch_size, sample_size, -1)
        next_state_action  = _next_state_action[:, 0, :]
        nxt_preact = _nxt_preact[:, 0, :]

        #get partition func
        if self.use_ptfnc > 0:
            logp_ptfunc = self.est_partition_func(next_state_batch=next_state_batch, mask_batch=mask_batch, sample_size=self.use_ptfnc, ptflogvar=self.ptflogvar)
        elif self.use_ptfnc == 0:
            logp_ptfunc = 0
        else:
            raise NotImplementedError

        # view
        nxt_action = next_state_action.view(batch_size, 1, -1).detach()
        nxt_preact = nxt_preact.view(batch_size, 1, -1).detach()

        # get next_state_neglogp
        scaled_nxt_preact_sub_mean = self.std_scale*(nxt_preact-nxt_preact_mean)
        stdmat = torch.zeros(batch_size, 1, 1, device=self.device).fill_(0)
        next_state_logp = \
                self.cdae.logprob(scaled_nxt_preact_sub_mean, nxt_context, std=stdmat, scale=self.std_scale).detach() \
                - logp_ptfunc \
                - safe_log(1. - nxt_action.pow(2)).sum(dim=2, keepdim=True).detach() \
                + self.num_actions * math.log(self.std_scale) # log p(x) = log p(y) + log scale (if y = ax)

        #next_state_logp = torch.mean(next_state_logp, dim=1, keepdim=True)
        next_state_neglogp = -next_state_logp

        # view
        next_state_neglogp = next_state_neglogp.view(batch_size, 1)

        # reward critic
        qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
        min_qf_next_target = torch.min(qf1_next_target, qf2_next_target).detach() + self.alpha * next_state_neglogp
        next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        # update mean of neglogp critic
        if self.mean_sub_method == 'ms':
            self.critic.update_mean(next_q_value)
        elif self.mean_sub_method == 'entms':
            self.critic.update_mean(mask_batch*self.gamma*self.alpha*next_state_neglogp)
        elif self.mean_sub_method == 'none':
            pass
        else:
            raise NotImplementedError

        # reward critic
        if self.mean_sub_method == 'none':
            offset = 0
        else:
            offset = self.critic.mean
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1 + offset, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2 + offset, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        ''' update policy '''
        # context
        _, preact_mean, _, _ = self.policy.evaluate(state_batch, eval=True)
        preact_mean = preact_mean.view(batch_size, 1, -1).detach()

        # forward current policy
        sample_size = 1 #self.train_nz_ent
        action, preact, hidden, eps = self.policy.evaluate(state_batch, num_samples=sample_size)

        # view
        action_reshaped = action.view(batch_size, sample_size, -1)
        preact_reshaped = preact.view(batch_size, sample_size, -1)

        # eval current action
        qf1_act, qf2_act = self.critic(state_batch, action_reshaped[:, 0, :])
        min_qf_act = torch.min(qf1_act, qf2_act)

        # detach
        action_detached = action_reshaped.detach()
        preact_detached = preact_reshaped.detach()

        # estimate neglogp
        scaled_preact_sub_mean = self.std_scale*(preact_reshaped.detach()-preact_mean)
        if self.dae_ctx_type == 'state':
            context = state_batch.view(batch_size, 1, -1).detach()
        elif self.dae_ctx_type == 'hidden':
            context = hidden.view(batch_size, 1, -1).detach()
        stdmat = torch.zeros(batch_size, sample_size, 1, device=self.device).fill_(0)
        glogpz = self.cdae.glogprob(scaled_preact_sub_mean, context, std=stdmat, scale=self.std_scale).detach()
        glogpx = self.std_scale * glogpz
        glogpy = glogpx + 2.*action_detached
        negglogp = -glogpy

        # estimate loss
        policy_loss = (-min_qf_act).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        # jacobian clamping
        lmbd = 1. + float(updates)**self.nu / 1000. if self.nu > 0 else self.lmbd
        lmbd = min(lmbd, self.lmbd) if self.lmbd > 0 else lmbd
        info['lmbd'] = lmbd
        if lmbd > 0:
            jaclmp_loss = lmbd*self.policy.jac_clamping_loss(preact, hidden, eps, num_eps_samples=sample_size, num_pert_samples=self.num_pert_samples, eta_min=self.eta, activation=self.jac_act)
            policy_loss += jaclmp_loss

        # estimate aux_losses
        policy_aux_loss_for_curr = aux_loss_for_grad(preact_reshaped, self.alpha*(-negglogp)/float(batch_size*sample_size))

        # update
        self.critic_optim.zero_grad()
        qf1_loss.backward(retain_graph=True)
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_aux_loss_for_curr.backward(retain_graph=True)
        policy_loss.backward()
        self.policy_optim.step()

        # update alpha
        if self.automatic_entropy_tuning:
            raise NotImplementedError
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        # update target value fuctions
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return (qf1_loss.item(), qf2_loss.item(), policy_loss.item(),
                cdae_loss.item(),
                info,
                )

    def save_model(self, info):
        """
        Save the weights of the network (actor and critic separately)
        """
        # policy
        save_checkpoint({
            **info,
            'state_dict': self.policy.state_dict(),
            'optimizer' : self.policy_optim.state_dict(),
            }, self.args, filename='policy-ckpt.pth.tar')

        # critic
        save_checkpoint({
            **info,
            'state_dict': self.critic.state_dict(),
            'optimizer' : self.critic_optim.state_dict(),
            }, self.args, filename='critic-ckpt.pth.tar')
        save_checkpoint({
            **info,
            'state_dict': self.critic_target.state_dict(),
            #'optimizer' : self.critic_optim.state_dict(),
            }, self.args, filename='critic_target-ckpt.pth.tar')

        # cdae
        save_checkpoint({
            **info,
            'state_dict': self.cdae.state_dict(),
            'optimizer' : self.cdae_optim.state_dict(),
            }, self.args, filename='cdae-ckpt.pth.tar')

    def load_model(self, args):
        """
        Jointly or separately load actor and critic weights
        """
        # policy
        load_checkpoint(
            model=self.policy,
            optimizer=self.policy_optim,
            opt=args,
            device=self.device,
            filename='policy-ckpt.pth.tar',
            )

        # critic
        load_checkpoint(
            model=self.critic,
            optimizer=self.critic_optim,
            opt=args,
            device=self.device,
            filename='critic-ckpt.pth.tar',
            )
        load_checkpoint(
            model=self.critic_target,
            #optimizer=self.critic_optim,
            opt=args,
            device=self.device,
            filename='critic_target-ckpt.pth.tar',
            )

        # cdae
        load_checkpoint(
            model=self.cdae,
            optimizer=self.cdae_optim,
            opt=args,
            device=self.device,
            filename='cdae-ckpt.pth.tar',
            )
