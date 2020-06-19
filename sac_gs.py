"""
PyTorch code for SAC. Copied and modified from PyTorch code for SAC-NF (Mazoure et al., 2019): https://arxiv.org/abs/1905.06893
"""

import os
import torch
import torch.nn.functional as F
from torch.optim import Adam
from utils.sac import soft_update, hard_update
from model import GaussianPolicy, QNetwork, DeterministicPolicy
from utils import save_checkpoint, load_checkpoint


class SAC(object):
    """
    SAC class from Haarnoja et al. (2018)
    We leave the option to use automatice_entropy_tuning to avoid selecting entropy rate alpha
    """
    def __init__(self, num_inputs, action_space, args):
        #self.n_flow = args.n_flows
        #assert self.n_flow == 0
        self.num_inputs = num_inputs
        #self.flow_family = args.flow_family
        self.num_layers = args.num_layers
        self.args=args

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha

        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.device = torch.device("cuda" if args.cuda else "cpu") 

        self.critic = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)

        self.critic_target = QNetwork(num_inputs, action_space.shape[0], args.hidden_size).to(self.device)
        hard_update(self.critic_target, self.critic)

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = Adam([self.log_alpha], lr=args.lr)

        self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, self.num_layers, args).to(self.device)
        self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)

    def select_action(self, state, eval=False):
        """
        Select action for a state
        (Train) Sample an action from NF{N(mu(s),Sigma(s))}
        (Eval) Pass mu(s) through NF{}
        """
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if not eval:
            self.policy.train()
            action, _, _, _, _ = self.policy.evaluate(state)
        else:
            self.policy.eval()
            action, _, _, _, _ = self.policy.evaluate(state,eval=True)

        action = action.detach().cpu().numpy()
        return action[0]

    def update_parameters(self, memory, batch_size, updates):
        """
        Update parameters of SAC-NF
        Exactly like SAC, but keep two separate Adam optimizers for the Gaussian policy AND the NF layers
        .backward() on them sequentially
        """
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size=batch_size)

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        # for visualization
        info = {}

        ''' update critic '''
        with torch.no_grad():
            next_state_action, next_state_log_pi, _,_,_ = self.policy.evaluate(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * (min_qf_next_target)

        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value) # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        pi, log_pi, _,_,_ = self.policy.evaluate(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        nf_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # update
        self.critic_optim.zero_grad()
        qf1_loss.backward()
        self.critic_optim.step()

        self.critic_optim.zero_grad()
        qf2_loss.backward()
        self.critic_optim.step()

        self.policy_optim.zero_grad()
        policy_loss.backward()#retain_graph=True)
        self.policy_optim.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

        # update target value fuctions
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item(), info

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
