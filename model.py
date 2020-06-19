import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from copy import deepcopy
import sys


try:
    #sys.path.append('../torchkit/torchkit') # Chin-Wei's NAF code for NAF
    from torchkit import nn as nn_, flows, utils
    from torchkit.transforms import from_numpy, binarize
    from torch.autograd import Variable
except:
    print('No torchkit. IAF will not run. Check README.md to install...')

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

def safe_log(z):
    return torch.log(z + 1e-7)

def weights_init_policy_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1: #\
        #    and not classname.find('WeightNormalizedLinear') != -1 \
        #    and not classname.find('ResLinear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=0.5)
        torch.nn.init.constant_(m.bias, 0)
 
def weights_init_value_fn(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# msc functions
def get_grad(logprob, input):
    return torch.autograd.grad(logprob, input, retain_graph=True, create_graph=True)[0]

# models
class MAF(object):
    
    def __init__(self, args, p):

        self.args = args        
        self.__dict__.update(args.__dict__)
        self.p = p
        
        dim = p
        dimc = 1
        dimh = p
        flowtype = args.flow_family
        num_flow_layers = args.n_flows
        num_ds_dim = p
        num_ds_layers = 2
        fixed_order = False
                 
        act = nn.ELU()
        if flowtype == 'iaf':
            flow = flows.IAF
        elif flowtype == 'dsf':
            flow = lambda **kwargs:flows.IAF_DSF(num_ds_dim=num_ds_dim,
                                                 num_ds_layers=num_ds_layers,
                                                 **kwargs)
        elif flowtype == 'ddsf':
            flow = lambda **kwargs:flows.IAF_DDSF(num_ds_dim=num_ds_dim,
                                                  num_ds_layers=num_ds_layers,
                                                  **kwargs)
        
        
        sequels = [nn_.SequentialFlow(
            flow(dim=dim,
                 hid_dim=dimh,
                 context_dim=dimc,
                 num_layers=2+1,
                 activation=act,
                 fixed_order=fixed_order),
            flows.FlipFlow(1)) for i in range(num_flow_layers)] + \
                  [flows.LinearFlow(dim, dimc),]
                
                
        self.flow = nn.Sequential(
                *sequels)

    def parameters(self):
        return self.flow.parameters()

    def named_parameters(self):
        return self.flow.named_parameters()
    
        
    def state_dict(self):
        return self.flow.state_dict()

    def load_state_dict(self, states):
        self.flow.load_state_dict(states)
         
    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.flow.parameters(),
                                self.args.clip)

class RLNN(nn.Module):

    def __init__(self):
        super(RLNN, self).__init__()
#        self.state_dim = state_dim
#        self.action_dim = action_dim
#        self.max_action = max_action

    def set_params(self, params):
        """
        Set the params of the network to the given parameters
        """
        cpt = 0
        for param in self.parameters():
            tmp = np.product(param.size())

            if torch.cuda.is_available():
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            else:
                param.data.copy_(torch.from_numpy(
                    params[cpt:cpt + tmp]).view(param.size()))
            cpt += tmp

    def get_params(self):
        """
        Returns parameters of the actor
        """
        return deepcopy(np.hstack([to_numpy(v).flatten() for v in
                                   self.parameters()]))

    def get_grads(self):
        """
        Returns the current gradient
        """
        return deepcopy(np.hstack([to_numpy(v.grad).flatten() for v in self.parameters()]))

    def get_size(self):
        """
        Returns the number of parameters of the network
        """
        return self.get_params().shape[0]

    def load_model(self, filename, net_name):
        """
        Loads the model
        """
        if filename is None:
            return

        self.load_state_dict(
            torch.load('{}/{}.pkl'.format(filename, net_name),
                       map_location=lambda storage, loc: storage)
        )

    def save_model(self, output, net_name):
        """
        Saves the model
        """
        torch.save(
            self.state_dict(),
            '{}/{}.pkl'.format(output, net_name)
        )

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, hidden_dim):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class QNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers=1, nonlinearity='relu', norm='none', tau=0.005, update_method='avg'):
        super(QNetwork, self).__init__()
        assert num_layers == 1
        assert nonlinearity == 'relu'
        assert norm == 'none'

        # Q1 architecture
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        #self.main1 = MLP(num_inputs + num_actions, hidden_dim, 1, nonlinearity=nonlinearity, num_hidden_layers=num_layers, use_nonlinearity_output=False, norm=norm)

        # Q2 architecture
        self.linear4 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear5 = nn.Linear(hidden_dim, hidden_dim)
        self.linear6 = nn.Linear(hidden_dim, 1)
        #self.main2 = MLP(num_inputs + num_actions, hidden_dim, 1, nonlinearity=nonlinearity, num_hidden_layers=num_layers, use_nonlinearity_output=False, norm=norm)

        #self.apply(weights_init_)

        # mean activation of q network
        self.register_buffer('mean', torch.zeros(1, 1))
        self.register_buffer('count', torch.zeros(1).long())
        self.tau = tau
        self.update_method = update_method
        assert update_method in ['avg', 'exp']

    def update_mean(self, input, tau=None):
        tau = self.tau if tau is None else tau
        assert input.dim() == 2
        assert input.size(1) == 1

        batch_size = input.size(0)
        input = torch.mean(input, dim=0, keepdim=True)
        count = self.count.item()

        if count == 0:
            self.mean.data.copy_(input.data)
            self.count.data.add_(batch_size)
        elif self.update_method == 'exp':
            self.mean.data.copy_(self.mean.data * (1.0 - tau) + input.data * tau)
            self.count.data.add_(batch_size)
        elif self.update_method == 'avg':
            self.mean.data.copy_(self.mean.data * float(count)/float(count+batch_size) + input.data * float(batch_size)/float(count+batch_size))
            self.count.data.add_(batch_size)
        else:
            raise NotImplementedError

    def forward(self, state, action):
        xu = torch.cat([state, action], 1)

        x1 = F.relu(self.linear1(xu))
        x1 = F.relu(self.linear2(x1))
        x1 = self.linear3(x1)
        #x1 = self.main1(xu)

        x2 = F.relu(self.linear4(xu))
        x2 = F.relu(self.linear5(x2))
        x2 = self.linear6(x2)
        #x2 = self.main2(xu)

        return x1, x2

class RadialFlow(nn.Module):

    def __init__(self, dim,args):
        super().__init__()

        self.z0 = nn.Parameter(torch.Tensor(1, dim))
        self.log_alpha = nn.Parameter(torch.Tensor(1))
        self.beta = nn.Parameter(torch.Tensor(1))

        self.reset_parameters()

    def reset_parameters(self):

        if True:                                                                                                                         
            self.z0.data.uniform_(-0.01, 0.01)                                                                                       
            self.log_alpha.data.uniform_(-0.01, 0.01)                                                                                        
            self.beta.data.uniform_(-0.01, 0.01)                                                                                         
        else:                                                                                                            
            torch.nn.init.xavier_normal_(self.z0,gain=0.5)                                                                          
            torch.nn.init.constant_(self.log_alpha,0)                                                                           
            torch.nn.init.constant_(self.beta,1)
            
    def forward(self, z):
        r = torch.norm(z-self.z0,p='fro',dim=-1).view(-1,1)
        h = 1/(self.log_alpha.exp()+r)
        return z + self.beta * h * (z-self.z0)

class PlanarFlow(nn.Module):

    def __init__(self, dim,args):
        super().__init__()

        
        if args.hadamard:
            self.weight = nn.Parameter(torch.Tensor(dim, dim))
            self.bias = nn.Parameter(torch.Tensor(1,dim))
            self.scale = nn.Parameter(torch.Tensor(1, dim))
        else:
            self.weight = nn.Parameter(torch.Tensor(1, dim))
            self.bias = nn.Parameter(torch.Tensor(1))
            self.scale = nn.Parameter(torch.Tensor(1, dim))

        self.tanh = nn.Tanh()
        
        self.reset_parameters()

    def reset_parameters(self):

        if True:
            self.weight.data.uniform_(-0.01, 0.01)
            self.scale.data.uniform_(-0.01, 0.01)      
            self.bias.data.uniform_(-0.01, 0.01)      
        else:
            torch.nn.init.xavier_normal_(self.weight,gain=1)
            torch.nn.init.xavier_normal_(self.scale,gain=1)
            torch.nn.init.constant_(self.bias,0)

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        return z + self.scale * self.tanh(activation)


"""
normalizing flow code borrowed from: https://github.com/ex4sperans/variational-inference-with-normalizing-flows
"""
class NormalizingFlow(nn.Module):

    def __init__(self, dim, flow_length,flow_family,state_dim,args):
        super().__init__()
        
        self.dim = dim
        self.flow_family = flow_family

        flow,jacobian = None,None
        if flow_family == 'planar':
            flow = PlanarFlow
            jacobian = PlanarFlowLogDetJacobian
        elif flow_family == 'radial':
            flow = RadialFlow
            jacobian = RadialFlowLogDetJacobian
        
        if flow_family in ['iaf','dsf','ddsf']:
            self.transforms = MAF(args,dim)

        else:
        
            self.transforms = nn.Sequential(*(
                flow(dim,args) for i in range(flow_length)
            ))

            self.log_jacobians = nn.Sequential(*(
                jacobian(t,args) for t in self.transforms
            ))

    def forward(self, z):
        if self.flow_family in ['iaf','dsf','ddsf']:
            zk, log_jac = self.transforms.density(z)
            return zk, log_jac, None
        log_jacobians = []
        zs = []
        zs.append(z)
        for transform, log_jacobian in zip(self.transforms, self.log_jacobians):
            log_jacobians.append(log_jacobian(z))
            z = transform(z)
            zs.append(z)
        zk = z
        return zk, torch.stack(log_jacobians).sum(-1).transpose(1,0), zs

class PlanarFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, affine,args):
        super().__init__()

        self.weight = affine.weight
        self.bias = affine.bias
        self.scale = affine.scale
        self.tanh = affine.tanh
        self.args = args

    def forward(self, z):
        activation = F.linear(z, self.weight, self.bias)
        differentiate=[]
#        print(activation.size())
        if self.args.hadamard:
            psi = (1 - self.tanh(activation) ** 2) 
            
            J = (1+psi * self.scale * torch.diag(self.weight))
          
            det_grads = safe_log(J.abs().sum(-1)).unsqueeze(-1)
           
            return det_grads

            b=torch.FloatTensor(differentiate)
            b=b.view(activation.size()[0],activation.size()[1]) 
            psi = b * self.weight
        else:
            psi = (1 - self.tanh(activation) ** 2) * self.weight
        det_grad = 1 + torch.mm(psi, self.scale.t())
        return safe_log(det_grad.abs())

class RadialFlowLogDetJacobian(nn.Module):
    """A helper class to compute the determinant of the gradient of
    the planar flow transformation."""

    def __init__(self, radial,args):
        super().__init__()

        self.z0 = radial.z0
        self.log_alpha = radial.log_alpha
        self.beta = radial.beta

    def forward(self, z):
        r = torch.norm(z-self.z0,p='fro',dim=-1).view(-1,1)
        alpha = self.log_alpha.exp()
        h = 1/(alpha+r)
        h_prime = -1/(alpha+r)**2
        d = z.size(-1)
        return safe_log(torch.abs((1+self.beta*h)**(d-1)*(1+self.beta*h+self.beta*h_prime*r)))

class NormalizingFlowPolicy(RLNN):
    def __init__(self, num_inputs, num_actions, hidden_dim,n_flow,flow_family,args):
        super(NormalizingFlowPolicy, self).__init__()

        self.flow_family = flow_family
        self.n_flow = n_flow
        self.args=args
        self.num_actions = num_actions

        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        if self.args.sigma==0:
            self.log_std_linear = nn.Linear(num_inputs, num_actions)
        elif self.args.sigma==-1:
            self.log_sigma_eps = torch.nn.Parameter(torch.zeros(num_actions))
        self.n_flow = NormalizingFlow(num_actions,n_flow,flow_family,num_inputs,args)

        self.apply(weights_init_policy_fn)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        mean = self.mean_linear(x)

        if self.args.sigma==0:
            """
            Case 1. Learned Sigma(s)
            """
            log_std = self.log_std_linear(state)
        elif self.args.sigma==-1:
            """
            Case 2. Learned Sigma
            """
            log_std = self.log_sigma_eps
        elif self.args.sigma>0:
            """
            Case 3. Fixed Sigma (by user)
            """
            size = (mean.size(0),self.num_actions)
            std_ = torch.ones(size)*self.args.sigma 
            log_std=torch.log(std_)
            if self.args.cuda:
                log_std=log_std.cuda()            
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def evaluate(self, state, eval=False, num_samples=1):
        mean, log_std = self.forward(state)
        std_ = log_std.exp()

        if num_samples > 1:
            batch_size = mean.size(0)
            mean = mean.unsqueeze(1).expand(-1, num_samples, -1).contiguous().view(batch_size*num_samples, -1)
            if len(std_.size()) == 1:
                std_ = std_.unsqueeze(0).unsqueeze(0).expand(batch_size, num_samples, -1).contiguous().view(batch_size*num_samples, -1)
            elif len(std_.size()) == 2:
                std_ = std_.unsqueeze(1).expand(-1, num_samples, -1).contiguous().view(batch_size*num_samples, -1)
            else:
                raise NotImplementedError
        normal = Normal(mean,std_)
        eps = normal.rsample()
        x_t = eps

        if eval:
            x_t=mean

        if self.args.cuda:
            x_t=x_t.cuda()

        x_t, log_jacobians, zs = self.n_flow(x_t)
        action = torch.tanh(x_t)

        if self.flow_family in ['iaf','dsf','ddsf']:
            log_prob = 0
            log_prob -= safe_log(1 - action.pow(2))
            log_prob = log_prob.sum(-1, keepdim=True)
            log_prob -= log_jacobians.mean(-1,keepdim=True)
        else:
            log_prob = normal.log_prob(eps)
            if self.args.cuda:
                log_prob=log_prob.cuda()

            log_prob -= safe_log(1 - action.pow(2))
            log_prob = log_prob.sum(-1, keepdim=True)
            log_prob -= log_jacobians.sum(-1,keepdim=True)

        return action, log_prob, x_t, eps, log_jacobians

class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x))
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

# stochastic policy
def sample_noise(sz, std=None, device=torch.device('cpu')):
    std = std if std is not None else 1
    eps = torch.randn(*sz).to(device)
    return std * eps

from models.layers import MLP, ResMLP, Identity
from utils import minrelu, cond_jac_clamping_loss

class StochasticPolicy(RLNN):
    def __init__(self, num_inputs, num_actions, hidden_dim, noise_dim, num_enc_layers, num_fc_layers, args, nonlinearity='elu', fc_type='mlp'):
        super(StochasticPolicy, self).__init__()

        self.num_enc_layers = num_enc_layers
        self.num_fc_layers = num_fc_layers
        self.args=args
        self.num_actions = num_actions
        self.noise_dim = noise_dim
        assert noise_dim >= num_actions, 'noise_dim: {}, num_actions: {}'.format(noise_dim, num_actions)

        inp_dim = num_inputs if num_enc_layers < 0 else hidden_dim
        if num_enc_layers < 0:
            self.encode = Identity()
        else:
            self.encode = MLP(num_inputs, hidden_dim, hidden_dim, nonlinearity=nonlinearity, num_hidden_layers=num_enc_layers, use_nonlinearity_output=True)

        if fc_type == 'mlp':
            self.fc = MLP(inp_dim+noise_dim, hidden_dim, num_actions, nonlinearity=nonlinearity, num_hidden_layers=num_fc_layers, use_nonlinearity_output=False)
            torch.nn.init.normal_(self.fc.fc.weight, std=1.)
        elif fc_type == 'wnres':
            self.fc = ResMLP(inp_dim+noise_dim, hidden_dim, num_actions, nonlinearity=nonlinearity, num_hidden_layers=num_fc_layers, layer='wnlinear', use_nonlinearity_output=False)
        elif fc_type == 'res':
            self.fc = ResMLP(inp_dim+noise_dim, hidden_dim, num_actions, nonlinearity=nonlinearity, num_hidden_layers=num_fc_layers, layer='linear', use_nonlinearity_output=False)
        elif fc_type == 'mlpdeep':
            self.fc = MLP(inp_dim+noise_dim, 64, num_actions, nonlinearity=nonlinearity, num_hidden_layers=num_fc_layers, use_nonlinearity_output=False)
            torch.nn.init.normal_(self.fc.fc.weight, std=1.)
        elif fc_type == 'wnresdeep':
            self.fc = ResMLP(inp_dim+noise_dim, 64, num_actions, nonlinearity=nonlinearity, num_hidden_layers=num_fc_layers, layer='wnlinear', use_nonlinearity_output=False)
        elif fc_type == 'resdeep':
            self.fc = ResMLP(inp_dim+noise_dim, 64, num_actions, nonlinearity=nonlinearity, num_hidden_layers=num_fc_layers, layer='linear', use_nonlinearity_output=False)
        else:
            raise NotImplementedError

    def forward(self, state):
        raise NotImplementedError
        stt = self.encode(state)
        return stt

    def evaluate(self, state, eval=False, std=None, num_samples=1):
        # init
        batch_size = state.size(0)

        # sample noise
        if eval:
            eps = sample_noise(sz=[batch_size*num_samples, self.noise_dim], std=0, device=state.device)
        else:
            eps = sample_noise(sz=[batch_size*num_samples, self.noise_dim], std=std, device=state.device)

        # encode state
        stt = self.encode(state)

        # forward
        x_t = self.forward_w_eps(stt, eps=eps, num_samples=num_samples)
        action = torch.tanh(x_t)

        return action, x_t, stt, eps

    def forward_w_eps(self, stt, eps, num_samples):
        # init
        batch_size = stt.size(0)

        # view
        stt = stt.unsqueeze(1).expand(-1, num_samples, -1).contiguous()
        stt = stt.view(batch_size*num_samples, -1)

        # concat
        stt_nos = torch.cat([stt, eps], dim=1)

        # forward
        x_t = self.fc(stt_nos)

        return x_t

    def jac_clamping_loss(self, x_t, stt, eps, num_eps_samples, num_pert_samples, eta_min, p=2, EPS=0.01, activation=None):
        def forward(stt, eps_bar, num_eps_samples, num_pert_samples):
            return self.forward_w_eps(stt, eps=eps_bar, num_samples=num_eps_samples*num_pert_samples)
        if activation is not None:
            def postprocessing(x):
                return activation(x)
        else:
            postprocessing = None
        return cond_jac_clamping_loss(forward=forward, x=x_t, ctx=stt, z=eps, num_z_samples=num_eps_samples, num_pert_samples=num_pert_samples, eta_min=eta_min, p=p, EPS=EPS, postprocessing=postprocessing)


class GaussianPolicy(RLNN):
    def __init__(self, num_inputs, num_actions, hidden_dim, num_layers, args):
        super(GaussianPolicy, self).__init__()

        self.args=args
        self.num_actions = num_actions
        self.num_layers = num_layers

        if num_layers == 1:
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
        elif num_layers == 2:
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        elif num_layers == 3:
            self.linear1 = nn.Linear(num_inputs, hidden_dim)
            self.linear2 = nn.Linear(hidden_dim, hidden_dim)
            self.linear3 = nn.Linear(hidden_dim, hidden_dim)
        else:
            raise NotImplementedError
        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

    def forward(self, state):
        if self.num_layers == 1:
            hidden = F.relu(self.linear1(state))
        elif self.num_layers == 2:
            hidden = F.relu(self.linear1(state))
            hidden = F.relu(self.linear2(hidden))
        elif self.num_layers == 3:
            hidden = F.relu(self.linear1(state))
            hidden = F.relu(self.linear2(hidden))
            hidden = F.relu(self.linear3(hidden))
        mean = self.mean_linear(hidden)
        log_std = self.log_std_linear(hidden)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)

        return mean, log_std

    def evaluate(self, state, eval=False, num_samples=1):
        mean, log_std = self.forward(state)
        std_ = log_std.exp()

        if num_samples > 1:
            batch_size = mean.size(0)
            mean = mean.unsqueeze(1).expand(-1, num_samples, -1).contiguous().view(batch_size*num_samples, -1)
            std_ = std_.unsqueeze(1).expand(-1, num_samples, -1).contiguous().view(batch_size*num_samples, -1)

        normal = Normal(mean,std_)
        eps = normal.rsample()
        x_t = eps

        if eval:
            x_t=mean

        if self.args.cuda:
            x_t=x_t.cuda()

        action = torch.tanh(x_t)

        log_prob = normal.log_prob(eps)
        if self.args.cuda:
            log_prob=log_prob.cuda()

        log_prob -= safe_log(1 - action.pow(2))
        log_prob = log_prob.sum(-1, keepdim=True)

        return action, log_prob, x_t, eps, 0 #log_jacobians
