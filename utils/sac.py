import math
import torch
import numpy as np

def get_params(policy,flow_family):
    if flow_family in ['iaf','dsf','ddsf']:
        gaussian = policy.parameters()
        nf = policy.n_flow.transforms.parameters()
        return gaussian, nf
    gaussian, nf = [],[]
    for key,value in policy.named_parameters():
        if "n_flow" in key:
            nf.append(value)
        else:
            gaussian.append(value)
    return gaussian, nf

def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p

def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

def rollout(n_episodes,env,agent_actor,save_in_memory,eval,random):
    inner_steps=0
    scores = []
    traj=[]
    for _ in range(n_episodes):
        state = env.reset()
        if save_in_memory:
            traj.append(state)
        episode_reward = 0
        while True:
            if random:
                action = env.action_space.sample()
                print('random policy')
            else:
                action = agent_actor.select_action(state,eval=eval)  # Sample action from policy
            next_state, reward, done, _ = env.step(action)  # Step
         
            if save_in_memory:
                traj.append(next_state)
            state = next_state
            inner_steps += 1
            episode_reward += reward
            if done:
                scores.append(episode_reward)
                break
    return np.mean(scores), inner_steps, traj