"""
PyTorch code for SAC. Copied and modified from PyTorch code for SAC-NF (Mazoure et al., 2019): https://arxiv.org/abs/1905.06893
"""

import os
import sys
import argparse
import time
import datetime
import itertools
import random
import pickle
import glob

import gym
import numpy as np
import torch
from sac_gs import SAC
from normalized_actions import NormalizedActions
from replay_memory import ReplayMemory
import pandas as pd
try:
    import pybulletgym
except:
    print('No PyBullet Gym. Skipping...')
from utils import logging, get_time, print_args
from utils import save_checkpoint, load_checkpoint

from tensorboardX import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch code for SAC-NF (Mazoure et al., 2019,https://arxiv.org/abs/1905.06893)')
parser.add_argument('--env-name', default="Ant-v2",
                    help='name of the environment to run')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default:True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(tau) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers (default: 1)')
parser.add_argument('--actor_lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.05, metavar='G',
                    help='Temperature parameter alpha determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Temperature parameter alpha automaically adjusted.')
#parser.add_argument('--seed', type=int, default=456, metavar='N',
#                    help='random seed (default: 456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=3000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--hadamard',type=int,default=1)
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--cache', default='experiments', type=str)
parser.add_argument('--experiment', default=None, help='name of experiment')
parser.add_argument('--nb_evals', type=int, default=10,
                    help='nb of evaluations')
parser.add_argument('--resume', dest='resume', action='store_true', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                    help='flag to resume the experiments')
parser.add_argument('--exp-num', type=int, default=0,
                    help='experiment number')

# seed
parser.add_argument('--seed', type=int, default=456, metavar='N',
                    help='random seed (default: 456)')

# log
parser.add_argument('--log-interval', type=int, default=1000,
                    help='log print-out interval (step)')
parser.add_argument('--eval-interval', type=int, default=10000,
                    help='eval interval (step)')
parser.add_argument('--ckpt-interval', type=int, default=5000,
                    help='checkpoint interval (step)')

args = parser.parse_args()
args.hadamard = bool(args.hadamard)

# set env
if args.env_name == 'Humanoidrllab':
    from rllab.envs.mujoco.humanoid_env import HumanoidEnv
    from rllab.envs.normalized_env import normalize
    env = normalize(HumanoidEnv())
    max_episode_steps = float('inf')
    if args.seed >= 0:
        global seed_
        seed_ = args.seed
else:
    env = gym.make(args.env_name)
    max_episode_steps=env._max_episode_steps
    env=NormalizedActions(env)
    if args.seed >= 0:
        env.seed(args.seed)
if args.seed >= 0:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# set args
args.num_actions = env.action_space.shape[0]
args.max_action = env.action_space.high
args.min_action = env.action_space.low

# set cache folder
if args.cache is None:
    args.cache = 'experiments'
if args.experiment is None:
    args.experiment = '-'.join(['sac',
                                'mnh{}'.format(args.num_layers),
                                'sstep{}'.format(args.start_steps),
                                'a{}'.format(args.alpha),
                                'mlr{}'.format(args.lr),
                                'seed{}'.format(args.seed),
                                'exp{}'.format(args.exp_num),
                                ])
args.path = os.path.join(args.cache, args.experiment)
if args.resume:
    listing = glob.glob(args.path+'-19*') + glob.glob(args.path+'-20*')
    if len(listing) == 0:
        args.path = '{}-{}'.format(args.path, get_time())
    else:
        path_sorted = sorted(listing, key=lambda x: datetime.datetime.strptime(x, args.path+'-%y%m%d-%H:%M:%S'))
        args.path = path_sorted[-1]
        pass
else:
    args.path = '{}-{}'.format(args.path, get_time())
os.system('mkdir -p {}'.format(args.path))

# print args
logging(str(args), path=args.path)

# init tensorboard
writer = SummaryWriter(args.path)

# print config
configuration_setup='SAC'
configuration_setup+='\n'
configuration_setup+=print_args(args)
#for arg in vars(args):
#    configuration_setup+=' {} : {}'.format(str(arg),str(getattr(args, arg)))
#    configuration_setup+='\n'
logging(configuration_setup, path=args.path)

# init sac
agent = SAC(env.observation_space.shape[0], env.action_space, args)
logging("----------------------------------------", path=args.path)
logging(str(agent.critic), path=args.path)
logging("----------------------------------------", path=args.path)
logging(str(agent.policy), path=args.path)
logging("----------------------------------------", path=args.path)

# memory
memory = ReplayMemory(args.replay_size)

# resume
args.start_episode = 1
args.offset_time = 0 # elapsed
args.total_numsteps = 0
args.updates = 0
args.eval_steps = 0
args.ckpt_steps = 0
agent.load_model(args)
memory.load(os.path.join(args.path, 'replay_memory'), 'pkl')

# Training Loop
total_numsteps = args.total_numsteps # 0
updates = args.updates # 0
eval_steps = args.eval_steps # 0
ckpt_steps = args.ckpt_steps # 0
start_episode = args.start_episode # 1
offset_time = args.offset_time # 0
start_time = time.time()
if 'dataframe' in args:
    df = args.dataframe
else:
    df = pd.DataFrame(columns=["total_steps", "score_eval", "time_so_far"])

for i_episode in itertools.count(start_episode):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = np.random.uniform(env.action_space.low,env.action_space.high,env.action_space.shape[0])  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy
        if len(memory) > args.start_steps:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                (critic_1_loss, critic_2_loss,
                 policy_loss,
                 _, _,
                 policy_info,
                 )= agent.update_parameters(memory, args.batch_size, updates)
                updates += 1

                # log
                if updates % args.log_interval == 0:
                    logging("Episode: {}"
                            ", update: {}"
                            ", critic_1 loss: {:.3f}"
                            ", critic_2 loss: {:.3f}"
                            .format(
                            i_episode,
                            updates,
                            critic_1_loss,
                            critic_2_loss,
                            ), path=args.path)

                    writer.add_scalar('train/critic_1/loss/update', critic_1_loss, updates)
                    writer.add_scalar('train/critic_2/loss/update', critic_2_loss, updates)
        else:
            pass

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        eval_steps += 1
        ckpt_steps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask) # Append transition to memory

        state = next_state

    elapsed = round((time.time() - start_time + offset_time),2)
    logging("Episode: {}"
            ", time (sec): {}"
            ", total numsteps: {}"
            ", episode steps: {}"
            ", reward: {}"
            .format(
            i_episode,
            elapsed,
            total_numsteps,
            episode_steps,
            round(episode_reward, 2),
            ), path=args.path)
    writer.add_scalar('train/ep_reward/episode', episode_reward, i_episode)
    writer.add_scalar('train/ep_reward/step', episode_reward, total_numsteps)

    # evaluation
    if eval_steps>=args.eval_interval or total_numsteps > args.num_steps:
        logging('evaluation time', path=args.path)
        r=[]
        for _ in range(args.nb_evals):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, eval=True)

                next_state, reward, done, _ = env.step(action)
                episode_reward += reward

                state = next_state
            r.append(episode_reward)
        mean_reward=np.mean(r)

        # add to data frame
        res = {"total_steps": total_numsteps,
               "score_eval": mean_reward,
               "time_so_far": round((time.time() - start_time),2)}
        df = df.append(res, ignore_index=True)

        # add to log
        logging("----------------------------------------", path=args.path)
        logging("Test Episode: {}, mean reward: {}, ep reward: {}"
                .format(
                i_episode, round(mean_reward, 2), round(episode_reward, 2),
                ), path=args.path)
        logging("----------------------------------------", path=args.path)
        writer.add_scalar('test/ep_reward/mean/step', mean_reward, total_numsteps)
        writer.add_scalar('test/ep_reward/episode/step', episode_reward, total_numsteps)

        # writer
        writer.flush()

        # reset count
        eval_steps%=args.eval_interval

    if ckpt_steps>=args.ckpt_interval and args.ckpt_interval > 0:
        training_info = {
            'start_episode': i_episode+1,
            'offset_time': round((time.time() - start_time + offset_time),2),
            'total_numsteps': total_numsteps,
            'updates': updates,
            'eval_steps': eval_steps,
            'ckpt_steps': ckpt_steps,
            'dataframe': df,
            }
        agent.save_model(training_info)
        memory.save(os.path.join(args.path, 'replay_memory'), 'pkl')
        ckpt_steps%=args.ckpt_interval

    if total_numsteps > args.num_steps:
        break

env.close()
