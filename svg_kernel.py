import argparse
import os
import random
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

PERIOD = 7

# Env
import gym, json
from gym import spaces
from epipolicy.core.epidemic import construct_epidemic
from epipolicy.obj.act import construct_act

class EpiEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, session):
        super(EpiEnv, self).__init__()
        self.epi = construct_epidemic(session)
        total_population = np.sum(self.epi.static.default_state.obs.current_comp)
        obs_count = self.epi.static.compartment_count * self.epi.static.locale_count * self.epi.static.group_count
        action_count = 0
        action_param_count =  0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                action_count += 1
                action_param_count += len(itv.cp_list)
        self.act_domain = np.zeros((action_param_count, 2), dtype=np.float64)
        index = 0
        for itv in self.epi.static.interventions:
            if not itv.is_cost:
                for cp in itv.cp_list:
                    self.act_domain[index, 0] = cp.min_value
                    self.act_domain[index, 1] = cp.max_value
                    index += 1
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=0, high=1, shape=(action_count,), dtype=np.float64)
        # Example for using image as input:
        self.observation_space = spaces.Box(low=0, high=total_population, shape=(obs_count,), dtype=np.float64)

    def step(self, action):
        expanded_action = np.zeros(len(self.act_domain), dtype=np.float64)
        index = 0
        for i in range(len(self.act_domain)):
            if self.act_domain[i, 0] == self.act_domain[i, 1]:
                expanded_action[i] = self.act_domain[i, 0]
            else:
                expanded_action[i] = action[index]
                index += 1

        epi_action = []
        index = 0
        for itv_id, itv in enumerate(self.epi.static.interventions):
            if not itv.is_cost:
                epi_action.append(construct_act(itv_id, expanded_action[index:index+len(itv.cp_list)]))
                index += len(itv.cp_list)

        total_r = 0
        for i in range(PERIOD):
            state, r, done = self.epi.step(epi_action)
            total_r += r
            if done:
                break
        return state.obs.current_comp.flatten(), total_r, done, dict()

    def reset(self):
        state = self.epi.reset()
        return state.obs.current_comp.flatten()  # reward, done, info can't be included
    def render(self, mode='human'):
        pass
    def close(self):
        pass

def parse_args(main_args = None):
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default="PPO",
        help="the name of this experiment")
    parser.add_argument("--gym-id", type=str, default="HalfCheetahBulletEnv-v0",
        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--total-timesteps", type=int, default=700000,
        help="total timesteps of the experiments")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="ppo-implementation-details",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="weather to capture videos of the agent performances (check out `videos` folder)")

    parser.add_argument("--policy_plot_interval", type=int, default=1,
        help="seed of the experiment")

    # Algorithm specific arguments
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=2048,
        help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Use GAE for advantage computation")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=32,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=10,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.0,
        help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    if main_args is not None:
        args = parser.parse_args(main_args.split())
    else:
        args = parser.parse_args()
    args.num_steps //= PERIOD
    args.total_timesteps //= PERIOD
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    return args

epi_ids = ["SIR_A"]#, "SIR_B", "SIRV_A", "SIRV_B", "COVID_A", "COVID_B", "COVID_C"]

def make_env(gym_id, seed, idx, capture_video, run_name):
    def thunk():
        if gym_id in epi_ids:
            fp = open('jsons/{}.json'.format(gym_id), 'r')
            session = json.load(fp)
            env = EpiEnv(session)
        else:
            env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        # Our env is deterministic
        # env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def make_primal_env(gym_id):
    def thunk():
        if gym_id in epi_ids:
            fp = open('jsons/{}.json'.format(gym_id), 'r')
            session = json.load(fp)
            env = EpiEnv(session)
        else:
            env = gym.make(gym_id)
        return env
    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, env):
        super(Agent, self).__init__()
        self.actor_mean_sigma = nn.Sequential(
            layer_init(nn.Linear(np.array(env.observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2*np.prod(env.action_space.shape)), std=0.01),
        )
        self.m = MultivariateNormal(
            torch.zeros(np.prod(env.action_space.shape)), 
            torch.eye(np.prod(env.action_space.shape))
        )

    def get_action(self, x):
        actor_mean_sigma = self.actor_mean_sigma(x)
        action_mean = actor_mean_sigma[0,:np.prod(env.action_space.shape)]
        action_sigma = actor_mean_sigma[0,np.prod(env.action_space.shape):]
        epsilon = self.m.sample()
        
        action = action_mean + action_sigma * epsilon 
		
        # Apply the sigmoid to ensure the action is between 0 and 1
        action = torch.sigmoid(action)
        return action

env = make_primal_env(args.gym_id)()
agent = Agent(env)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

#### FROM SCRATCH IMPLEMENTATION OF SVG
NUM_SAMPLES = 10
UNROLL_HORIZON = 10
GAMMA = 0.99
NUM_UPDATES
# we need to store the rewards as we unroll

rewards = torch.zeros((UNROLL_HORIZON, NUM_SAMPLES), requires_grad=True)
actions = torch.zeros((UNROLL_HORIZON, NUM_SAMPLES) + env.action_space.shape)
obs = torch.zeros((UNROLL_HORIZON, NUM_SAMPLES) + env.observation_space.shape)

for update in range(0, NUM_UPDATES):
	for i in range(NUM_SAMPLES):
		next_obs = torch.Tensor(env.reset()).unsqueeze(0) # initial state
		for t in range(UNROLL_HORIZON):

			obs[t,i] = next_obs
			# ALGO LOGIC: action logic
			with torch.no_grad():
				action = agent.get_action(next_obs)
			actions[t,i] = action
			
			next_obs, reward, done, info = env.step(action.cpu().numpy())
			with torch.no_grad():
				rewards[t,i] = torch.tensor(reward, dtype=torch.float32)
			next_obs = torch.Tensor(next_obs).unsqueeze(0)

	# compute policy returns
	loss = torch.zeros((NUM_SAMPLES))
	for t in range(UNROLL_HORIZON):
		loss +=  GAMMA**t * rewards[t,:]

	mean_loss = loss.mean()

	optimizer.zero_grad()
	mean_loss.backward()
	optimizer.step()