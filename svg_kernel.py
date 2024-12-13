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
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.actor_mean_sigma = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 2*np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.m = MultivariateNormal(
            torch.zeros(np.prod(envs.single_action_space.shape)), 
            torch.eye(np.prod(envs.single_action_space.shape))
        )

    def get_action(self, x):
        action_mean = self.actor_mean_sigma(x)[:np.prod(envs.single_action_space.shape)]
        action_sigma = self.actor_mean_sigma(x)[np.prod(envs.single_action_space.shape):]
        epsilon = m.sample()
        
        action = action_mean + action_sigma * epsilon 
        # TODO: ensure action is within boundaries
        return action, None, None, None

from tqdm import tqdm
args = parse_args("--gym-id SIR_A --seed 1")
run_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
print("Running", run_name)
writer = SummaryWriter(f"runs/{run_name}")
writer.add_text(
    "hyperparameters",
    "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
)

# TRY NOT TO MODIFY: seeding
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = args.torch_deterministic

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
)
# envs = gym.vector.AsyncVectorEnv(
#     [make_env(args.gym_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
# )
test_env = make_primal_env(args.gym_id)()
# test_env = make_env(args.gym_id, args.seed, 0, args.capture_video, run_name)()
assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

agent = Agent(envs).to(device)
optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

# ALGO Logic: Storage setup
obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
values = torch.zeros((args.num_steps, args.num_envs)).to(device)

# TRY NOT TO MODIFY: start the game
global_step = 0
start_time = time.time()
next_obs = torch.Tensor(envs.reset()).to(device)
next_done = torch.zeros(args.num_envs).to(device)
num_updates = args.total_timesteps // args.batch_size
ROLLOUT_HORIZON = 35

csv_file = open('runs/{}/records.csv'.format(run_name), 'w')

for update in tqdm(range(1, num_updates + 1)):
    # Annealing the rate if instructed to do so.
    if args.anneal_lr:
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

    for step in range(0, args.num_steps):
        global_step += 1 * args.num_envs
        obs[step] = next_obs
        dones[step] = next_done

        # ALGO LOGIC: action logic
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs)
        actions[step] = action

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, reward, done, info = envs.step(action.cpu().numpy())
        rewards[step] = torch.tensor(reward).to(device).view(-1)
        next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

        for item in info:
            if "episode" in item.keys():
                print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                break

    # bootstrap value if not done
    with torch.no_grad():
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
            else:
                nextnonterminal = 1.0 - dones[t + 1]
            next_return = returns[t + 1]
            returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return

    # flatten the batch
    b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
    b_logprobs = logprobs.reshape(-1)
    b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
    b_returns = returns.reshape(-1)

    # Optimizing the policy and value network
    b_inds = np.arange(args.batch_size)
    clipfracs = []
    for epoch in tqdm(range(max(1, args.update_epochs))): # eventually remove epoch as we look at trajectories only once
        np.random.shuffle(b_inds)
        for start in range(0, args.batch_size, args.minibatch_size):
            # TODO: do we actually need mini-batches as we look at all trajectories at once?
            end = start + args.minibatch_size
            mb_inds = b_inds[start:end]

            mb_returns = b_returns[mb_inds]
            pg_loss = -mb_returns
            
            loss = pg_loss 

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
            optimizer.step()


    y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
    var_y = np.var(y_true)
    explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

    # TRY NOT TO MODIFY: record rewards for plotting purposes
    writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
    writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
    writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
    writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
    writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
    writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
    writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
    writer.add_scalar("losses/explained_variance", explained_var, global_step)
    print("SPS:", int(global_step / (time.time() - start_time)))
    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    # PLOT POLICY
    if args.gym_id in epi_ids and (update - 1) % args.policy_plot_interval == 0:
        test_obs = torch.Tensor(test_env.reset())
        env_obs = torch.Tensor(envs.reset()).to(device)
        timestep = 0
        total_r = 0
        done = False
        itv_line = []
        while not done:
            with torch.no_grad():
                action_mean = agent.get_action_mean(env_obs)
                test_action_mean = torch.mean(action_mean, 0)
                test_action_mean = torch.clamp(test_action_mean, 0, 1)
             
            test_obs, r, done, _ = test_env.step(test_action_mean.cpu().numpy())
            test_obs = torch.Tensor(test_obs)
            itv_index = 0
            itv_array = []
            for itv in test_env.epi.static.interventions:
                if not itv.is_cost:
                    v = float(test_action_mean[itv_index])
                    writer.add_scalar('charts/policy_{}/{}'.format(global_step, itv.name), v, timestep)
                    itv_array.append(v)
                    itv_index += 1
            itv_line.append(itv_array)
            
            env_obs, _, _, _ = envs.step(action_mean.cpu().numpy())
            env_obs = torch.Tensor(env_obs).to(device)

            total_r += r
            timestep += PERIOD
            
        line = '|'.join([str(global_step), str(total_r), str(itv_line)]) + '\n'
        csv_file.write(line)
        writer.add_scalar('charts/learning_curve', total_r, global_step)
        print("At global step {}, total_rewards={}".format(global_step, total_r))

csv_file.close()
envs.close()
writer.close()