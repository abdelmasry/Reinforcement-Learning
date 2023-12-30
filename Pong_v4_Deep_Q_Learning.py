"""
###############################################################################################################
##################################### Reinforcement Learning Assignment#3 #####################################
###############################################################################################################
###############################################################################################################
#################################### Assignment Description(Pong env) #########################################
In this game you control the right paddle, you compete against the left paddle
controlled by the computer. You each try to keep deflecting the ball away from your
goal and into your opponent's goal.
#################################### Environment Description ##################################################
###############################################################################################################
######################################### Observations ######################################
By default, the environment returns the RGB image that is displayed to human
players as an observation. So we can assume that the respective observation spaces are:
* Box([0 ... 0], [255 ... 255], (128,), uint8)
* Box([[0 ... 0] ... [0 ... 0]], [[255 ... 255] ... [255 ... 255]], (250, 160), uint8)
#############################################################################################
########################################### Actions #########################################
By default, all actions that can be performed on an Atari 2600 are available in this
environment. However, if you use v0 or v4 or specify (full_action_space=False) during
initialization, only a reduced number of actions (those that are meaningful in this
game) are available.
## Type: Discrete (6)
| Num |    Action   |
|:---:|:-----------:|
|  0  |    NOOP     |
|  1  |    Fire     |
|  2  |    Right    |
|  3  |    Left     |
|  4  |  Right Fire |
|  5  |  Left Fire  |
#############################################################################################
############################################ Reward #########################################
* You get score points for getting the ball to pass the opponent's paddle. You lose
points if the ball passes your paddle.
#############################################################################################
########################################### Assignment Requirements ###########################################
* Implementation in python Deep Q_Learning algorithms based on the Pong-v4 environment.
* Report contain comparison between Q-Learning and Deep Q-Learning in terms of accuracy and conversion time.
###############################################################################################################
############################################## NAMES AND IDS ##################################################
1- Abdelrahman Mahmoud Alsayed -> 20190732
2- Yousef Yasser Ezzat -> 20190652
3- Omar Mohammed Abdelbary -> 20190358
4- Belal Kamal Ashraf -> 20190137
"""
#################################################### Imports ##################################################
import time
import gym
import cv2
import torch
import warnings
import gym.spaces
import collections
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

device = torch.device("mps")  # GPU -> METAL PERFORMANCE SHADDERS for MAC
warnings.filterwarnings("ignore")
###############################################################################################################
############################################### Hyper-Parameters ##############################################
eps_start = 1.0
eps_min = 0.02
gamma = 0.99
batch_size = 32
eps_decay = 0.999985
replay_size = 10000
learning_rate = 1e-4
MEAN_REWARD_BOUND = -15.0
replay_start_size = 10000
sync_target_frames = 1000


###############################################################################################################
################################################# AGENT CLASS #################################################
class Agent:
    def __init__(self, env, exp_buffer):
        self.env = env
        self.exp_buffer = exp_buffer
        self._reset()

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def play_step(self, net, epsilon=0.0, device="cpu"):
        done_reward = None
        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array([self.state], copy=False)
            state_v = torch.tensor(state_a).to(device)
            q_vals_v = net(state_v)
            _, act_v = torch.max(q_vals_v, dim=1)
            action = int(act_v.item())

        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        self.exp_buffer.append(self.state, action, reward, new_state, is_done)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward


###############################################################################################################
############################################# REPLAY BUFFER CLASS #############################################
class ReplayBuffer:
    """
    For generating batches, from the observations
    To be taken as an input for the network
    """
    def __init__(self, buffer_size, batch_size):
        self.buffer = collections.deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = collections.namedtuple(
            "Experience", field_names=["state", "action", "reward", "new_state", "done"]
        )

    def __len__(self):
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.buffer.append(e)

    def sample(self):
        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[idx] for idx in indices]
        )
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8),
        )


###############################################################################################################
################################################### WRAPPERS ##################################################
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(MaxAndSkipEnv, self).__init__(env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = collections.deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class ProcessFrame84(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

    def observation(self, obs):
        return ProcessFrame84.process(obs)

    @staticmethod
    def process(frame):
        if frame.size == 210 * 160 * 3:
            img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
        elif frame.size == 250 * 160 * 3:
            img = np.reshape(frame, [250, 160, 3]).astype(np.float32)
        else:
            assert False, "Unknown resolution."
        img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
        resized_screen = cv2.resize(img, (84, 110), interpolation=cv2.INTER_AREA)
        x_t = resized_screen[18:102, :]
        x_t = np.reshape(x_t, [84, 84, 1])
        return x_t.astype(np.uint8)


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps, dtype=np.float32):
        super(BufferWrapper, self).__init__(env)
        self.dtype = dtype
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            old_space.low.repeat(n_steps, axis=0),
            old_space.high.repeat(n_steps, axis=0),
            dtype=dtype,
        )

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=self.dtype)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super(ImageToPyTorch, self).__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_shape[-1], old_shape[0], old_shape[1]),
            dtype=np.float32,
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


def make_env(env_name,render_mode:bool):
    if render_mode == True:
        env = gym.make(env_name, render_mode="human")
    else:    
        env = gym.make(env_name)        
    env = MaxAndSkipEnv(env)
    env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ImageToPyTorch(env)
    env = BufferWrapper(env, 4)
    return ScaledFloatFrame(env)


###############################################################################################################
############################################### Q-Network CLASS ###############################################
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)


###############################################################################################################
################################################ TRAIN FUNCTION ###############################################
def train(agent, writer, local_net, target_net, buffer, optimizer):
    best_mean_reward = None
    total_rewards = []
    frame_idx = 0
    epsilon = eps_start
    while True:
        frame_idx += 1
        epsilon = max(epsilon * eps_decay, eps_min)

        reward = agent.play_step(local_net, epsilon, device=device)
        if reward is not None:
            total_rewards.append(reward)

            mean_reward = np.mean(total_rewards[-100:])
            print(
                "%d:  %d games, mean reward %.3f, (epsilon %.2f)"
                % (frame_idx, len(total_rewards), mean_reward, epsilon)
            )

            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("reward_100", mean_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

            if best_mean_reward is None or best_mean_reward < mean_reward:
                torch.save(
                    local_net.state_dict(),
                    "/Users/abdelrahmanibrahim/Desktop/Senior/Reinforcement_Learning/Assignment#3/876.dat",
                )
                best_mean_reward = mean_reward
                if best_mean_reward is not None:
                    print("Best mean reward updated %.3f" % (best_mean_reward))

            if mean_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break

        if len(buffer) < replay_start_size:
            continue

        batch = buffer.sample()
        states, actions, rewards, next_states, dones = batch

        states_v = torch.tensor(states).to(device)
        next_states_v = torch.tensor(next_states).to(device)
        actions_v = torch.tensor(actions).to(device)
        rewards_v = torch.tensor(rewards).to(device)
        done_mask = torch.ByteTensor(dones).to(device)

        state_action_values = (
            local_net(states_v).gather(1, actions_v.unsqueeze(-1)).squeeze(-1)
        )

        next_state_values = target_net(next_states_v).max(1)[0]

        next_state_values[done_mask] = 0.0

        next_state_values = next_state_values.detach() # we don't calculate the gradient 

        expected_state_action_values = next_state_values * gamma + rewards_v

        loss_t = nn.MSELoss()(state_action_values, expected_state_action_values) # local net vs. target net

        optimizer.zero_grad()
        loss_t.backward()
        optimizer.step()

        if frame_idx % sync_target_frames == 0:
            target_net.load_state_dict(local_net.state_dict())

    writer.close()


###############################################################################################################
################################################### SOLVING ###################################################
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
ENV = make_env(DEFAULT_ENV_NAME,render_mode=False)
LOCAL_NET = DQN(ENV.observation_space.shape, ENV.action_space.n).to(device)
TARGET_NET = DQN(ENV.observation_space.shape, ENV.action_space.n).to(device)
WRITER = SummaryWriter(comment="-" + DEFAULT_ENV_NAME)
BUFFER = ReplayBuffer(replay_size, batch_size=batch_size)
AGENT = Agent(ENV, BUFFER)
OPTIMIZER = optim.Adam(LOCAL_NET.parameters(), lr=learning_rate)
#train(AGENT, WRITER, LOCAL_NET, TARGET_NET, BUFFER, OPTIMIZER)  # takes around an hour
#----------------------------------------------------DONE-----------------------------------------------------#

