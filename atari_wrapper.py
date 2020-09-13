import numpy as np
import collections
from skimage.color import rgb2gray
from skimage.transform import resize
import gym

class RepeatActionMaxFrame(gym.Wrapper):
    def __init__(self, env, repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
        super(RepeatActionMaxFrame, self).__init__(env)
        self.repeat = repeat
        self.shape = env.observation_space.low.shape
        self.frame_buffer = np.zeros(shape=(2, *self.shape))
        self.clip_rewards = clip_rewards
        self.no_ops = no_ops
        self.fire_first = fire_first

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self.repeat):
            state, reward, done, info = self.env.step(action)
            if self.clip_rewards: reward = np.clip(np.array([reward]), -1, 1)[0]
            total_reward += reward
            idx = i % 2
            self.frame_buffer[idx] = state
            if done: break

        max_frame = np.maximum(self.frame_buffer[0], self.frame_buffer[1])
        return max_frame, total_reward, done, info

    def reset(self):
        state = self.env.reset()
        no_ops = np.random.randint(self.no_ops) + 1 if self.no_ops > 0 else 0
        for _ in range(no_ops):
            _, _, done, _ = self.env.step(0)
            if done: self.env.reset()
        if self.fire_first:
            assert self.env.unwrapped.get_action_meanings()[1] == 'FIRE'
            state, *_ = self.env.step(1)
        self.frame_buffer = np.zeros(shape=(2, *self.shape))
        self.frame_buffer[0] = state
        return state

class PreprocessFrame(gym.ObservationWrapper):
    def __init__(self, shape, env):
        super(PreprocessFrame, self).__init__(env)
        self.shape = (shape[2], shape[0], shape[1])
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=self.shape, dtype=np.float32)

    def observation(self, state):
        new_frame = rgb2gray(state)
        resized_screen = resize(new_frame, self.shape[1:])
        state = resized_screen.reshape(self.shape) / 255.
        return state
    
class StackFrames(gym.ObservationWrapper):
    def __init__(self, env, repeat):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box(
                                env.observation_space.low.repeat(repeat, axis=0),
                                env.observation_space.high.repeat(repeat, axis=0),
                                dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)

    def reset(self):
        self.stack.clear()
        state = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(state)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

    def observation(self, state):
        self.stack.append(state)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

def make_environment(env_name, shape=(84, 84, 1), repeat=4, clip_rewards=False, no_ops=0, fire_first=False):
    env = gym.make(env_name)
    env = RepeatActionMaxFrame(env, repeat, clip_rewards, no_ops, fire_first)
    env = PreprocessFrame(shape, env)
    env = StackFrames(env, repeat)
    return env