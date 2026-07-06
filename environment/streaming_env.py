import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class StreamingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # 5 bitrate choices
        self.bitrates = [240, 360, 480, 720, 1080]

        # action: index of bitrate
        self.action_space = spaces.Discrete(len(self.bitrates))

        # state: bandwidth, buffer, prev_bitrate
        self.observation_space = spaces.Box(
            low=0,
            high=100,
            shape=(3,),
            dtype=np.float32
        )

        self.max_buffer = 20.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.buffer = 5.0
        self.prev_bitrate = 720
        self.t = 0
        self.bandwidth = self._get_bandwidth()

        return self._get_state(), {}

    def _get_state(self):
        return np.array([
            self.bandwidth,
            self.buffer,
            self.prev_bitrate
        ], dtype=np.float32)

    def _get_bandwidth(self):
        base = random.choice([1, 2, 3, 5, 8])
        noise = random.uniform(-0.5, 0.5)
        return max(0.5, base + noise)

    def step(self, action):

        bitrate = self.bitrates[action]

        self.bandwidth = self._get_bandwidth()

        chunk_size = (bitrate / 1000) * 2.0
        download_time = chunk_size / self.bandwidth

        # buffer dynamics
        self.buffer += 2.0 - download_time
        self.buffer = np.clip(self.buffer, 0, self.max_buffer)

        # rebuffer
        rebuffer = 1.0 if self.buffer <= 0.1 else 0.0

        # reward
        quality = bitrate / 1000.0
        switch = 1.0 if bitrate != self.prev_bitrate else 0.0

        reward = quality - 2.0 * rebuffer - 0.1 * switch

        self.prev_bitrate = bitrate
        self.t += 1

        terminated = self.t >= 100
        truncated = False

        return self._get_state(), reward, terminated, truncated, {
            "buffer": self.buffer,
            "bandwidth": self.bandwidth,
            "bitrate": bitrate
        }