import numpy as np
import gymnasium as gym
from gymnasium import spaces


class GridEnv(gym.Env):
    """
    M x N grid environment.
    Observation: MxN np.ndarray with ints:
      0 empty, 1 obstacle, 2 goal, 3 agent
    Actions: Discrete(4): 0 up, 1 down, 2 left, 3 right
    """
    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        M=5,
        N=5,
        K=3,
        max_steps=200,
        seed=None,
        *,
        reward_goal=10.0,
        reward_collision=-10.0,
        reward_step=-0.1,
        init_random=True,
        start_pos=None,
    ):
        """
        Arguments:
          M, N, K, max_steps, seed : same as before
          reward_goal, reward_collision, reward_step : floats to tweak rewards
          init_random (bool): if True, sample random agent/goal/obstacles now (in __init__)
                              — avoids needing to call reset() just to get positions.
          start_pos (tuple or None): if provided, use this (i,j) as the agent start position.
                                     Must be inside grid.
        """
        super().__init__()

        assert M >= 2 and N >= 2, "Grid must be at least 2x2"
        assert 0 <= K < M * N - 2, "K must be less than available free cells"

        self.M = M
        self.N = N
        self.K = K
        self.max_steps = max_steps

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=3, shape=(self.M, self.N), dtype=np.int8)

        self._rng = np.random.default_rng(seed)

        # state placeholders
        self.agent_pos = None
        self.goal_pos = None
        self.obstacles = None
        self.step_count = 0

        # rewards (now configurable)
        self.reward_goal = float(reward_goal)
        self.reward_collision = float(reward_collision)
        self.reward_step = float(reward_step)

        # validate start_pos if provided
        if start_pos is not None:
            if not (isinstance(start_pos, (tuple, list)) and len(start_pos) == 2):
                raise ValueError("start_pos must be a tuple (i,j) or None")
            si, sj = start_pos
            if not (0 <= si < self.M and 0 <= sj < self.N):
                raise ValueError("start_pos is out of bounds")
            agent_start = (int(si), int(sj))
        else:
            agent_start = None

        # store start_pos so reset() can reuse it
        self._start_pos = agent_start

        # If a start_pos was provided, set agent_pos immediately even if init_random is False.
        # This lets users inspect or visualize the agent position without calling reset().
        if agent_start is not None:
            self.agent_pos = agent_start
        else:
            self.agent_pos = None

        # optionally initialize random world immediately so user doesn't have to call reset()
        if init_random:
            # _place_random_positions will respect agent_start if not None
            agent, goal, obstacles = self._place_random_positions(agent_start=agent_start)
            self.agent_pos = agent
            self.goal_pos = goal
            self.obstacles = obstacles
            # step_count remains 0 until steps begin
        else:
            # if not init_random, leave goal_pos/obstacles unset until reset()
            self.goal_pos = None
            self.obstacles = None

    def _make_empty_grid(self):
        return np.zeros((self.M, self.N), dtype=np.int8)
    
    def _place_random_positions(self, agent_start=None):
        """
        Sample (agent, goal, obstacles) without overlap.
        If agent_start is provided, agent is fixed there.
        """
        all_positions = [(i, j) for i in range(self.M) for j in range(self.N)]

        # If agent fixed, remove it from candidate pool and choose remaining (goal + K obstacles)
        if agent_start is not None:
            if agent_start not in all_positions:
                raise ValueError("agent_start out of bounds")
            # remaining positions
            remaining = [p for p in all_positions if p != agent_start]
            # need to choose 1 + K positions from remaining
            chosen = self._rng.choice(len(remaining), size=1 + self.K, replace=False)
            goal = remaining[int(chosen[0])]
            obstacles = {remaining[int(idx)] for idx in chosen[1:]}
            return agent_start, goal, obstacles

        # else sample agent, goal, obstacles together (no replacement)
        chosen = self._rng.choice(len(all_positions), size=2 + self.K, replace=False)
        agent = all_positions[int(chosen[0])]
        goal = all_positions[int(chosen[1])]
        obstacles = {all_positions[int(idx)] for idx in chosen[2:]}
        return agent, goal, obstacles
    
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.step_count = 0

        # choose positions
        agent, goal, obstacles = self._place_random_positions()
        self.agent_pos = agent
        self.goal_pos = goal
        self.obstacles = obstacles

        obs = self._get_observation()
        info = {"agent_pos": self.agent_pos, "goal_pos": self.goal_pos, "obstacles": self.obstacles}
        return obs, info

    def _get_observation(self):
        grid = self._make_empty_grid()
        for (i, j) in self.obstacles:
            grid[i, j] = 1
        gi, gj = self.goal_pos
        grid[gi, gj] = 2
        ai, aj = self.agent_pos
        grid[ai, aj] = 3
        return grid.copy()
    
    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action {action}"
        self.step_count += 1

        di, dj = 0, 0
        if action == 0:   # up
            di, dj = -1, 0
        elif action == 1: # down
            di, dj = 1, 0
        elif action == 2: # left
            di, dj = 0, -1
        elif action == 3: # right
            di, dj = 0, 1

        new_i = self.agent_pos[0] + di
        new_j = self.agent_pos[1] + dj

        # Check bounds
        if not (0 <= new_i < self.M and 0 <= new_j < self.N):
            obs = self._get_observation()
            reward = self.reward_collision
            terminated = True
            truncated = False
            info = {"reason": "out_of_bounds"}
            return obs, reward, terminated, truncated, info

        new_pos = (new_i, new_j)

        # Collision with obstacle
        if new_pos in self.obstacles:
            # move into obstacle considered collision (we can keep agent where it was)
            reward = self.reward_collision
            terminated = True
            truncated = False
            info = {"reason": "hit_obstacle", "pos": new_pos}
            obs = self._get_observation()
            return obs, reward, terminated, truncated, info

        # Move agent
        self.agent_pos = new_pos

        # Reached goal?
        if self.agent_pos == self.goal_pos:
            reward = self.reward_goal
            terminated = True
            truncated = False
            info = {"reason": "reached_goal"}
            obs = self._get_observation()
            return obs, reward, terminated, truncated, info

        # Step penalty and check truncation by max_steps
        reward = self.reward_step
        terminated = False
        truncated = False
        if self.step_count >= self.max_steps:
            truncated = True
            info = {"reason": "max_steps_exceeded"}
        else:
            info = {}

        obs = self._get_observation()
        return obs, reward, terminated, truncated, info