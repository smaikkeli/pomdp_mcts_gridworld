from typing import Any
import numpy as np

import numpy as np
from copy import deepcopy
from gymnasium import spaces
from gymnasium.core import ObsType

from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.constants import DIR_TO_VEC

from src.core.actions import Actions
from src.core.grid import ModifiedGrid
from src.core.agent import Agent

from src.planning.mcts import choose_action

# Environment class
class GridWorld(MiniGridEnv):
    def __init__(self, 
                 size= 16, 
                 start_pos=None, 
                 agent_view_size = 7, 
                 mode_densities = None, 
                 mode_positions = None, 
                 **kwargs):

        mission_space = MissionSpace(mission_func=self._gen_mission)
        size = size + 2
        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            **kwargs,
        )

        assert agent_view_size % 2 == 1, "View size must be odd"
        
        self.actions = Actions
        self.action_space = spaces.Discrete(len(Actions))
        self.grid = ModifiedGrid(size, size)
        self.step_count = 0
        
        self.agent_dir = 0
        self.agent_start_pos = start_pos
        self.agent_previous_pos = None
        self.agent_view_size = agent_view_size
        self.goal_pos = None
        
        self.agent = None
        self.mode_densities = mode_densities
        self.mode_positions = mode_positions
        
    def choose_action(self, num_sim = 1000, exploration_weight = 2):
        best_action = choose_action(self, num_simulations = num_sim, max_depth = (self.agent_view_size), exploration_weight = exploration_weight)
        return best_action
    
    def max_neighboring_reward(self):
        '''
        Checks the best reward by moving to the neighboring cells
        '''
        best_reward = -np.inf
        best_action = None
        for action in range(len(Actions)):
            env_copy = self.copy()
            obs, reward, terminated, truncated, _ = env_copy.step(action)
            if reward > best_reward:
                best_reward = reward
                best_action = action

        return best_action
        
    def reset(self, seed=None, options=None):
        obs, _ = super().reset(seed=seed)
        self.agent = self.initialize_agent()
        return obs, {}
    
    # New step function to handle new actions
    def step(self, action: Actions):

        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        def move_and_check_goal(new_pos):
            nonlocal terminated
            cell = self.grid.get(*new_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = new_pos
            if cell is not None and cell.type == 'goal':
                terminated = True

            self.agent.move_agent(self.agent_pos)
        
        # Get the action and move the agent
        if action == self.actions.right:
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
            self.agent_dir = 0
        elif action == self.actions.down:
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
            self.agent_dir = 1
        elif action == self.actions.left:
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
            self.agent_dir = 2
        elif action == self.actions.up:
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
            self.agent_dir = 3
        
        # Move agent and check if goal reached
        self.agent_previous_pos = self.agent_pos
        move_and_check_goal(new_pos)
        
        reward = self._reward()

        obs = self.gen_obs()

        self.agent.update_beliefs(obs)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, {}
    
    def _reward(self) -> float:
        """
        Compute the reward based on the belief state about the goal's location
        """

        beliefs = self.agent.get_goal_belief_state()
        current_position = self.agent_pos
        previous_position = self.agent_previous_pos

        if (current_position == self.goal_pos):
            return 100

        def belief_potential(pos):
            """
            Calculate the potential of the current position based on belief state.
            Higher belief values at nearby locations should contribute more to the potential.
            """
            total_belief = 0
            for i in range(self.width):
                for j in range(self.height):
                    belief = beliefs[i, j]
                    distance = abs(i - pos[0]) + abs(j - pos[1])
                    total_belief += belief / (distance + 1)
            return total_belief
        
        # Calculate the potential at the current position
        potential_current = belief_potential(current_position)
        potential_previous = belief_potential(previous_position)
        
        reward = potential_current - potential_previous

        return reward
    
    @staticmethod
    def _gen_mission():
        return 'Reach the goal'
    
    def _gen_grid(self, width, height):
        
        # Create an empty grid
        self.grid = ModifiedGrid(width, height)

        # Surround the grid with walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = 0
        else:
            self.place_agent()
        
        # If possible, place the initial goal to the area where agent cant see

        self.goal_pos = self.place_obj(Goal())

        self.mission = "Reach the goal"

    def initialize_agent(self):

        agent = Agent(self.width, self.height, self.agent_pos, self.agent_view_size)
        agent.initialize_belief_state(mode_densities = self.mode_densities, mode_positions = self.mode_positions)
        obs = self.gen_obs()
        agent.update_beliefs(obs)

        return agent
    
    def get_view_exts(self, agent_view_size):
        '''
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        '''

        agent_view_size = agent_view_size or self.agent_view_size

        topX = self.agent_pos[0] - agent_view_size // 2
        topY = self.agent_pos[1] - agent_view_size // 2

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY
    
    def _get_outside_view_indices(self):
        outside_view_indices = []

        topX, topY, botX, botY = self.get_view_exts(self.agent_view_size)

        for i in range(self.width):
            for j in range(self.height):
                if not (topX <= i < botX and topY <= j < botY):
                    cell = self.grid.get(i,j)

                    if cell is None or cell.type != 'wall':
                        outside_view_indices.append((i,j))
        
        return outside_view_indices


    
    def gen_obs_grid(self, agent_view_size=None):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        if agent_view_size is None, self.agent_view_size is used
        """

        topX, topY, _, _ = self.get_view_exts(agent_view_size)

        agent_view_size = agent_view_size or self.agent_view_size

        grid = self.grid.slice(topX, topY, agent_view_size, agent_view_size)

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(
                agent_pos=(agent_view_size // 2, agent_view_size // 2)  # Center the agent
            )
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = (grid.width // 2, grid.height // 2)  # Center the agent
        if self.carrying:
            grid.set(*agent_pos, self.carrying)
        else:
            grid.set(*agent_pos, None)

        return grid, vis_mask
    
    def gen_obs(self):
        '''
        Generate the agent's view (partially observable low-resolution encoding)
        '''

        grid, vis_mask = self.gen_obs_grid()

        #Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        #Observations are disctionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's position in the grid
        obs = {"image" : image, "agent_pos" : self.agent_pos}

        return obs
    
    def get_full_render(self, highlight, tile_size):
        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the top-left corner of the agent's view area
        top_left = (
            self.agent_pos[0] - self.agent_view_size // 2,
            self.agent_pos[1] - self.agent_view_size // 2
        )

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(self.agent_view_size):
            for vis_i in range(self.agent_view_size):
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell relative to the centered view
                abs_i, abs_j = top_left[0] + vis_i, top_left[1] + vis_j

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            self.agent_pos,  # Centered position
            self.agent_dir,  # Agent direction (invariant)
            highlight_mask=highlight_mask if highlight else None,
        )

        return img
    
    
    def copy(self):
        new_env = GridWorld(self.width - 2, start_pos=self.agent_pos, agent_view_size=self.agent_view_size)
        new_env.grid = self.grid.copy()
        new_env.step_count = self.step_count
        new_env.agent_dir = self.agent_dir
        new_env.agent_pos = self.agent_pos
        new_env.goal_pos = self.goal_pos
        new_env.agent = self.agent.copy()
        return new_env