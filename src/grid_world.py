import numpy as np

from core.actions import Actions
from manual_control import NewManualControl

from core.grid import ModifiedGrid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

###TODO
# - IMPORT THE NEW WORKING GRID CLASS
# - Implement belief state inside GridWorld

# Environment class
class GridWorld(MiniGridEnv):
    def __init__(self, size=18, start_pos=None, goal_pos=None, **kwargs):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            **kwargs,
        )

        self.grid = ModifiedGrid(size, size)

        self.agent_dir = 0
        self.agent_start_pos = start_pos
        self.goal_pos = goal_pos
        
        #Reward range positive and minus infinity
        self.reward_range = (-np.inf, np.inf)

        # Overwrite MiniGrid actions with [up, down, left, right]
        self.actions = Actions

        self.action_space = spaces.Discrete(len(self.actions))

        #Size of belief matrix, not including walls
        #self.beliefs = self._initialize_belief_state()

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

    def _reward(self) -> float:
        """
        Compute the reward to be given upon success
        """

        return 1 - 0.9 * (self.step_count / self.max_steps)
    
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
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        # Place the initial goal
        if self.goal_pos is not None:
            self.put_obj(Goal(), *self.goal_pos)
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"

    # New step function to handle new actions
    def step(self, action: Actions):

        self.step_count += 1

        reward = 0
        terminated = False
        truncated = False

        def move_and_check_goal(new_pos):
            nonlocal reward, terminated
            cell = self.grid.get(*new_pos)
            if cell is None or cell.can_overlap():
                self.agent_pos = new_pos
            if cell is not None and cell.type == 'goal':
                reward = self._reward()
                terminated = True
        
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
        move_and_check_goal(new_pos)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        print(self._get_outside_view_indices())

        return obs, reward, terminated, truncated, {}
    
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
    
class Agent():
    def __init__(self, env):
        self.position = env.agent_pos

        self.goal_beliefs = self._init_goal_belief(env)

    def get_goal_belief_state(self):
        return self.goal_beliefs

    def _init_goal_belief(self, env, mode=0.7):
        beliefs = env._get_outside_view_indices()

        # Choose one index randomly
        goal_index = np.random.choice(len(beliefs))

        # Initialize belief state with zero probability
        belief_state = np.zeros((env.width, env.height))

        # Assign high probability to the chosen goal index
        goal_pos = beliefs[goal_index]
        belief_state[goal_pos] = mode

        # Distribute the remaining probability uniformly among the other indices
        remaining_probability = (1 - mode) / (len(beliefs) - 1)
        for i, pos in enumerate(beliefs):
            if i != goal_index:
                belief_state[pos] = remaining_probability

        belief_state /= np.sum(belief_state)

        print(belief_state)

        return belief_state

    def update_beliefs(self):
        pass


def main():
    env = GridWorld(render_mode = "human", size = 10, agent_view_size = 5)
    #agent = Agent(env)
    #Manual control
    mc = NewManualControl(env)
    mc.start()

    #Make simulation loop
    '''
    done = False
    obs = env.reset()
    #Get state
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(obs)
    '''


if __name__ == "__main__":
    main()