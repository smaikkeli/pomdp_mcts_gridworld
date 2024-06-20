import numpy as np

from core.actions import Actions
from manual_control import NewManualControl

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.minigrid_env import MiniGridEnv

from gymnasium import spaces

# Environment class
class GridWorld(MiniGridEnv):
    def __init__(self, size=18, start_pos=None, goal_pos=None, **kwargs):

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            **kwargs,
        )
        
        #Reward range positive and minus infinity
        self.reward_range = (-np.inf, np.inf)
                
        self.agent_start_pos = start_pos
        self.goal_pos = goal_pos

        # Overwrite MiniGrid actions with [up, down, left, right]
        self.actions = Actions

        self.action_space = spaces.Discrete(len(self.actions))

        #Initialize belief states
        
        #Initialize MCTS

    def sample_initial_belief(self):

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
        self.grid = Grid(width, height)

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
        if action == self.actions.left:
            new_pos = (self.agent_pos[0] - 1, self.agent_pos[1])
        elif action == self.actions.right:
            new_pos = (self.agent_pos[0] + 1, self.agent_pos[1])
        elif action == self.actions.up:
            new_pos = (self.agent_pos[0], self.agent_pos[1] - 1)
        elif action == self.actions.down:
            new_pos = (self.agent_pos[0], self.agent_pos[1] + 1)
        
        # Move agent and check if goal reached
        move_and_check_goal(new_pos)

        if self.step_count >= self.max_steps:
            truncated = True

        if self.render_mode == "human":
            self.render()

        obs = self.gen_obs()

        return obs, reward, terminated, truncated, {}


def main():
    env = GridWorld(render_mode = "human", agent_view_size = 5)
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