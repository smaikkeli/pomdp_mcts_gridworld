import numpy as np
import mcts

from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Goal
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from gymnasium import spaces


#Environment class
class GridWorld(MiniGridEnv):
    def __init__(self, size=18, start_pos = None, start_dir = 0, goal_pos = None, **kwargs,):
        self.agent_start_pos = start_pos
        self.agent_start_dir = start_dir
        self.goal_pos = goal_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return 'Reach the goal'
    
    def _gen_grid(self, width, height):
        #Create an empty grid
        self.grid = Grid(width, height)

        #Surround the grid with walls
        self.grid.wall_rect(0, 0, width, height)

        #Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
        
        #Place the initial goal
        if self.goal_pos is not None:
            self.put_obj(Goal(), *self.goal_pos)
        else:
            self.place_obj(Goal())

        self.mission = "Reach the goal"


#POMDP MCTS Agent
class MCTSAgent():
    def __init__(self, env, num_simulations = 100):
        self.env = env
        self.num_simulations = num_simulations

    def getPossibleActions(self, state):
        return self.env.action_space[0:3]
        

    def simulate(self):
        pass


    
def main():
    env = GridWorld(render_mode = "rgb_array", agent_view_size = 5)
    #Make simulation loop
    done = False
    obs = env.reset()
    #Get state
    while not done:
        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        


if __name__ == "__main__":
    main()