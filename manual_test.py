import numpy as np

from src.grid_world import GridWorld
from src.core.agent import Agent
from src.manual_control import NewManualControl

from minigrid.wrappers import SymbolicObsWrapper


def main():
    np.set_printoptions(precision=4, suppress=True)
    
    grid_size = 7
    view_size = 3
    
    env = GridWorld(render_mode = "human", size = grid_size, agent_view_size = view_size)
    #Manual controls
    #mc = NewManualControl(env)
    #mc.start()

    #Make simulation loop
    done = False
    obs = env.reset()
    #Observations are independent of the agent direction
    #Always from top to bottom, left to right
    #How to handle??


    #Get state
    while not done:
        #Action chosen by the agent?? Should the agent be initialized in main?
        action = env.choose_action(100)
        obs, reward, done, _, info = env.step(action)
        beliefs = env.agent.get_goal_belief_state()
        print(beliefs)
        #print(obs)

if __name__ == "__main__":
    main()