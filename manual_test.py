import numpy as np

from src.grid_world import GridWorld
from src.core.agent import Agent
from src.manual_control import NewManualControl

from minigrid.wrappers import SymbolicObsWrapper


def main():
    np.set_printoptions(precision=4, suppress=True)
    env = GridWorld(render_mode = "human", size = 6, agent_view_size = 3)
    #Manual controls
    mc = NewManualControl(env)
    mc.start()

    #Make simulation loop
    '''
    done = False
    obs = env.reset()
    #Observations are independent of the agent direction
    #Always from top to bottom, left to right
    #How to handle??


    #Get state
    while not done:
        #Action chosen by the agent?? Should the agent be initialized in main?
        #action = agent.get_action(obs)

        action = env.action_space.sample()
        obs, reward, done, _, info = env.step(action)
        print(obs)
    '''

if __name__ == "__main__":
    main()