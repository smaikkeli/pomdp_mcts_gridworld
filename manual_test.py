import numpy as np

from src.grid_world import GridWorld
from src.core.agent import Agent
from src.manual_control import NewManualControl

from minigrid.wrappers import SymbolicObsWrapper


def main():
    np.set_printoptions(precision=4, suppress=True)
    
    grid_size = 16
    view_size = 7

    assert view_size % 2 == 1, "View size must be odd"
    
    env = GridWorld(render_mode = "human", size = grid_size, agent_view_size = view_size)

    '''
    #Manual controls
    mc = NewManualControl(env)
    mc.start()
    '''
    #Get state
    while True:

        done = False
        obs = env.reset()
        beliefs = env.agent.get_goal_belief_state()

        while not done:
            #Action chosen by the agent?? Should the agent be initialized in main?
            action = env.max_neighboring_reward()
            obs, reward, done, _, info = env.step(action)
            beliefs = env.agent.get_goal_belief_state()
            print(f"reward={reward:.2f}")
            print(beliefs.T)
            #print(obs) 
            
if __name__ == "__main__":
    main()