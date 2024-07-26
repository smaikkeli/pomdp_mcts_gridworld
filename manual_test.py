import numpy as np

from src.grid_world import GridWorld
from src.manual_control import NewManualControl

def main():
    np.set_printoptions(precision=4, suppress=True)
    
    grid_size = 3
    view_size = 3

    assert view_size % 2 == 1, "View size must be odd"
    
    env = GridWorld(render_mode = "human", size = grid_size, agent_view_size = view_size, mode_densities = [0.8, 0.1], mode_positions=[(0,0), (4,4)])

    #Manual controls
    '''
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