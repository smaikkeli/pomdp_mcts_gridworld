import numpy as np

from src.grid_world import GridWorld
from src.manual_control import NewManualControl

def main():
    '''
    Manual test
    
    Here you can play around the parameters to see how the agent behaves
    Comment initial_goal_pos to change the goal position in each run
    Comment fully observable to use non stationary belief state
    '''
    np.set_printoptions(precision=4, suppress=True)
    
    grid_size = 6
    view_size = 3

    assert view_size % 2 == 1, "View size must be odd"
    
    init_goal_pos = tuple(np.random.randint(1, grid_size, size = 2))
    env = GridWorld(render_mode = "human", size = grid_size, agent_view_size = view_size, init_goal_pos=init_goal_pos, mode_densities = [0.2,0.3], mode_positions = [(1,1), (1,2)])
    env.set_fully_observable()
    
    #Manual controls
    #mc = NewManualControl(env)
    #mc.start()
    
    #Get state
    while True:

        done = False
        obs = env.reset()
        beliefs = env.agent.get_goal_belief_state()
        print(beliefs)

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