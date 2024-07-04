import numpy as np

from minigrid.core.constants import IDX_TO_OBJECT, DIR_TO_VEC, OBJECT_TO_IDX

'''
Agent class is used within the GridWorld class to represent the agents beliefs
The agent knows its position, and has a belief state of the goal position
'''
class Agent():
    def __init__(self, width, height, pos, view_size):
        self.width = width
        self.height = height
        self.view_size = view_size
        self.position = pos
        self.visited_positions = np.zeros((width, height), dtype = int)
        self.visited_positions[pos] = 1

        #Keep track of the goal densities, and their positions
        self.mode_densities = {}

        #Start with uniform belief state
        self.goal_beliefs = np.ones((width, height)) / (width * height)

    def get_mode_densities(self):
        return self.mode_densities

    def get_goal_belief_state(self):
        return self.goal_beliefs
    
    def move_agent(self, pos):
        self.position = pos
        self.visited_positions[pos] += 1

    def initialize_belief_state(self, mode_densities = [0.7, 0.2]):
        '''
        Initializes the agents belief about the goal position
        Arguments:
            - mode_densities: list of floats, the density of the modes (goals) of the distribution

        Returns:
            - beliefs: numpy array, the agents belief state
        '''
        
        unobserved_area = self.get_outside_view_indices()

        beliefs = np.zeros((self.height, self.width), dtype = np.float64)
        num_of_goals = len(mode_densities)
        
        assert num_of_goals <= len(unobserved_area)
        
        #Choose distinct indices for the modes
        chosen_indices = np.random.choice(len(unobserved_area), num_of_goals, replace = False)
        
        for i, density in enumerate(mode_densities):
            goal_index = chosen_indices[i]
            goal_pos = unobserved_area[goal_index]
            beliefs[goal_pos] = density

            self.mode_densities[goal_pos] = density

        remaining_density = 1 - np.sum(beliefs)
        remaining_indices = [i for i in range(len(unobserved_area)) if i not in chosen_indices]
        
        if remaining_indices:
            equal_density = remaining_density / len(remaining_indices)
            for i in remaining_indices:
                beliefs[unobserved_area[i]] += equal_density
        
        self.goal_beliefs = beliefs

    def update_beliefs(self, obs):
        '''
        Update the agent's belief state given an observation
        Arguments:
            - obs: observations from the environment
        '''

        image = obs['image'][:,:,0]
        view_size = image.shape[0]

        #Make a mask of the goal position
        mask = np.array(image == OBJECT_TO_IDX['goal'])

        topx = self.position[0] - view_size // 2
        topy = self.position[1] - view_size // 2

        botx = topx + view_size
        boty = topy + view_size

        #The likelihood of observing the goal is 1 if the goal is in the agents view
        #Otherwise, the likelihood is 0
        if np.sum(mask) == 0:
            new_beliefs = self.goal_beliefs.copy()
            new_beliefs[topx:botx, topy:boty] = 0
        else:
            goal_pos = np.where(mask)
            goal_pos = (topx + goal_pos[0][0], topy + goal_pos[1][0])
            new_beliefs = np.zeros((self.width, self.height))
            new_beliefs[goal_pos] = 1

        normalization_constant = np.sum(new_beliefs)
        new_beliefs /= (normalization_constant + 1e-10)

        #update the mode densities
        for goal_pos in self.mode_densities:
            self.mode_densities[goal_pos] = new_beliefs[goal_pos]

        self.goal_beliefs = new_beliefs
    
    def get_outside_view_indices(self):
        outside_view_indices = []

        topX, topY, botX, botY = self.get_view_exts(self.view_size)

        for i in range(self.width):
            for j in range(self.height):
                if not (topX <= i < botX and topY <= j < botY):
                    outside_view_indices.append((i,j))
        
        return outside_view_indices
    
    def get_view_exts(self, agent_view_size):
        '''
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        if agent_view_size is None, use self.agent_view_size
        '''

        agent_view_size = agent_view_size or self.view_size

        topX = self.position[0] - agent_view_size // 2
        topY = self.position[1] - agent_view_size // 2

        botX = topX + agent_view_size
        botY = topY + agent_view_size

        return topX, topY, botX, botY
    
    def copy(self):
        new_agent = Agent(self.width, self.height, self.position, self.view_size)
        new_agent.goal_beliefs = self.goal_beliefs.copy()
        new_agent.visited_positions = self.visited_positions.copy()
        return new_agent



        
