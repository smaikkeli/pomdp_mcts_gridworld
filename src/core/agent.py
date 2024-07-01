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

        #Start with uniform belief state
        self.goal_beliefs = np.ones((width, height)) / (width * height)

    def get_goal_belief_state(self):
        return self.goal_beliefs
    
    def set_position(self, pos):
        self.position = pos

    def initialize_belief_state(self, goal_densities = [0.7, 0.2]):
        '''
        Initializes the agents belief about the goal position
        Arguments:
            - unobserved_area: list of tuples, the indices of the unobserved area
            - goal_densities: list of floats, the density of the modes (goals) of the distribution

        Returns:
            - beliefs: numpy array, the agents belief state
        '''
        
        unobserved_area = self.get_outside_view_indices()

        beliefs = np.zeros((self.height, self.width), dtype = np.float64)
        num_of_goals = len(goal_densities)
        
        #There must always be space for setting the belief goals
        #outside the agents view
        assert num_of_goals <= len(unobserved_area)
        
        #Choose distinct indices for the modes
        chosen_indices = np.random.choice(len(unobserved_area), num_of_goals, replace = False)
        
        for i, density in enumerate(goal_densities):
            goal_index = chosen_indices[i]
            goal_pos = unobserved_area[goal_index]
            beliefs[goal_pos] = density
            
        #Generate a probability distribution of length of unobserved area,
        #where num_of_goals are the modes of the distribution, and the rest
        #of the density is spread evenly across rest of the indices

        remaining_density = 1 - np.sum(beliefs)
        remaining_indices = [i for i in range(len(unobserved_area)) if i not in chosen_indices]
        
        if remaining_indices:
            equal_density = remaining_density / len(remaining_indices)
            for i in remaining_indices:
                beliefs[unobserved_area[i]] += equal_density
        
        self.goal_beliefs = beliefs
    
    ##How to update the beliefs given a new observation?
    ##The observations are deterministic, so the agent can be fully certain
    ##What is needed to update with bayes?

    ## How to model the probability of observing a goal state?
    ## When the agent moves, the distribution is updated such that the density is moved to the unobserved area
    ## Until the the agent finds the goal, then the density is exactly at the goal position
    def move_and_update_beliefs(self, position, obs):
        '''
        Update the agent's belief state given an observation
        Arguments:
            - obs: observations from the environment
        '''
        #Get the agent's position
        self.position = position
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
            #Adjust the position to the grid
            goal_pos = (topx + goal_pos[0][0], topy + goal_pos[1][0])
            new_beliefs = np.zeros((self.width, self.height))
            new_beliefs[goal_pos] = 1

        #Normalization constant needs to be calculated
        #The prior is the belief state before the observation
        #The likelihood is the observation
        #The posterior is the updated belief state
        normalization_constant = np.sum(new_beliefs)
        new_beliefs /= (normalization_constant + 1e-10)

        self.goal_beliefs = new_beliefs


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
    
    def get_outside_view_indices(self):
        outside_view_indices = []

        topX, topY, botX, botY = self.get_view_exts(self.view_size)

        for i in range(self.width):
            for j in range(self.height):
                if not (topX <= i < botX and topY <= j < botY):
                    outside_view_indices.append((i,j))
        
        return outside_view_indices
    
    def copy(self):
        new_agent = Agent(self.width, self.height, self.position, self.view_size)
        new_agent.goal_beliefs = self.goal_beliefs.copy()
        return new_agent



        
