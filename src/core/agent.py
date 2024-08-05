import numpy as np

from minigrid.core.constants import OBJECT_TO_IDX

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
    
    def move_agent(self, pos):
        self.position = pos

    def initialize_belief_state(self, mode_densities = [0.4, 0.3], mode_positions = None):
        '''
        Initializes the agent's belief about the goal position
        Arguments:
            - mode_densities: list of floats, the density of the modes (goals) of the distribution
            - mode_positions: list of tuples, the positions of the modes (goals) of the distribution
        Returns:
            - beliefs: numpy array, the agent's belief state
        '''
        
        unobserved_area = self.get_outside_view_indices()

        beliefs = np.zeros((self.height, self.width), dtype=np.float64)
        num_of_goals = len(mode_densities)
        
        if mode_positions is None:
            # Choose distinct indices for the modes from unobserved area
            chosen_indices = np.random.choice(len(unobserved_area), min(num_of_goals, len(unobserved_area)), replace=False)
            mode_positions = [tuple(unobserved_area[i]) for i in chosen_indices]
        else:
            # Use mode_positions as provided, without filtering
            mode_positions = [tuple(pos) for pos in mode_positions[:num_of_goals]]
        
        # Set beliefs for mode positions
        for pos, density in zip(mode_positions, mode_densities[:len(mode_positions)]):
            beliefs[pos] = density

        # Calculate remaining density and positions
        remaining_density = max(0, 1 - np.sum(beliefs))
        all_positions = set((x, y) for x in range(self.height) for y in range(self.width))
        remaining_positions = list(all_positions - set(mode_positions))
        
        if remaining_positions:
            equal_density = remaining_density / len(remaining_positions)
            for pos in remaining_positions:
                beliefs[pos] += equal_density
        elif remaining_density > 0:
            # If no remaining positions but there's remaining density, distribute it equally among mode positions
            equal_density = remaining_density / len(mode_positions)
            for pos in mode_positions:
                beliefs[pos] += equal_density
        
        # Normalize beliefs to ensure they sum to 1
        total_belief = np.sum(beliefs)
        if total_belief > 0:
            beliefs /= total_belief
        
        self.goal_beliefs = beliefs
        return beliefs
    
    def stationary_belief_state(self, goal_position):
        '''
        For testing purposes, initializes belief to the goal
        '''
        
        self.goal_beliefs = np.zeros((self.width, self.height), dtype = np.float32)
        self.goal_beliefs[goal_position] = 1
    
    def make_goal_mask(self, obs):
        #Extract objects from the observation
        image = obs['image'][:,:,0]
        return np.array(image == OBJECT_TO_IDX['goal'])

    def update_beliefs(self, obs):
        '''
        Update the agent's belief state given an observation
        Arguments:
            - obs: observations from the environment
        '''

        mask = self.make_goal_mask(obs)
        view_size = mask.shape[0]

        topx, topy, botx, boty = self.get_view_exts(view_size)
        
        # Slice the mask from dimensions where it goes over the grid
        if topx < 0:
            mask = mask[-topx:]
            topx = 0
        if topy < 0:
            mask = mask[:, -topy:]
            topy = 0
        if botx > self.width:
            mask = mask[:self.width - topx]
            botx = self.width
        if boty > self.height:
            mask = mask[:, :self.height - topy]
            boty = self.height
            

        #The likelihood of observing the goal is 1 if the goal is in the agents view
        #Otherwise 0
        
        #No goal found
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
        return new_agent


        
