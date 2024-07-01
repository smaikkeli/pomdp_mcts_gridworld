import numpy as np
from abc import ABC, abstractmethod

class MCTSNode(ABC):
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value = 0
        self.untried_actions = list(range(state.action_space.n))
        
    def is_terminal(self):
        #Check if goal reached or max steps exceeded
        return self.state.step_count >= self.state.max_steps or self.state.goal_pos == self.state.agent_pos
    
    def is_fully_expanded(self):
        return len(self.untried_actions) == 0
    
    def best_child(self, c_param=1.414):
        #ucb
        choices_weights = [
            (c.value / (c.visits + 1e-5)) + c_param * np.sqrt((2 * np.log(self.visits + 1)) / (c.visits + 1e-5))
            for action, c in self.children.items()
        ]
        
        return self.children[max(self.children, key=lambda action: choices_weights[list(self.children.keys()).index(action)])]

    def expand(self):
        action = self.untried_actions.pop()
        next_state = self.state.copy()
        next_state.step(action)
        child_node = MCTSNode(next_state, parent=self, action = action)
        self.children[action] = child_node
        return child_node
    
class MCTS:
    def __init__(self, exploration_weight = 2):
        self.exploration_weight = exploration_weight
        
    def search(self, initial_state, num_simulations, max_depth):
        root = MCTSNode(initial_state)
        
        for _ in range(num_simulations):
            node = root
            depth = 0
            
            #Selection
            while not node.is_terminal() and node.is_fully_expanded():
                node = node.best_child(c_param=self.exploration_weight)
                depth += 1
            
            #Expansion
            if not node.is_terminal() and not node.is_fully_expanded() and depth < max_depth:
                node = node.expand()
                depth += 1
                
            #Simulation
            reward = self.simulate(node.state, max_depth - depth)
            
            while node is not None:
                node.visits += 1
                node.value += reward
                node = node.parent
            
        return max(root.children, key=lambda a: root.children[a].visits)
    
    def simulate(self, state, max_steps):
        current_state = state.copy()
        for _ in range(max_steps):
            if current_state.goal_pos == current_state.agent_pos:
                break
            
            #Take random step
            action = current_state.action_space.sample()
            current_state.step(action)
            
        return state._reward()
        

def choose_action(env, num_simulations = 1000, max_depth=10):
    mcts = MCTS()
    env = env.copy()
    return mcts.search(env, num_simulations, max_depth)