import numpy as np
import torch
from itertools import permutations

from src.grid_world import GridWorld

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda") if USE_CUDA else torch.device("cpu")

class Sampler():
  def __init__(self, grid_size, agent_view_size, traj_length, fixed_goal = False):
    self.grid_size = grid_size
    self.agent_view_size = agent_view_size
    self.traj_length = traj_length
    self.fixed_goal = fixed_goal

  def generate_batch(self, densities = None):
    '''
    General batch creation process, one should implement this in code
    '''
    
    user_params = self.generate_user_parameters(densities)
    
    num_trajectories = np.random.randint(1, 10)
    
    trajectories = self.generate_user_trajectories(num_trajectories=num_trajectories, user_params=user_params)
    
    xc, yc, xt, yt = self.build_context_and_target(trajectories)
    
    return xc, yc, xt, yt, user_params
  
  def generate_user_parameters(self, densities = None):
    '''
    Generates a set of parameters that define the user's behavior.
    args: 
      mode_densities: list of modes, cumulative sum must not exceed 1
    '''
    
    mode_densities = None
    if densities is None:
      mode_densities = self.sample_mode_densities()
    else:
      mode_densities = densities
    
    mode_positions = np.random.randint(1, self.grid_size, (len(mode_densities), 2))
    init_goal_pos = None
    
    #Sample goal_position that is used if fixed_goal = True
    if self.fixed_goal:
      init_goal_pos = tuple(np.random.randint(1, self.grid_size + 1, size = 2))
    
    return {
      'mode_densities': mode_densities,
      'mode_positions': mode_positions,
      'goal_position': init_goal_pos
    }

  def sample_mode_densities(self, max_modes = 3, total_density = 0.9):
    '''
    Samples a list of mode densities that guide the agent's
    behavior.
    '''
    
    num_modes = np.random.randint(1, max_modes + 1)
    densities = np.random.dirichlet(np.ones(num_modes)) * total_density
    return list(densities)

  def generate_user_trajectories(self, num_trajectories, user_params):
    mode_densities = user_params['mode_densities']
    mode_positions = user_params['mode_positions']
    init_goal_position = user_params['goal_position']
    
    #env = GridWorld(render_mode = "rgb_array", size = self.grid_size, agent_view_size = self.agent_view_size, mode_densities = mode_densities, mode_positions=mode_positions)
    env = GridWorld(render_mode = "rgb_array", size = self.grid_size, agent_view_size = self.agent_view_size, init_goal_pos = init_goal_position, mode_densities = mode_densities, mode_positions = mode_positions)
    
    trajectories = []
    
    while len(trajectories) < num_trajectories:
      trajectory = []
      
      obs = env.reset()
      state = obs[0]['agent_pos']
      
      done = False

      while not done and len(trajectory) < self.traj_length:
        
        #Add possibility for random action
        if np.random.random() < 0.1:
          action = env.action_space.sample()
        else:
          action = env.max_neighboring_reward()

        action_onehot = [0 for _ in range(env.action_space.n + 1)]
        action_onehot[action] = 1

        trajectory.append((state, action_onehot))

        next_obs, _, done, truncated, _ = env.step(action)
        state = next_obs['agent_pos']
        
        done = done or truncated

      trajectory.append((state, action_onehot))

      if len(trajectory) < self.traj_length:
        remaining = self.traj_length - len(trajectory)
        fill_state = [state] * remaining
        
        #Fill with stationary action
        actions = [[0,0,0,0,1]] * remaining
        trajectory += list(zip(fill_state, actions))

      trajectories.append(trajectory[:self.traj_length])
      
    return trajectories  


  def split_trajectory_half(self, trajectory):
    '''
    Splits a trajectory into two halves.
    '''
    half = len(trajectory) // 2
    return trajectory[:half], trajectory[half:]

  def build_context_and_target(self, dataset, device = device):
    '''
    Multiple tasks are created for each of the trajectories in the dataset. Each
    trajectory is chosen as the target, and the rest of the trajectories are 
    permuted to form the context. The permutations are chosen randomly and 
    the trajectories are concatenated to to from one task.
    '''
    num_traj = len(dataset)
    xc, yc, xt, yt = [], [], [], []

    for i in range(num_traj):
        # Pick one as target and split it
        context_part, target_part = self.split_trajectory_half(dataset[i])
        
        #context_part = dataset[i][:-1]
        #target_part = [dataset[i][-1]]

        # Choose the ids of past context trajectories
        past_context_ids = list(range(i)) + list(range(i+1, num_traj))

        # Generate all permutations of past context ids
        all_permutations = list(permutations(past_context_ids))

        #Choose a subset of permutations, one permutation is fed max once
        selected_permutations = np.random.choice(len(all_permutations),
                                                size=min(5, len(all_permutations)),
                                                replace=False)
        
        # Generate multiple tasks from the chosen target set and different context
        for p_idx in selected_permutations:
            p = list(all_permutations[p_idx])
            
            past_contexts = [dataset[j] for j in p]

            full_context = past_contexts + [context_part]

            # Separate states and actions for context and target

            context_s = torch.tensor([state for traj in full_context for state, _ in traj], dtype = torch.float32).to(device)
            context_a = torch.tensor([action for traj in full_context for _, action in traj], dtype = torch.float32).to(device)
            target_s = torch.tensor([state for state, _ in target_part], dtype = torch.float32).to(device)
            target_a = torch.tensor([action for _, action in target_part], dtype = torch.float32).to(device)

            xc.append(context_s)
            yc.append(context_a)
            xt.append(target_s)
            yt.append(target_a)

    xc = torch.stack(xc, dim = 0)
    yc = torch.stack(yc, dim = 0)
    xt = torch.stack(xt, dim = 0)
    yt = torch.stack(yt, dim = 0)

    return xc, yc, xt, yt
