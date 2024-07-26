import numpy as np

from src.utils.sampler import Sampler

def main():
  sampler = Sampler(grid_size = 0, agent_view_size = 5, traj_length = 10)
  
  test_sample = sampler.sample_mode_densities(max_modes = 5, total_density = 0.5)
  assert np.isclose(np.sum(test_sample), 0.5)
  
  test_param = sampler.generate_user_parameters(10)

  
  
  
if __name__ == "__main__":
  main()