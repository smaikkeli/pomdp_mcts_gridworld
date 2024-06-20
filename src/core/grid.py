import numpy as np
from minigrid.core.grid import Grid as BaseGrid

class ModifiedGrid(BaseGrid):
  def __init__(self, width: int, height: int):
    super().__init__(
      width = width,
      height = height,
    )

  def process_vis(self, agent_pos: tuple[int, int]):
    mask = np.zeros(shape=(self.width, self.height), dtype=bool)
    mask[agent_pos[0], agent_pos[1]] = True

    queue = [(agent_pos[0], agent_pos[1])]  # Use a queue to propagate visibility

    # Process visibility using BFS (breadth-first search)
    while queue:
        x, y = queue.pop(0)

        # Explore neighbors: right, left, down, up, diagonals
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and not mask[nx, ny]:
                cell = self.get(nx, ny)
                if cell and not cell.see_behind():
                    continue
                mask[nx, ny] = True
                queue.append((nx, ny))

    # Set cells outside the mask to None
    for j in range(self.height):
        for i in range(self.width):
            if not mask[i, j]:
                self.set(i, j, None)

    return mask