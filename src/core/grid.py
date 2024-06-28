import numpy as np

from minigrid.core.grid import Grid as BaseGrid

from minigrid.core.constants import OBJECT_TO_IDX
from minigrid.core.world_object import Wall, WorldObj

class ModifiedGrid(BaseGrid):
  def __init__(self, width: int, height: int):
    super().__init__(
      width = width,
      height = height,
    )

  def __eq__(self, other: 'ModifiedGrid') -> bool:
      grid1 = self.encode()
      grid2 = other.encode()
      return np.array_equal(grid2, grid1)

  def __ne__(self, other: 'ModifiedGrid') -> bool:
      return not self == other

  def copy(self) -> 'ModifiedGrid':
      from copy import deepcopy

      return deepcopy(self)
  
  def rotate_left(self) -> 'ModifiedGrid':
    """
    Rotate the grid to the left (counter-clockwise)
    """

    grid = ModifiedGrid(self.height, self.width)

    for i in range(self.width):
        for j in range(self.height):
            v = self.get(i, j)
            grid.set(j, grid.height - 1 - i, v)

    return grid
  
  def slice(self, topX: int, topY: int, width: int, height: int) -> 'ModifiedGrid':
    """
    Get a subset of the grid
    """

    grid = ModifiedGrid(width, height)

    for j in range(0, height):
        for i in range(0, width):
            x = topX + i
            y = topY + j

            if 0 <= x < self.width and 0 <= y < self.height:
                v = self.get(x, y)
            else:
                v = Wall()

            grid.set(i, j, v)

    return grid
  
  def render(
    self,
    tile_size: int,
    agent_pos: tuple[int, int],
    agent_dir: int | None = None,
    highlight_mask: np.ndarray | None = None,
  ) -> np.ndarray:
    """
    Render this grid at a given scale
    :param r: target renderer object
    :param tile_size: tile size in pixels
    """

    if highlight_mask is None:
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

    # Compute the total grid size
    width_px = self.width * tile_size
    height_px = self.height * tile_size

    img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

    # Render the grid
    for j in range(0, self.height):
        for i in range(0, self.width):
            cell = self.get(i, j)

            agent_here = np.array_equal(agent_pos, (i, j))
            assert highlight_mask is not None
            tile_img = ModifiedGrid.render_tile(
                cell,
                agent_dir=agent_dir if agent_here else None,
                highlight=highlight_mask[i, j],
                tile_size=tile_size,
            )

            ymin = j * tile_size
            ymax = (j + 1) * tile_size
            xmin = i * tile_size
            xmax = (i + 1) * tile_size
            img[ymin:ymax, xmin:xmax, :] = tile_img

    return img
  
  @staticmethod
  def decode(array: np.ndarray) -> tuple['ModifiedGrid', np.ndarray]:
      """
      Decode an array grid encoding back into a grid
      """

      width, height, channels = array.shape
      assert channels == 3

      vis_mask = np.ones(shape=(width, height), dtype=bool)

      grid = ModifiedGrid(width, height)
      for i in range(width):
          for j in range(height):
              type_idx, color_idx, state = array[i, j]
              v = WorldObj.decode(type_idx, color_idx, state)
              grid.set(i, j, v)
              vis_mask[i, j] = type_idx != OBJECT_TO_IDX["unseen"]

      return grid, vis_mask


  def process_vis(self, agent_pos: tuple[int, int]):
    mask = np.zeros(shape=(self.width, self.height), dtype=bool)
    mask[agent_pos[0], agent_pos[1]] = True

    queue = [(agent_pos[0], agent_pos[1])]  # Use a queue to propagate visibility

    marked = np.zeros(shape=(self.width, self.height), dtype=bool)

    # Process visibility using BFS (breadth-first search)
    while queue:
        x, y = queue.pop(0)

        # Explore neighbors: right, left, down, up, diagonals
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and not mask[nx, ny] and not marked[nx, ny]:
                cell = self.get(nx, ny)
                if cell and not cell.see_behind():
                    marked[nx, ny] = True
                    continue
                mask[nx, ny] = True
                queue.append((nx, ny))

    # Set cells outside the mask to None
    for j in range(self.height):
        for i in range(self.width):
            if not mask[i, j]:
                self.set(i, j, None)

    return mask