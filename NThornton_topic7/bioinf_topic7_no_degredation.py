from functools import wraps
import numpy as np
from numba import jit  ## "just in time" -- speeds shit up
from numba import prange  ## for parallelization
import matplotlib
matplotlib.use('TkAgg')  ## renderer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

# Constants
# ========
FPS = 60
GRID_W = 200
GRID_H = 200
ZEROS = np.zeros((GRID_W, GRID_H))
# ========


def timefn(fn):
    """wrapper to time the enclosed function"""

    @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        print("@timefn: {} took {} seconds".format(fn.__name__, t2 - t1))
        return result

    return measure_time


def initialize_grid():
    """
Generate start grid, with species 1, 2, and 3 populating 1% of the space
Return this grid to main function
    """
    grid = ZEROS.copy()
    g_len = grid.shape[0]
    for x in range(g_len):
        for y in range(g_len):
            state = np.random.random()
            if state <= 0.1:
                grid[y][x] = np.random.randint(1, 4)
    return grid


# @jit(parallel=True, nopython=True, fastmath=True)
@jit
def update_grid(grid):
    """Calculate the next iteration of the grid using the previous iteration

    :param grid: 2D grid of dead/alive cells
    :returns: 2D grid of dead/alive cells

    """
    r_ab = 2     # radius of dispersal for antibiotics
    r_prop = 4   # radius of dispersal for cell propagation

    g_len = grid.shape[0]
    grid_new = ZEROS.copy()
    open_space = []

    for y in prange(g_len):
        for x in range(g_len):
            sp_present = np.zeros((1,3))  # accumulator for neighboring cells; next loop sums neighboring cells
            if not grid[y][x]:
                open_space.append((y,x))  # if space is empty, save it to propagate a cell into later
                continue                  # if cell exists; check adjacent populations

            for j in range(-r_ab, (r_ab + 1), 1):
                for i in range(-r_ab, (r_ab + 1), 1):          # cycle through 5x5 grid and checks which species present
                    if (np.abs(i) + np.abs(j)) > (r_ab + 1):   # limits these loops to circle
                        continue
                    if grid[(y + j) % g_len][(x + i) % g_len] == 1:
                        sp_present[0][0] = 1
                    elif grid[(y + j) % g_len][(x + i) % g_len] == 2:
                        sp_present[0][1] = 1
                    elif grid[(y + j) % g_len][(x + i) % g_len] == 3:
                        sp_present[0][2] = 1

            if grid[y][x] == 1 and sp_present[0][2] == 1:      # determines outcome for cell:
                grid_new[y][x] = 0
            elif grid[y][x] == 2 and sp_present[0][0] == 1:
                grid_new[y][x] = 0
            elif grid[y][x] == 3 and sp_present[0][1] == 1:
                grid_new[y][x] = 0
            else:
                grid_new[y][x] = grid[y][x]

    grid = grid_new.copy()

    for p in range(len(open_space)):   # iterate through list of empty spots and determine what cell type grows
        sp_present = np.zeros((1, 3))
        for j in range(-r_prop, (r_prop + 1), 1):
            for i in range(-r_prop, (r_prop + 1), 1):
                if (np.abs(i) + np.abs(j)) > (r_prop + 1):
                    continue
                if grid[(open_space[p][0] + j) % g_len][(open_space[p][1] + i) % g_len] == 1:
                    sp_present[0][0] = 1
                elif grid[(open_space[p][0] + j) % g_len][(open_space[p][1] + i) % g_len] == 2:
                    sp_present[0][1] = 1
                elif grid[(open_space[p][0] + j) % g_len][(open_space[p][1] + i) % g_len] == 3:
                    sp_present[0][2] = 1

        if sp_present.any():
            grid_new[open_space[p][0]][open_space[p][1]] = np.random.choice(np.nonzero(sp_present)[1]) + 1

    return grid_new


def update(frame, im, grid):
    """function which is called each tick of the animation

    :param frame: The current frame index
    :type frame: int
    :param im: The image being updated
    :type im: matplotlib imshow
    :param grid: 2D grid of dead/alive cells
    :type grid: np.array
    :returns: updated image
    """
    new_grid = update_grid(grid)
    im.set_array(new_grid)
    grid[:] = new_grid[:]
    return (im, )  # returns a tuple


# Function for generating population counts; called by draw_N()
def value_count(grid):
    count = np.bincount(grid.flatten().astype('int'))
    return count


@timefn
def draw_N(grid, N):  # this runs grid_update() without rendering image and runs value_count()
    cell_count = np.zeros((N,3))
    for i in range(N):
        grid[:] = update_grid(grid)[:]
        count = value_count(grid)
        cell_count[N-1][:] = count[1:]
    return cell_count, grid


@timefn
def main():
    grid = initialize_grid()
    # draw_N(grid, 200)
    # update_grid.parallel_diagnostics(level=1)
    fig = plt.figure()
    im1 = plt.imshow(
        grid,
        cmap="tab10",
        aspect="equal",
        interpolation="none"  # this sets grid resolution at screen resolution
    )
    ani = FuncAnimation(
        fig,
        update,
        fargs=(im1, grid),
        frames=FPS * 50,
        interval=1000 / FPS,
        repeat=False,  # stop rendering once hits max number of frames
        blit=True,     # prevents redrawing pixels that haven't changed
    )
    plt.show()
    return ani


if __name__ == "__main__":
    ani, cell_count = main()
    ## ani.save("./test.mpg")
