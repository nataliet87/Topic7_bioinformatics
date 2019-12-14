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
            if state <= 0.01:
                grid[y][x] = np.random.randint(1, 4)
    return grid


## can use numpy convolution to flatten some of these loops!?!
## also; try to hoist logic above loops or leave it all inside to make parallelization easier (how about sorting first??)
# @jit(parallel=True, nopython=True, fastmath=True)
@jit

def update_grid(grid):
    """Calculate the next iteration of the grid using the previous grid

    :param grid: 2D grid of dead/alive cells
    :returns: 2D grid of dead/alive cells

    """
    g_len = grid.shape[0]
    grid_new = ZEROS.copy()
    open_space = []

    r_prop = 5   # radius of dispersal for cell propagation
    r_ab = 2     # radius of dispersal for antibiotics
    r_deg = 2    # radius of dispersal for antibiotics degrader
    r_survival = np.maximum(r_ab, r_deg)

    for y in prange(g_len):                # iterate through grid
        for x in range(g_len):
            total_ab = np.zeros((1,3))     # accumulator for cells w/in antibiotic radius
            total_deg = np.zeros((1,3))    # accumulator for cells w/in antibiotic degrader radius
            if not grid[y][x]:
                open_space.append((y,x))   # generates list of coordinates at which cells will grow (empty spots)
                continue

            for j in range(-r_survival, (r_survival + 1), 1):        # cycle through antibiotic/degrader radius
                for i in range(-r_survival, (r_survival + 1), 1):
                    if (np.abs(i) + np.abs(j)) < (r_ab + 1):                 # take cell tally for antibiotic producers
                        if grid[(y + j) % g_len][(x + i) % g_len] == 1:
                            total_ab[0][0] += 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 2:
                            total_ab[0][1] += 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 3:
                            total_ab[0][2] += 1

                    elif (np.abs(i) + np.abs(j)) < (r_deg + 1):              # take cell tally for antibiotic degradors
                        if grid[(y + j) % g_len][(x + i) % g_len] == 1:
                            total_deg[0][0] += 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 2:
                            total_deg[0][1] += 1
                        elif grid[(y + j) % g_len][(x + i) % g_len] == 3:
                            total_deg[0][2] += 1

                if grid[y][x] == 1 and (total_deg[0][1] >= total_ab[0][2]):  # determine outcomes for cells
                    grid_new[y][x] = 1
                elif grid[y][x] == 2 and (total_deg[0][2] >= total_ab[0][0]):
                    grid_new[y][x] = 2
                elif grid[y][x] == 3 and (total_deg[0][0] >= total_ab[0][1]):
                    grid_new[y][x] = 3
                else:
                    grid_new[y][x] = 0

    grid = grid_new.copy()

    for p in range(len(open_space)):    # iterate through list of empty spots and determine what cell type grows
        sp_present = np.zeros((1, 3))
        for j in range(-r_prop, (r_prop + 1), 1):
            for i in range(-r_prop, (r_prop + 1), 1):
                if (np.abs(i) + np.abs(j)) > (r_prop + 1):
                    continue
                elif grid[(open_space[p][0] + j) % g_len][(open_space[p][1] + i) % g_len] == 1:
                    sp_present[0][0] = 1
                elif grid[(open_space[p][0] + j) % g_len][(open_space[p][1] + i) % g_len] == 2:
                    sp_present[0][1] = 1
                elif grid[(open_space[p][0] + j) % g_len][(open_space[p][1] + i) % g_len] == 3:
                    sp_present[0][2] = 1

        if sp_present.any():
            grid_new[open_space[p][0]][open_space[p][1]] = np.random.choice(np.nonzero(sp_present)[1]) + 1

    return grid_new


# Function for generating population counts; called by draw_N()
def value_count(grid):
    count = np.bincount(grid.flatten().astype('int'))
    return count


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
    return (im, )


@timefn
def draw_N(grid, N):  # this just runs the grid update without rendering an image; useful for optimization
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
    im = plt.imshow(
        grid,
        cmap="tab10",
        aspect="equal",
        interpolation="none"   # this sets grid resolution at screen resolution
    )
    ani = FuncAnimation(
        fig,
        update,
        fargs=(im, grid),
        frames=FPS * 50,
        interval=1000 / FPS,
        repeat=False,  # stop rendering once you've hit max number of frames
        blit=True,     # prevents redrawing pixels that haven't changed
    )
    plt.show()
    return ani


if __name__ == "__main__":
    ani = main()
    ## ani.save("./test.mpg")
