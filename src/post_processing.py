import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
from IPython.display import HTML
import numpy as np

def configure_matplotlib():
    SMALL_SIZE = 18
    MEDIUM_SIZE = 20
    BIGGER_SIZE = 22

    rcParams.update({'figure.autolayout': True, 
                     'text.usetex': True,
                     'text.latex.preamble':r"\usepackage{amsmath}",
                     'legend.loc':'upper right',
                     'font.size': SMALL_SIZE,
                     'axes.titlesize': BIGGER_SIZE,
                     'axes.labelsize': MEDIUM_SIZE,
                     'xtick.labelsize': SMALL_SIZE,
                     'legend.fontsize': SMALL_SIZE,
                     'figure.titlesize': BIGGER_SIZE
                     })
    

def plot_vertical_displacement(time_step, coordinates, values):
    configure_matplotlib()
    fig, ax = plt.subplots()

    n_times = values.shape[1]
    
    step_animation = 10
    interval_frames = time_step * step_animation * 1000
    line1, = ax.plot(coordinates, values[0::2, 0])

    ax.set_xlim(0, max(coordinates))
    ax.set_ylim(np.min(values[0::2, :]), np.max(values[0::2, :]))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$q$')
    ax.grid(True)
    ax.set_title("Vertical Displacement")

    def update(ii):
        line1.set_ydata(values[0::2, ii])
        line1.set_label(f'$t= {ii*time_step:.1f}$ [s]')
        leg = ax.legend()

        return line1, leg


    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=range(0, n_times, step_animation), \
                            blit=True, interval=interval_frames)
    
    plt.show()
