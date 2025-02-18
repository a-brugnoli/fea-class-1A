import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib import rcParams
# from IPython.display import HTML
import numpy as np
from math import pi

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
  

def animate_mode(coordinates, mode_shape, omega_mode):
    fig, ax = plt.subplots()
    ax.set_xlim(0, max(coordinates))
    max_ampl = max(max(mode_shape), abs(min(mode_shape)))
    ax.set_ylim(- 0.2 - max_ampl, + 0.2 + max_ampl)
    line, = ax.plot(coordinates, mode_shape, 'b', lw=2)

    period = pi/omega_mode
    n_times = 200
    t_vec_ms = 1e3*np.linspace(0, 2*period, n_times)
    # Animation function
    def update(t):
        y = mode_shape * np.cos(omega_mode * t)  # Oscillating mode shape
        line.set_ydata(y)
        line.set_label(f'$t= {t:.1f}$ [ms]')
        leg = ax.legend()

        return line, leg

    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=t_vec_ms, interval=50, blit=True)

    return anim


def plot_vertical_displacement(time_step, coordinates, values_dofs):
    configure_matplotlib()
    fig, ax = plt.subplots()

    n_times = values_dofs.shape[1]

    vertical_displacement = values_dofs[::2, :]
    
    step_animation = 100
    interval_frames = time_step * step_animation * 1000
    line, = ax.plot(coordinates, vertical_displacement[:, 0])

    ax.set_xlim(0, max(coordinates))
    ax.set_ylim(np.min(vertical_displacement), np.max(vertical_displacement))
    ax.set_xlabel('$x$')
    ax.set_ylabel('$q$')
    ax.grid(True)
    ax.set_title("Vertical Displacement")

    def update(ii):
        line.set_ydata(vertical_displacement[:, ii])
        line.set_label(f'$t= {ii*time_step:.1f}$ [s]')
        leg = ax.legend()

        return line, leg


    # Create animation
    anim = animation.FuncAnimation(fig, update, frames=range(0, n_times, step_animation), \
                            blit=True, interval=interval_frames)
    
    return anim