import os
import time
from datetime import datetime


def time_it(f):
    time_it.active = 0

    def tt(*args, **kwargs):
        time_it.active += 1
        t0 = time.time()
        tabs = '\t' * (time_it.active - 1)
        name = f.__name__
        print('{tabs}Executing <{name}>'.format(tabs=tabs, name=name))
        res = f(*args, **kwargs)
        print('{tabs}Function <{name}> execution time: {time:.3f} seconds'.format(
            tabs=tabs, name=name, time=time.time() - t0))
        time_it.active -= 1
        return res

    return tt


def load_plot_config(use_latex=False):
    import matplotlib
    import matplotlib.pyplot as plt
    if use_latex:
        matplotlib.rc('text', usetex=True)
    TICK_SIZE = 2

    SMALL_SIZE = 8
    MEDIUM_SIZE = 12
    BIGGER_SIZE = 14

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rc('xtick', direction='out')
    plt.rc('ytick', direction='out')
    plt.rc('xtick', top=False)
    plt.rc('ytick', right=False)
    plt.rc('axes', labelpad=5)
    plt.rc('axes.spines', right=False)
    plt.rc('axes.spines', top=False)
    plt.rc('patch', facecolor='None')
    plt.rc('axes', facecolor='None')
    plt.rc('axes', linewidth=1)
    plt.rc('ytick.major', size=TICK_SIZE)
    plt.rc('xtick.major', width=1)
    plt.rc('xtick.major', size=TICK_SIZE)
    plt.rc('ytick.major', width=1)

    plot_options = Options()
    plot_options.line_width = 2
    plot_options.scatter_size = 15
    plot_options.figsize = (13, 5)

    return plot_options


class Options(object):

    def __str__(self):
        return str(self.__dict__)


def check_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def get_run_datetime():
    """Return the date and time string when the script is run."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")