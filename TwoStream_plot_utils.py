import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams["font.family"] = "serif"
rcParams["font.serif"] = "Times New Roman"
rcParams['text.usetex'] = True
rcParams['text.latex.preamble'] = [r'\usepackage{amsmath} \usepackage{bm} \usepackage{physics}']

def plot(twostream, downsample=None, fname='', **kwargs):
    '''
    Make a plot of the output from the iterative algorithm, as specified in the
        project spec sheet.

    Args
    ---------------
    twostream            (TwoStream): A TwoStream object that has been iterated.
    downsample (int, default = None): Factor by which to downsample the TwoStream
        output, to make plotting quicker.
    fname     (string, default = ''): Path to save the figure to, if any.
    **kwargs                  (dict): Can be used for various plotting keyword arguments.

    Returns
    ---------------
    fig (plt Figure): Figure object for the matplotlib subplots.
    axl   (plt Axis): Axis for the left subplot.
    axr   (plt Axis): Axis for the right subplot.
    '''
    # Get the data
    z = twostream.z
    num_z = len(twostream.z)
    I_minus = twostream.I_minus
    I_plus = twostream.I_plus
    J = twostream.J
    S = twostream.S
    S_series = twostream.S_series

    if downsample is not None:
        z = z[::downsample]
        I_minus = I_minus[::downsample]
        I_plus = I_plus[::downsample]
        J = J[::downsample]
        S = S[::downsample]
        S_series = [s[::downsample] for s in S_series]

    fig, axes = plt.subplots(figsize=(12.8, 5), ncols=2)
    axl, axr = axes.flatten()

    # Left panel
    axl.plot(z, I_minus, color='blue', label=r'$I_-/B$', alpha=0.6)
    axl.plot(z, I_plus, color='red', label=r'$I_+/B$', alpha=0.6)
    axl.plot(z, J, color='green', label=r'$J/B$', alpha=0.6)
    axl.plot(z, S, color='orange', label=r'$S/B$', alpha=0.6)

    # Right panel
    # If S_series has too many samples, dont plot and label them all
    iterations = np.arange(0, (len(S_series) + 1) * twostream.sample_freq, twostream.sample_freq)
    S_and_iters = list(zip(S_series, iterations)) # Keep track of which iteration corresponds to which source function approx.
    max_S_num = 11 # Can adjust this if needed
    if len(S_series) > max_S_num:
        S_first = S_and_iters[0]
        S_last  = S_and_iters[-1]
        interval = int(len(S_series) / (max_S_num - 2))
        S_middle = S_and_iters[interval:(max_S_num-1)*interval:interval]
        #import pdb; pdb.set_trace()
        S_and_iters = [S_first] + S_middle + [S_last]

    colors = plt.cm.plasma(np.linspace(0., 0.8, len(S_and_iters)))[::-1]
    for i,s_and_iter in enumerate(S_and_iters):
        axr.plot(z, s_and_iter[0], alpha=0.6, color=colors[i], label='{} iterations'.format(s_and_iter[1]))

    # Axis labels
    axl.set_ylabel(r'$Y/B$', fontsize=14)
    axl.set_xlabel(r'$z$', fontsize=14)

    axr.set_ylabel(r'$S/B$', fontsize=14)
    axr.set_xlabel(r'$z$', fontsize=14)

    # Axis limits
    axl.set_ylim([0, 1.1])
    axr.set_ylim([0, 1.1])

    # Legends
    axl.legend(fancybox=True)
    axr.legend(fancybox=True)

    # Panel titles
    spacing_str = r'$N = $' + ' {:.1e}, '.format(num_z)
    epsilon_str = r'$\epsilon =$' + ' {}'.format(twostream.epsilon)
    axl.set_title(spacing_str + epsilon_str, fontsize=14.)
    axr.set_title(spacing_str + epsilon_str, fontsize=14.)

    # Get rid of extra whitespace between subplots
    fig.tight_layout()

    # Save figure if given a filename to save under
    if fname != '':
        fig.savefig(fname)

    plt.show()

    return fig, axl, axr
