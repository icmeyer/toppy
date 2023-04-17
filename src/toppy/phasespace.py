import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors


from ParticlePhaseSpace import DataLoaders
from ParticlePhaseSpace import PhaseSpace

# https://www.python-graph-gallery.com/86-avoid-overlapping-in-scatterplot-with-2d-density
# Acknowledgement to plotting in ParticlePhaseSpace:
# https://github.com/bwheelz36/ParticlePhaseSpace/blob/main/ParticlePhaseSpace/_ParticlePhaseSpace.py

def make_histogram(weights, x_data, y_data, xgrid, ygrid):
    h, xedges, yedges = np.histogram2d(x_data, y_data,
                                       bins=[xgrid, ygrid], weights=weights)
    h = h/np.sum(np.sum(h))
    h[h == 0] = np.NaN
    return h, xedges, yedges

def plot_hist_2d(ax, histogram, norm):
    pcm = ax.pcolormesh(histogram[1], histogram[2], histogram[0], 
                        cmap='inferno', norm=norm, rasterized=True)
    ax.set_aspect('equal')
    return pcm

        

def custom_pairplot(df, variables):
    fig = plt.figure(figsize=(12,9))
    axd = fig.subplot_mosaic(
        """
        abcj
        defj
        ghij
        """,
        width_ratios = [1,1,1,0.2]
    )
    # To share x axes:
    axd['a'].get_shared_x_axes().join(axd['a'], axd['c'])

    diags = ['a', 'e', 'i']
    offdiags = ['b', 'c', 'd', 'f', 'g', 'h']
    variable_tuples = []
    for i in range(3):
        for j in range(3):
            if i != j:
                variable_tuples.append([variables[i],variables[j]])

    nbins = 40
    histbins = np.linspace(-1, 1, nbins)
    for i, diag in enumerate(diags):
        # axd[diag].hist(df[variables[i]], bins=nbins)
        counts, bins = np.histogram(df[variables[i]], bins=histbins)
        # Normalize the counts (probability mass function)
        counts = counts/np.sum(counts)
        axd[diag].stairs(counts, bins)
        axd[diag].set_xlabel(variables[i])
        axd[diag].set_ylabel('Probability Mass Function')

    nbins = 20
    xlim = [-1, 1]
    ylim = [-1, 1]
    xgrid = np.linspace(xlim[0], xlim[1], nbins)
    ygrid = np.linspace(ylim[0], ylim[1], nbins)
    histograms = []
    vmin = 0
    vmax = 0
    for i, offdiag in enumerate(offdiags):
        pair = variable_tuples[i]
        histogram = make_histogram(df['weight'], df[pair[0]], 
                                   df[pair[1]], xgrid, ygrid)
        vmax = np.nanmax([vmax, np.nanmax(histogram[0])])
        histograms.append(histogram)

    # norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)

    for i, offdiag in enumerate(offdiags):
        pcm = plot_hist_2d(axd[offdiag], histograms[i], norm)
        pair = variable_tuples[i]
        axd[offdiag].set_xlabel(pair[1])
        axd[offdiag].set_ylabel(pair[0])

    cbar = fig.colorbar(pcm, cax=axd['j'])
    cbar.set_label('Probability Mass Function (values sum to 1)')

    fig.tight_layout()


if __name__=='__main__':
    phsp_file = '/Users/isaacmeyer/research/sarrp/SARRP_topas/4DHead/ASCIIOutput.phsp'
    phsp_obj = PhaseSpace(DataLoaders.Load_TopasData(phsp_file))
    phsp_obj.fill.direction_cosines()

    cosine_names = ['Direction Cosine X', 
                    'Direction Cosine Y',
                    'Direction Cosine Z']


    # Photon plotting
    df = phsp_obj.ps_data
    df = df[df['particle type [pdg_code]'] == 22]
    custom_pairplot(df, cosine_names)
    plt.show()
