import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
import os

from ParticlePhaseSpace import DataLoaders, PhaseSpace

# https://www.python-graph-gallery.com/86-avoid-overlapping-in-scatterplot-with-2d-density
# Acknowledgement to plotting in ParticlePhaseSpace:
# https://github.com/bwheelz36/ParticlePhaseSpace/blob/main/ParticlePhaseSpace/_ParticlePhaseSpace.py

pdg2particle = {11: 'electrons',
                -11: 'positrons',
                2212: 'protons',
                22: 'gammas',
                2112: 'neutrons',
                0: 'optical_photons'}
particle2pdg = {v: k for k, v in pdg2particle.items()}

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

def plot_hist_1d(ax, xmin, xmax, nbins, data, xlabel, title=None):
    histbins = np.linspace(xmin, xmax, nbins + 1)
    counts, bins = np.histogram(data, bins=histbins)
    # Normalize the counts (probability mass function)
    counts = counts/np.sum(counts)
    ax.stairs(counts, bins)
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Probability Mass Function')
    if title is not None:
        ax.set_title(title)

def custom_pairplot(df, variables, title, filename):
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
    for i, diag in enumerate(diags):
        plot_hist_1d(axd[diag], -1, 1, nbins, df[variables[i]], variables[i])

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

    fig.suptitle(title)
    fig.tight_layout()

    fig.savefig(filename + '.pdf', dpi=400)

def plot_energy_spectra(df, particles, filename, nbins=100):
    fig = plt.figure(figsize = [8, 4])

    for i, particle in enumerate(particles):
        ax = fig.add_subplot(1, len(particles), i+1)
        pdg = particle2pdg[particle]
        dfpart = df[df['particle type [pdg_code]'] == pdg]
        nparts = len(dfpart)
        title = '{:s} Energy Distribution \n'.format(particle)
        title += '{:.1e} Particles \n'.format(nparts)
        print(pdg)
        print(dfpart.columns)
        print(dfpart)
        energies = dfpart['Ek [MeV]']
        xmin = np.amin(energies)
        xmax = np.amax(energies)
        plot_hist_1d(ax, xmin, xmax, nbins, energies, 'Kinetic Energy [MeV]')
        ax.set_title(title)
    
    fig.tight_layout()
    fig.savefig(filename + '.pdf', dpi=400)

def get_phsp_with_KE(phsp_file):
    phsp_obj = PhaseSpace(DataLoaders.Load_TopasData(phsp_file))
    phsp_obj.fill.direction_cosines()
    phsp_obj.fill.kinetic_E()
    return phsp_obj

def plot_3D_dist(phsp_file, figdir, dim_labels=['x [mm]','y [mm]','z [mm]']):
    os.system('mkdir -p {:s}'.format(figdir))
    phsp_obj = get_phsp_with_KE(phsp_file)
    df = phsp_obj.ps_data

    print(df)
    print(df[dim_labels[0]].values)
    print(df[dim_labels[1]].values)
    print(df[dim_labels[2]].values)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df[dim_labels[0]].values,
               df[dim_labels[1]].values,
               df[dim_labels[2]].values, marker='.', alpha=0.1)
    ax.set_xlabel(dim_labels[0])
    ax.set_ylabel(dim_labels[1])
    ax.set_zlabel(dim_labels[2])
    plt.show()

    
def plot_dists(phsp_file, figdir):
    os.system('mkdir -p {:s}'.format(figdir))
    
    phsp_obj = get_phsp_with_KE(phsp_file)

    # Photon plotting
    df = phsp_obj.ps_data

    # get partricles in dataset
    pdgs = list(set(df['particle type [pdg_code]'].to_numpy()))
    particles = [pdg2particle[pdg] for pdg in pdgs]

    cosine_names = ['Direction Cosine X', 
                    'Direction Cosine Y',
                    'Direction Cosine Z']
    for particle in particles:
        pdg = particle2pdg[particle]
        dfpart = df[df['particle type [pdg_code]'] == pdg]
        nparts = len(dfpart)
        title = '{:s} Cosine Distribution Correlations \n'.format(particle)
        title += '{:.1e} Particles \n'.format(nparts)
        filename = figdir + '{:s}_cosines'.format(particle)
        custom_pairplot(dfpart, cosine_names, title, filename)

    filename = figdir + 'energy_distributions'
    plot_energy_spectra(df, particles, filename)



if __name__=='__main__':
    plot_dists(phsp_file, 'tests/')
