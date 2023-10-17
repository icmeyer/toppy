import matplotlib.pyplot as plt
import numpy as np

from ParticlePhaseSpace import DataLoaders, PhaseSpace

import toppy

def import_df(file):
    print('importing ', file)
    phsp = PhaseSpace(DataLoaders.Load_TopasData(file))
    phsp.fill.direction_cosines()
    phsp.fill.kinetic_E()
    df = phsp.ps_data
    print('done importing ', file)
    return df

file_list = ['/Users/isaacmeyer/research/murine_multiscale/moby_simulations/SARRP_phasespace_scoring/3mm/lefthip_phasespace.phsp',
             '/Users/isaacmeyer/research/murine_multiscale/moby_simulations/SARRP_phasespace_scoring/3mm/righthip_phasespace.phsp']
names = ['Left Hip', 'Right Hip']

# figdir = 'comparison'
# os.system('mkdir -p {:s}'.format(figdir))

dfs = {}
for f, file in enumerate(file_list):
    dfs[names[f]] = import_df(file)

fig = plt.figure()
ax = fig.add_subplot(111)

title = 'Photon Energy Distribution'
ax.set_title(title)
ax.set_xlabel('$E_k$ [MeV]')
ax.set_ylabel('Probability Mass Function')
nbins = 50

for phsp_name in dfs:
    df = dfs[phsp_name]
    dfpart = df[df['particle type [pdg_code]'] == 22]
    nparts = len(dfpart)

    energies = dfpart['Ek [MeV]']
    xmin = np.amin(energies)
    xmax = np.amax(energies)

    histbins = np.linspace(xmin, xmax, nbins + 1)
    counts, bins = np.histogram(energies, bins=histbins)
    # Normalize the counts (probability mass function)
    counts = counts/np.sum(counts)
    ax.stairs(counts, bins, label=phsp_name)

ax.legend()
plt.show()
fig.savefig('comparison.pdf')

