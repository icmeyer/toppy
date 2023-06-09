import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from toppy.tallies import import_csv
from toppy.plotting import hist_y, lower_limits_to_edges

def import_abs_dif_dvh(filename):
    return import_csv(filename, tallytype='DVH')

def get_abs_dvh_bins_and_vals(filename, scorer_name='TsMRCPScorer'):
    df = import_abs_dif_dvh(filename)
    bins = df['LowerLimit of {:s} ( Gy )'.format(scorer_name)].values
    bins = lower_limits_to_edges(bins)
    vals = df['Value'].values
    return bins, vals

def plot_abs_dif_dvh(filename, title):
    bins, volumes = get_abs_dvh_bins_and_vals(filename)

    dDVH = volumes/np.sum(volumes)
    cDVH = np.cumsum(dDVH[::-1])[::-1]

    fig = plt.figure()

    ax = fig.add_subplot(311)
    ax.step(dose_bins, hist_y(volumes))
    ax.set_yscale('log')
    ax.set_ylabel('Absolute\n Differential DVH [mm$^3$]')
    # ax.set_xscale('log')

    ax = fig.add_subplot(312, sharex=ax)
    ax.step(dose_bins, hist_y(dDVH))
    ax.set_ylabel('Differential DVH')
    ax.set_yscale('log')

    ax = fig.add_subplot(313, sharex=ax)
    ax.step(dose_bins, hist_y(cDVH))
    ax.set_xlabel('Dose [Gy] (arbitrary scale)')
    ax.set_ylabel('Cumulative DVH')
    ax.set_yscale('log')

    fig.suptitle(title)

    fig.tight_layout()

    return fig

def save_abs_dif_plots(csv_dir, figfolder):
    os.system('mkdir -p {:s}'.format(figfolder))

    for file in glob.glob(csv_dir + '*AbsDifVolHist.csv'):
        title = os.path.split(file)[1]
        print('Plotting ', title)
        try:
            fig = plot_abs_dif_dvh(file, title)
            fig.savefig(figfolder + title[:-4] + '.pdf', dpi=400)
        except:
            print('failed')
