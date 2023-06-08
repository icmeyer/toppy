import glob
import os
import matplotlib.pyplot as plt

import toppy

# filename = '/Users/isaacmeyer/topas_extensions/CancerRisk/examples/DoseLiver_AbsDifVolHist.csv'
# toppy.volume_histogram.plot_abs_dif_dvh(filename, 'liver')
# plt.show()

directory = '/Users/isaacmeyer/topas_extensions/CancerRisk/examples/test_all_SEER_organs/'
figfolder = './histogram_figs/'
toppy.volume_histogram.save_abs_dif_plots(directory, figfolder)
