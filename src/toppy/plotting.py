import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import copy

from .tallies import import_csv

def bin_enlarge(array, n):
    if len(array) % n : 
        raise ValueError("Cannot evenly divide bins into new structure") 
    nbins = int(len(array)/n)
    new_array = np.zeros(nbins)
    for i in range(nbins):
        new_array[i] = np.sum(array[n*i:n*(i+1)])
    return new_array

def hist_y(y):
    return np.hstack([y[0], y])

def compute_profile(data, axis, zlocation, shape, voxelwidths, profilewidth,
                    profilesmooth=1):
    """
    Currently written for even number bin stucture (center is bin boundary)

    axis: 0 or 1  for x or y
    zlocation: depth of profile [cm]
    shape: number of pixels in each dimension
    voxelwidths: width in each dimension [cm]
    profilewidth: thickness of line (how many voxels to include) [cm]
    """
    data = copy.deepcopy(data)
    tally_names = ['Sum', 'Variance', 'Count_in_Bin']

    # Find z index
    z_idx = int(zlocation/voxelwidths[2])
    for tally in tally_names:
        print(tally, data[tally].shape)
        data[tally] = data[tally][:,:,z_idx]

    # Determine thickness of profile of line in number of voxels (round up)
    nvoxels_eachside = int(np.ceil(profilewidth/2/voxelwidths[axis]))
    index_low = int(shape[axis]/2) - nvoxels_eachside
    index_high = int(shape[axis]/2) + nvoxels_eachside
    print('Using a profile width of {:d} voxels'.format(nvoxels_eachside))
    print('Indexes: '.format(index_low, index_high))
    if axis==0:
        for tally in tally_names:
            data[tally] = np.sum(data[tally][:, index_low:index_high], axis=1)
    elif axis==1:
        for tally in tally_names:
            data[tally] = np.sum(data[tally][index_low:index_high, :], axis=0)
    total_histories = 1e6
    if profilesmooth>1:
        for tally in tally_names:
            data[tally] = bin_enlarge(data[tally], profilesmooth)

    data['sum_SD'] = np.sqrt(data['Variance'] * total_histories)

    return data

def plot_integrated_profiles(data, nx, ny, nz, total_histories, variance=True):
    dimensions = ['X', 'Y', 'Z']

    data_arrays = {}
    for var in data.columns[2:]:
        data_arrays[var] = data[var].to_numpy().reshape(nx, ny, nz)[:,:,::-1]

    integrated = {}
    integrated_sd = {}
    integrated_percent_err = {}
    for i, dim in enumerate(dimensions):
        integrated[dim] = np.sum(data_arrays['Sum'], axis=i)
        if variance:
            integrated_sd[dim] = np.sqrt(np.sum(data_arrays['Variance'], axis=i)*total_histories)
            integrated_percent_err[dim] = integrated_sd[dim]/z_integrated[dim] * 100

    # Plot integrated profiles in phantom
    titles = ['ZY plane (X-integrated)', 'XZ plane (Y-integrated)', 'XY plane (Z-integrated)']
    for i, dim in enumerate(dimensions):

        fig = plt.figure()

        fig.suptitle(titles[i])

        norm = colors.Normalize(vmin=np.amin(integrated[dim]),
                                vmax=np.amax(integrated[dim]))
        cmap = "plasma"
        label = "Dose [Arbitrary Units]"
        ax = fig.add_subplot(211)
        gcf = ax.contourf(integrated[dim], cmap=cmap, norm=norm)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(gcf, cax=cax)
        cbar.set_label(label)
        ax.set_aspect(1)

        if variance:
            label = "Percent SD"
            norm = colors.Normalize(vmin=np.nanmin(integrated_percent_err[dim]),
                                    vmax=np.nanmax(integrated_percent_err[dim]))
            cmap = "plasma"
            ax = fig.add_subplot(212)
            gcf = ax.contourf(integrated_percent_err[dim], cmap=cmap, norm=norm)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = fig.colorbar(gcf, cax=cax)
            cbar.set_label(label)
            ax.set_aspect(1)

def plot_dose_profiles(data, depths):
    x_profiles = []
    axis = 0
    # profilewidth = 0.133
    profilewidth = 3
    profilesmooth = 1
    for depth in depths:
        xprof = compute_profile(data_arrays, axis, depth, [nx,ny,nz],
                                [0.0625, 0.0625, 0.1], 
                                profilewidth, profilesmooth)
        x_profiles.append(xprof)
        

    fig = plt.figure()
    ax = fig.add_subplot(111)
    xs = np.linspace(-5, 5, int(nx/profilesmooth) + 1)
    for i, depth in enumerate(depths):
        # s = ax.scatter(xs, x_profiles[i]['Sum'])
        # color = s.get_facecolors()[0]
        midpoints = (xs[1:] + xs[:-1])/2
        l = ax.step(xs, hist_y(x_profiles[i]['Sum']))
        color = l[0].get_color()
        ax.errorbar(midpoints, x_profiles[i]['Sum'], 
                    yerr=x_profiles[i]['sum_SD'], label = '{:.1f} cm'.format(depth),
                    fmt=' ',
                    capsize=2, capthick=1,
                    color=color)
    ax.legend()
    ax.set_xlabel('Y Position [cm]')
    ax.set_ylabel('Dose [arbitrary]')
    ax.set_title('X-Axis Dose Profile at Varying Depths')

def plot_dvh(vol_hist_file):
    data = import_csv(vol_hist_file, tallytype='DVH')
    print(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # ax.step(data['LowerLimit of DoseToMedium ( Gy )'], hist_y(data['Value']))
    ax.step(data['LowerLimit of DoseToMedium ( Gy )'], data['Value'])
    # ax.set_xscale('log')
    plt.show()




if __name__=='__main__':
    # data = import_csv('DoseAtPhantom.csv', ['X', 'Y', 'Z'])
    # csv = '/Users/isaacmeyer/research/secondary_risk/merging_APE_with_MCAUTO/Liver_edit/liver_mcauto/exec/Beam1/merging_beams/phantom_dose_3beam.csv'
    # csv = '/Users/isaacmeyer/research/secondary_risk/merging_APE_with_MCAUTO/Liver_edit/liver_mcauto/exec/Beam1/merging_beams/Beam1.csv'
    csv = '/Users/isaacmeyer/research/secondary_risk/merging_APE_with_MCAUTO/Liver_edit/liver_mcauto/exec/Beam1/merging_beams/phantom_dose_all.csv'

    data = import_csv(csv, tallytype='dose', dimensions=['X', 'Y', 'Z'])
    nx = 160
    ny = 160
    nz = 70
    total_histories = 1e6

    plot_integrated_profiles(data, nx, ny, nz, total_histories, variance=False)
    plt.show()

    depths = np.array([0.18, 5.55, 10.91, 16.28, 21.64, 32, 42.37, 52.73])
    depths *= 0.1 # convert to cm
    plot_dose_profiles(data, depths)
    plt.show()

    # file = '/Users/isaacmeyer/research/secondary_risk/dvh_example/DoseAtPhantom_VolHist.csv'
    # file = '/Users/isaacmeyer/research/secondary_risk/dvh_example/ExampleDose_VolHist.csv'
    # plot_dvh(file)

        

