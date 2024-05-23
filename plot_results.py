# plot_results.py: script to plot results

# import the necessary packages
import numpy as np

from astropy.table import Table, join
from matplotlib import pyplot as plt


def plot_feat_AV(outpath, feat_name, data):
    """
    Function to plot the feature properties vs. A(V)

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    Returns
    -------
        Plots with feature properties vs. A(V)
    """
    # create the figure
    fs = 18
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)

    # calculate the uncertainty on A(V)
    AV_err = np.sqrt(data["AV_runc"] ** 2 + data["AV_sunc"] ** 2)

    # plot the feature properties vs. A(V)
    props = ["amplitude", "wavelength(micron)", "std(micron)", "area(cm-1)"]
    for prop, ax in zip(props, axes.flat):
        # obtain the y-axis uncertainties
        if "(" in prop:
            index = prop.find("(")
            yerr = (
                data[prop[:index] + "_unc_min" + prop[index:]],
                data[prop[:index] + "_unc_plus" + prop[index:]],
            )
        else:
            yerr = data[prop + "_unc_min"], data[prop + "_unc_plus"]

        ax.errorbar(
            data["AV"],
            data[prop],
            xerr=AV_err,
            yerr=yerr,
            fmt="ok",
        )

    # finalize and save the figure
    axes[1, 0].set_xlabel("A(V)", fontsize=fs)
    axes[1, 1].set_xlabel("A(V)", fontsize=fs)
    axes[0, 0].set_ylabel("optical depth", fontsize=fs)
    axes[0, 1].set_ylabel(r"central wavelength ($\mu$m)", fontsize=fs)
    axes[1, 0].set_ylabel(r"standard deviation ($\mu$m)", fontsize=fs)
    axes[1, 1].set_ylabel("integrated area (cm$^{-1}$)", fontsize=fs)
    plt.subplots_adjust(hspace=0, wspace=0.3)
    plt.savefig(outpath + feat_name + "_AV.pdf", bbox_inches="tight")


def plot_feat_col(outpath, feat_name, data):
    """
    Function to plot the dust column densities vs. the integrated area of the feature

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    Returns
    -------
        Plot with dust column densities vs. the integrated area of the feature
    """
    # create the figure
    fs = 18
    fig, axes = plt.subplots(3, figsize=(7, 8), sharex=True)

    # plot elemental dust column densities vs. integrated area
    elems = ["Mg", "Fe", "O"]
    for elem, ax in zip(elems, axes.flat):
        ax.errorbar(
            data["area(cm-1)"],
            data["N(" + elem + ")_d"],
            #  xerr=[data["area_unc_min(cm-1)"], data["area_unc_plus(cm-1)"]],
            fmt="ok",
        )
        ax.set_ylabel("N(" + elem + ")$_{dust}$", fontsize=fs)

    # finalize and save the figure
    axes[2].set_xlabel("integrated area (cm$^{-1}$)", fontsize=fs)
    plt.subplots_adjust(hspace=0, wspace=0.3)
    plt.savefig(outpath + feat_name + "_col.pdf", bbox_inches="tight")


def main():
    # plotting settings for uniform plots
    fs = 18
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)

    # define the data path and the output path
    datapath = "/Users/mdecleir/Documents/MEAD/Extinction/JWST_data/"
    outpath = "/Users/mdecleir/Documents/MEAD/Plots/"

    # obtain the extinction properties
    ext_table = Table.read(
        "/Users/mdecleir/Documents/MEAD/Literature_data/Gordon+2009_tab2.dat",
        format="ascii",
    )

    # obtain the depletion measurements
    dep_table = Table.read(
        "/Users/mdecleir/stis_depletions/depletions.dat", format="ascii"
    )

    # obtain the fitting results of the features
    feat_58 = Table.read(datapath + "fit_results_58.txt", format="ascii")
    feat_10 = Table.read(datapath + "fit_results_10.txt", format="ascii")

    # merge tables
    joined_ext_58 = join(ext_table, feat_58, keys_left="Name", keys_right="name")
    joined_ext_10 = join(ext_table, feat_10, keys_left="Name", keys_right="name")
    joined_dep_10 = join(dep_table, feat_10, keys_left="star", keys_right="name")

    # define the stars that should be masked
    bad_star = "HD014434"
    bad_mask = joined_ext_10["Name"] == bad_star

    # create plots vs. A(V)
    # plot_feat_AV(outpath, "58", joined_ext_58)
    plot_feat_AV(outpath, "10", joined_ext_10[~bad_mask])

    # create plot vs. dust column densities
    # plot_feat_col(outpath, "10", joined_dep_10)


if __name__ == "__main__":
    main()
