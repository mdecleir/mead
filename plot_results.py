# plot_results.py: script to plot results

# import the necessary packages
import numpy as np

from astropy.table import Table, join
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


def plot_feat(outpath, feat_name, data):
    """
    Function to plot the feature properties vs. each other

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
    Plots with feature properties
    """
    # create the figure
    fs = 18
    fig, axes = plt.subplots(3, 3, figsize=(12, 12), sharex="col", sharey="row")

    # define the parameters to be plotted
    xpars = ["amplitude", "wavelength(micron)", "std(micron)"]
    xlabels = [
        r"$\tau(\lambda_0)$",
        r"$\lambda_0$ ($\mu$m)",
        r"$\sigma$ ($\mu$m)",
    ]
    ypars = ["wavelength(micron)", "std(micron)", "area(cm-1)"]
    ylabels = [
        r"$\lambda_0$ ($\mu$m)",
        r"$\sigma$ ($\mu$m)",
        r"$A$ (cm$^{-1}$)",
    ]
    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # obtain the x-axis uncertainties
        if "(" in xpar:
            index = xpar.find("(")
            x_unc = (
                data[xpar[:index] + "_unc_min" + xpar[index:]],
                data[xpar[:index] + "_unc_plus" + xpar[index:]],
            )
        else:
            x_unc = data[xpar + "_unc_min"], data[xpar + "_unc_plus"]

        # add the x-axis label
        axes[2, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # skip the duplicate plots
            if j < i:
                axes[j, i].axis("off")
                continue

            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                y_unc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                y_unc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]

            # plot the data
            axes[j, i].errorbar(
                data[xpar],
                data[ypar],
                xerr=x_unc,
                yerr=y_unc,
                fmt="ok",
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # finalize and save the figure
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(outpath + feat_name + "_params.pdf", bbox_inches="tight")


def plot_feat_AV_RV(outpath, feat_name, data):
    """
    Function to plot the feature properties vs. A(V) and R(V)

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
    Plots with feature properties vs. A(V) and R(V)
    """
    # define the data to be plotted
    xpars = ["AV", "RV"]
    xlabels = ["A(V)", "R(V)"]
    ypars = ["amplitude", "wavelength(micron)", "std(micron)", "area(cm-1)"]
    ylabels = [
        r"$\tau(\lambda_0)$",
        r"$\lambda_0$ ($\mu$m)",
        r"$\sigma$ ($\mu$m)",
        r"$A$ (cm$^{-1}$)",
    ]

    # create the figure
    fs = 18
    fig, axes = plt.subplots(
        len(ypars), len(xpars), figsize=(10, 16), sharex="col", sharey="row"
    )

    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # calculate the uncertainty on the x-axis value
        x_unc = np.sqrt(data[xpar + "_runc"] ** 2 + data[xpar + "_sunc"] ** 2)

        # add the x-axis label
        axes[3, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                y_unc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                y_unc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]

            # plot the properties
            axes[j, i].errorbar(
                data[xpar],
                data[ypar],
                xerr=x_unc,
                yerr=y_unc,
                fmt="ok",
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outpath + feat_name + "_AV_RV.pdf", bbox_inches="tight")


def plot_feat_norm_AV_RV(outpath, feat_name, data):
    """
    Function to plot the normalized feature properties vs. A(V) and R(V)

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
    Plots with normalized feature properties vs. A(V) and R(V)
    """
    # define the data to be plotted
    xpars = ["AV", "RV"]
    xlabels = ["$A(V)$", "$R(V)$"]
    ypars = ["amplitude", "area(cm-1)"]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$A/A(V)$ (cm$^{-1}$)",
    ]

    # create the figure
    fs = 18
    fig, axes = plt.subplots(
        len(ypars), len(xpars), figsize=(10, 10), sharex="col", sharey="row"
    )

    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # calculate the uncertainty on the x-axis value
        x_unc = np.sqrt(data[xpar + "_runc"] ** 2 + data[xpar + "_sunc"] ** 2)

        # add the x-axis label
        axes[-1, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                y_unc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                y_unc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]
            AV_unc = np.sqrt(data["AV_runc"] ** 2 + data["AV_sunc"] ** 2)
            y_min = (
                data[ypar]
                / data["AV"]
                * np.sqrt((y_unc[0] / data[ypar]) ** 2 + (AV_unc / data["AV"]) ** 2)
            )
            y_max = (
                data[ypar]
                / data["AV"]
                * np.sqrt((y_unc[1] / data[ypar]) ** 2 + (AV_unc / data["AV"]) ** 2)
            )

            # plot the normalized properties
            axes[j, i].errorbar(
                data[xpar],
                data[ypar] / data["AV"],
                xerr=x_unc,
                yerr=(y_min, y_max),
                fmt="ok",
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar] / data["AV"])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outpath + feat_name + "_norm_AV_RV.pdf", bbox_inches="tight")


def plot_feat_FM90(outpath, feat_name, data):
    """
    Function to plot the (normalized) feature properties vs. the FM90 parameters (UV extinction)

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
    Plots with (normalized) feature properties vs. the FM90 parameters
    """
    # define the data and uncertainties to be plotted
    xpars = [
        data["C2"],
        data["C3"] / data["gamma"] ** 2,
        np.pi * data["C3"] / (2 * data["gamma"]),
        data["C4"],
    ]
    xlabels = [
        "UV slope = C2",
        "UV bump amplitude \n = C3/$\gamma^2$",
        "UV bump area \n = $\pi$C3/(2$\gamma$)",
        "FUV rise = C4",
    ]

    C3_unc = np.sqrt(data["C3_runc"] ** 2 + data["C3_sunc"] ** 2)
    gamma_unc = np.sqrt(data["gamma_runc"] ** 2 + data["gamma_sunc"] ** 2)
    xuncs = [
        np.sqrt(data["C2_runc"] ** 2 + data["C2_sunc"] ** 2),
        data["C3"]
        / data["gamma"] ** 2
        * np.sqrt((C3_unc / data["C3"]) ** 2 + (2 * gamma_unc / data["gamma"]) ** 2),
        np.pi
        * data["C3"]
        / (2 * data["gamma"])
        * np.sqrt((C3_unc / data["C3"]) ** 2 + (gamma_unc / data["gamma"]) ** 2),
        np.sqrt(data["C4_runc"] ** 2 + data["C4_sunc"] ** 2),
    ]

    ypars = [
        "amp/AV",
        "wavelength(micron)",
        "std(micron)",
        "area/AV",
    ]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"$\sigma$ ($\mu$m)",
        r"$A/A(V)$ (cm$^{-1}$)",
    ]

    # create the figure
    fs = 18
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    for i, (xpar, xunc, xlabel) in enumerate(zip(xpars, xuncs, xlabels)):
        # add the x-axis label
        axes[-1, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                yunc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                yunc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]

            # plot the properties
            axes[j, i].errorbar(
                xpar,
                data[ypar],
                xerr=xunc,
                yerr=yunc,
                fmt="ok",
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$" % spearmanr(xpar, data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outpath + feat_name + "_FM90.pdf", bbox_inches="tight")


def plot_feat_H(outpath, feat_name, data):
    """
    Function to plot the feature properties vs. the hydrogen measurements

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
    Plots with feature properties vs. hydrogen measurements
    """
    # define the data to be plotted
    xpars = [
        "logNH",
        "logNHI",
        "logNH2",
        "fH2",
    ]
    xlabels = [
        "log(N(H))",
        "log(N(HI))",
        "log(N(H$_2$))",
        "f(H$_2$) = 2N(H$_2$)/N(H)",
    ]

    ypars = [
        "amp/AV",
        "wavelength(micron)",
        "std(micron)",
        "area/AV",
    ]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"$\sigma$ ($\mu$m)",
        r"$A/A(V)$ (cm$^{-1}$)",
    ]

    # create the figure
    fs = 18
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # add the x-axis label
        axes[-1, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                yunc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                yunc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]

            # plot the properties
            axes[j, i].errorbar(
                data[xpar],
                data[ypar],
                xerr=data["e_" + xpar],
                yerr=yunc,
                fmt="ok",
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outpath + feat_name + "_H.pdf", bbox_inches="tight")


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
    litpath = "/Users/mdecleir/Documents/MEAD/Literature_data/"
    outpath = "/Users/mdecleir/Documents/MEAD/Plots/"

    # obtain the extinction properties
    ext_table = Table.read(
        litpath + "Gordon+2009_tab2.dat",
        format="ascii",
    )

    # obtain the FM90 parameters
    fm90_table = Table.read(
        litpath + "Gordon+2009_tab3.dat",
        format="ascii",
    )

    # obtain the hydrogen measurements
    hyd_table = Table.read(
        litpath + "VanDePutte+2023_tab2.dat",
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
    joined_fm90_10 = join(joined_ext_10, fm90_table, keys="Name")
    joined_dep_10 = join(dep_table, feat_10, keys_left="star", keys_right="name")

    # add extra columns with normalized feature strength and area and uncertainties
    joined_fm90_10["amp/AV"] = joined_fm90_10["amplitude"] / joined_fm90_10["AV"]
    joined_fm90_10["area/AV"] = joined_fm90_10["area(cm-1)"] / joined_fm90_10["AV"]
    AV_unc = np.sqrt(joined_fm90_10["AV_runc"] ** 2 + joined_fm90_10["AV_sunc"] ** 2)
    joined_fm90_10["amp/AV_unc_min"] = joined_fm90_10["amp/AV"] * np.sqrt(
        (joined_fm90_10["amplitude_unc_min"] / joined_fm90_10["amplitude"]) ** 2
        + (AV_unc / joined_fm90_10["AV"]) ** 2
    )
    joined_fm90_10["amp/AV_unc_plus"] = joined_fm90_10["amp/AV"] * np.sqrt(
        (joined_fm90_10["amplitude_unc_plus"] / joined_fm90_10["amplitude"]) ** 2
        + (AV_unc / joined_fm90_10["AV"]) ** 2
    )
    joined_fm90_10["area/AV_unc_min"] = joined_fm90_10["area/AV"] * np.sqrt(
        (joined_fm90_10["area_unc_min(cm-1)"] / joined_fm90_10["area(cm-1)"]) ** 2
        + (AV_unc / joined_fm90_10["AV"]) ** 2
    )
    joined_fm90_10["area/AV_unc_plus"] = joined_fm90_10["area/AV"] * np.sqrt(
        (joined_fm90_10["area_unc_plus(cm-1)"] / joined_fm90_10["area(cm-1)"]) ** 2
        + (AV_unc / joined_fm90_10["AV"]) ** 2
    )

    # merge more tables
    joined_hyd_10 = join(joined_fm90_10, hyd_table, keys_left="Name", keys_right="Star")

    # calculate the uncertainty on f(H2)
    joined_hyd_10["e_fH2"] = (
        joined_hyd_10["fH2"]
        * np.log(10)
        * np.sqrt(joined_hyd_10["e_logNH2"] ** 2 + joined_hyd_10["e_logNH"] ** 2)
    )

    # define the stars that should be masked
    bad_star = "HD014434"
    bad_mask = feat_10["name"] == bad_star

    # plot the feature properties against each other
    plot_feat(outpath, "10", feat_10[~bad_mask])

    # create plots vs. A(V) and R(V)
    # plot_feat_AV(outpath, "58", joined_ext_58)
    plot_feat_AV_RV(outpath, "10", joined_ext_10[~bad_mask])
    plot_feat_norm_AV_RV(outpath, "10", joined_ext_10[~bad_mask])

    # create plots vs. the FM90 parameters
    plot_feat_FM90(outpath, "10", joined_fm90_10[~bad_mask])

    # create plots vs. hydrogen measurements
    plot_feat_H(outpath, "10", joined_hyd_10[~bad_mask])

    # create plot vs. dust column densities
    # plot_feat_col(outpath, "10", joined_dep_10)


if __name__ == "__main__":
    main()
