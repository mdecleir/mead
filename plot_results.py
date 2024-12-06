# plot_results.py: script to plot results

# import the necessary packages
import astropy.units as u
import math
import numpy as np
import os

from astropy.modeling.fitting import LinearLSQFitter
from astropy.modeling.models import Linear1D
from astropy.table import Table, join
from dust_extinction.shapes import FM90
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


def calc_A1500(data):
    """
    Function to calculate the absolute extinction at 1500 Angstrom, and its uncertainty

    Parameters
    ----------
    data : astropy Table
        data including FM90 extinction parameters

    Returns
    -------
    A1500s : numpy.ndarray
        Absolute extinction at 1500 Angstrom for all stars

    A1500_uncs : numpy.ndarray
        Uncertainty on A1500 for all stars
    """
    # obtain the extinction curve model for every star
    ext_model = FM90(
        C1=data["C1"],
        C2=data["C2"],
        C3=data["C3"],
        C4=data["C4"],
        xo=data["xo"],
        gamma=data["gamma"],
    )

    # calculate E(1500-V)/E(B-V)
    E1500 = ext_model(1500 * u.Angstrom)

    # convert E(1500-V)/E(B-V) to A(1500)
    # A1500 = E1500 * E(B-V) + A(V)
    A1500 = E1500 * data["EBV"].value + data["AV"].value

    # calculate the uncertainty on A(1500) using bootstrapping
    A1500_uncs = []
    # for every star
    for row in data:
        # obtain the FM90 and extinction parameters for this star
        mean = (
            row["C1"],
            row["C2"],
            row["C3"],
            row["C4"],
            row["xo"],
            row["gamma"],
            row["EBV"],
            row["AV"],
        )

        # create a covariance matrix for the FM90 and extinction parameters
        cov = np.zeros((8, 8))
        cov[0, 0] = np.sqrt(row["C1_runc"] ** 2 + row["C1_sunc"] ** 2)
        cov[1, 1] = np.sqrt(row["C2_runc"] ** 2 + row["C2_sunc"] ** 2)
        cov[2, 2] = np.sqrt(row["C3_runc"] ** 2 + row["C3_sunc"] ** 2)
        cov[3, 3] = np.sqrt(row["C4_runc"] ** 2 + row["C4_sunc"] ** 2)
        cov[4, 4] = np.sqrt(row["xo_runc"] ** 2 + row["xo_sunc"] ** 2)
        cov[5, 5] = np.sqrt(row["gamma_runc"] ** 2 + row["gamma_sunc"] ** 2)
        cov[6, 6] = np.sqrt(row["EBV_runc"] ** 2 + row["EBV_sunc"] ** 2)
        cov[7, 7] = np.sqrt(row["AV_runc"] ** 2 + row["AV_sunc"] ** 2)

        # obtain a set of 1000 FM90 and extinction parameter combinations
        FM90_set = np.random.multivariate_normal(mean=mean, cov=cov, size=1000)

        # evaluate the FM90 extinction model for all parameter combinations
        ext_model = FM90(
            C1=FM90_set[:, 0],
            C2=FM90_set[:, 1],
            C3=FM90_set[:, 2],
            C4=FM90_set[:, 3],
            xo=FM90_set[:, 4],
            gamma=FM90_set[:, 5],
        )

        # calculate E(1500-V)/E(B-V)
        E1500_chain = ext_model(1500 * u.Angstrom)

        # convert E(1500-V)/E(B-V) to A(1500)
        # A1500 = E1500 * E(B-V) + A(V)
        A1500_chain = E1500_chain * FM90_set[:, 6] + FM90_set[:, 7]

        # calculate the standard deviation of the sample
        A1500_uncs.append(np.std(A1500_chain))

    return A1500, A1500_uncs


def plot_feat(outpath, feat_name, data, bad_mask):
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

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    Returns
    -------
    Plots with feature properties
    """
    # define the parameters to be plotted
    # with FWHM
    # xpars = ["x_0(micron)", "tau", "FWHM(micron)", "shape"]
    # xlabels = [
    #     r"$\lambda_0$ ($\mu$m)",
    #     r"$\tau(\lambda_0)$",
    #     r"FWHM ($\mu$m)",
    #     r"$\alpha$",
    # ]
    # ypars = ["tau", "FWHM(micron)", "shape", "area(micron)"]
    # ylabels = [r"$\tau(\lambda_0)$", r"FWHM ($\mu$m)", r"$\alpha$", r"area ($\mu$m)"]

    # without FWHM
    # xpars = ["x_0(micron)", "tau", "shape"]
    # xlabels = [r"$\lambda_0$ ($\mu$m)", r"$\tau(\lambda_0)$", r"$\alpha$"]
    # ypars = ["tau", "shape", "area(micron)"]
    # ylabels = [r"$\tau(\lambda_0)$", r"$\alpha$", r"area ($\mu$m)"]

    # most compact plot
    xpars = ["tau", "shape"]
    xlabels = [r"$\tau(\lambda_0)$", r"$\alpha$"]
    ypars = ["area(micron)", "x_0(micron)"]
    ylabels = [r"area ($\mu$m)", r"$\lambda_0$ ($\mu$m)"]

    # create the figure
    fs = 20
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # obtain the x-axis uncertainties
        if "(" in xpar:
            index = xpar.find("(")
            xunc = (
                data[xpar[:index] + "_unc_min" + xpar[index:]],
                data[xpar[:index] + "_unc_plus" + xpar[index:]],
            )
        else:
            xunc = data[xpar + "_unc_min"], data[xpar + "_unc_plus"]

        # add the x-axis label
        axes[-1, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # skip the duplicate plots
            if j < i:
                axes[j, i].axis("off")
                continue

            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                yunc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                yunc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]

            # plot the data
            axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=(xunc[0][~bad_mask], xunc[1][~bad_mask]),
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=(xunc[0][bad_mask], xunc[1][bad_mask]),
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                alpha=0.25,
                markeredgewidth=0,
                elinewidth=0.4,
            )
            axes[j, i].tick_params(axis="both", labelsize=fs * 0.8)

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )
            axes[j, i].text(
                0.05,
                0.82,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_params.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(outname, bbox_inches="tight")


def plot_sil_lit(outpath, data, bad_mask, lit_table):
    """
    Function to plot the silicate feature peak optical depth vs. peak wavelength with literature data

    Parameters
    ----------
    outpath : string
        Path to store the plot

    data : astropy Table
        Data to plot

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    lit_table : astropy Table
        Literature data

    Returns
    -------
    Plot with the silicate feature peak optical depth vs. peak wavelength with literature data
    """
    # create the figure
    fs = 14
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    # plot the data
    ax.errorbar(
        data["tau"][~bad_mask],
        data["x_0(micron)"][~bad_mask],
        xerr=(data["tau_unc_min"][~bad_mask], data["tau_unc_plus"][~bad_mask]),
        yerr=(
            data["x_0_unc_min(micron)"][~bad_mask],
            data["x_0_unc_plus(micron)"][~bad_mask],
        ),
        fmt="ok",
        markeredgewidth=0,
        elinewidth=0.4,
        label="this work",
    )
    ax.errorbar(
        data["tau"][bad_mask],
        data["x_0(micron)"][bad_mask],
        xerr=(data["tau_unc_min"][bad_mask], data["tau_unc_plus"][bad_mask]),
        yerr=(
            data["x_0_unc_min(micron)"][bad_mask],
            data["x_0_unc_plus(micron)"][bad_mask],
        ),
        fmt="ok",
        alpha=0.25,
        markeredgewidth=0,
        elinewidth=0.4,
    )

    # add literature values
    ax.errorbar(
        lit_table["tau"],
        lit_table["lambda_o1"],
        xerr=(lit_table["tau_unc_min"], lit_table["tau_unc_plus"]),
        yerr=(lit_table["lambda_o1_unc_min"], lit_table["lambda_o1_unc_plus"]),
        fmt="x",
        color="green",
        elinewidth=0.4,
        label="Gordon+2021",
    )

    # calculate the Spearman correlation coefficient
    ax.text(
        0.05,
        0.92,
        r"$\rho = %.2f$"
        % spearmanr(data["x_0(micron)"][~bad_mask], data["tau"][~bad_mask])[0],
        transform=ax.transAxes,
        fontsize=fs * 0.8,
        ha="left",
    )
    ax.text(
        0.05,
        0.86,
        r"$\rho = %.2f$" % spearmanr(data["x_0(micron)"], data["tau"])[0],
        transform=ax.transAxes,
        fontsize=fs * 0.8,
        ha="left",
        alpha=0.25,
    )

    # add literature values
    lambdas = np.concatenate((data["x_0(micron)"], lit_table["lambda_o1"]))
    taus = np.concatenate((data["tau"], lit_table["tau"]))
    ax.text(
        0.05,
        0.8,
        r"$\rho = %.2f$" % spearmanr(lambdas, taus)[0],
        transform=ax.transAxes,
        fontsize=fs * 0.8,
        ha="left",
        color="green",
    )

    # rename the previous version of the plot
    outname = outpath + "10_params_lit.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    ax.set_xlabel(r"$\tau(\lambda_0)$", fontsize=fs)
    ax.set_ylabel(r"$\lambda_0$ ($\mu$m)", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs * 0.8)
    plt.legend()
    plt.savefig(outname, bbox_inches="tight")


def plot_feat_AV_RV(outpath, feat_name, data, bad_mask):
    """
    Function to plot the feature properties vs. A(V), A(1500A) and R(V)

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    Returns
    -------
    Plots with feature properties vs. A(V), A(1500A) and R(V)
    """
    # define the parameters to be plotted
    # xpars = ["AV", "A1500", "RV"]
    # xlabels = ["$A(V)$", r"$A(1500\AA)$", "$R(V)$"]
    # with FWHM and shape
    # ypars = ["x_0(micron)", "tau", "FWHM(micron)", "area(micron)", "shape"]
    # ylabels = [
    #     r"$\lambda_0$ ($\mu$m)",
    #     r"$\tau(\lambda_0)$",
    #     r"FWHM ($\mu$m)",
    #     r"area ($\mu$m)",
    #     r"$\alpha$",
    # ]
    # without FWHM or shape
    # ypars = ["x_0(micron)", "tau", "area(micron)"]
    # ylabels = [r"$\lambda_0$ ($\mu$m)", r"$\tau(\lambda_0)$", r"area ($\mu$m)"]

    # most compact plot
    xpars = ["AV", "A1500"]
    xlabels = ["$A(V)$", r"$A(1500\AA)$"]
    ypars = ["x_0(micron)", "tau"]
    ylabels = [r"$\lambda_0$ ($\mu$m)", r"$\tau(\lambda_0)$"]

    # create the figure
    fs = 20
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

            # plot the data
            axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data[xpar + "_unc"][~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=data[xpar + "_unc"][bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )
            axes[j, i].text(
                0.05,
                0.82,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )
            axes[j, i].tick_params(axis="both", labelsize=fs * 0.8)

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_AV_A1500_RV.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")


def plot_feat_AV_RV_lit(outpath, feat_name, data, bad_mask, lit_table):
    """
    Function to plot the feature properties vs. A(V) and R(V) with literature data

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    lit_table : astropy Table
        Literature data

    Returns
    -------
    Plots with feature properties vs. A(V) and R(V) with literature data
    """
    # define the parameters to be plotted
    # with R(V)
    # xpars = ["AV", "RV"]
    # xpars_lit = ["A(V)", "R(V)"]
    # xlabels = ["$A(V)$", "$R(V)$"]
    # without R(V)
    xpars = ["AV"]
    xpars_lit = ["A(V)"]
    xlabels = ["$A(V)$"]
    ypars = ["x_0(micron)", "tau"]
    ypars_lit = ["lambda_o1", "tau"]
    ylabels = [
        r"$\lambda_0$ ($\mu$m)",
        r"$\tau(\lambda_0)$",
    ]

    # create the figure
    fs = 20
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    for i, (xpar, xpar_lit, xlabel) in enumerate(zip(xpars, xpars_lit, xlabels)):
        # add the x-axis label
        if len(xpars) == 1:
            ax = axes[-1]
        else:
            ax = axes[-1, i]
        ax.set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ypar_lit, ylabel) in enumerate(zip(ypars, ypars_lit, ylabels)):
            if len(xpars) == 1:
                ax = axes[j]
            else:
                ax = axes[j, i]

            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                yunc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                yunc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]

            # plot the data
            handle1 = ax.errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data[xpar + "_unc"][~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            ax.errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=data[xpar + "_unc"][bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
                label="this work",
            )
            # add literature data
            handle2 = ax.errorbar(
                lit_table[xpar_lit],
                lit_table[ypar_lit],
                xerr=(
                    lit_table[xpar_lit + "_unc_min"],
                    lit_table[xpar_lit + "_unc_plus"],
                ),
                yerr=(
                    lit_table[ypar_lit + "_unc_min"],
                    lit_table[ypar_lit + "_unc_plus"],
                ),
                fmt="x",
                color="green",
                elinewidth=0.4,
                label="Gordon+2021",
            )

            # calculate the Spearman correlation coefficient
            ax.text(
                0.05,
                0.92,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )
            ax.text(
                0.05,
                0.85,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )
            ax.tick_params(axis="both", labelsize=fs * 0.8)

            # add literature values
            xs = np.concatenate((data[xpar], lit_table[xpar_lit]))
            ys = np.concatenate((data[ypar], lit_table[ypar_lit]))
            ax.text(
                0.05,
                0.78,
                r"$\rho = %.2f$" % spearmanr(xs, ys)[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
                color="green",
            )

            # add the y-axis label (once)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_AV_RV_lit.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    labels = ["this work", "Gordon+2021"]
    handles = [handle1, handle2]
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.91, 0.89),
        handletextpad=-0.2,
        fontsize=0.7 * fs,
    )
    fig.savefig(outname, bbox_inches="tight")


def plot_feat_norm_AV_RV(outpath, feat_name, data, bad_mask, lit_table):
    """
    Function to plot the normalized feature properties vs. A(V), A(1500A) and R(V)

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    Returns
    -------
    Plots with normalized feature properties vs. A(V), A(1500) and R(V)
    """
    # define the parameters to be plotted
    # with A1500
    # xpars = ["AV", "A1500", "RV"]
    # xlabels = ["$A(V)$", r"$A(1500\AA)$", "$R(V)$"]
    # without A1500
    xpars = ["AV", "RV"]
    xpars_lit = ["A(V)", "R(V)"]
    xlabels = ["$A(V)$", "$R(V)$"]
    ypars = ["tau/AV", "area/AV"]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"area/$A(V)$ ($\mu$m)",
    ]

    # create the figure
    fs = 20
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    for i, (xpar, xpar_lit, xlabel) in enumerate(zip(xpars, xpars_lit, xlabels)):
        # add the x-axis label
        axes[-1, i].set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # plot the data
            handle1 = axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data[xpar + "_unc"][~bad_mask],
                yerr=(
                    data[ypar + "_unc_min"][~bad_mask],
                    data[ypar + "_unc_plus"][~bad_mask],
                ),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=data[xpar + "_unc"][bad_mask],
                yerr=(
                    data[ypar + "_unc_min"][bad_mask],
                    data[ypar + "_unc_plus"][bad_mask],
                ),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
            )
            # add literature data
            if ypar == "tau/AV":
                handle2 = axes[j, i].errorbar(
                    lit_table[xpar_lit],
                    lit_table[ypar],
                    xerr=(
                        lit_table[xpar_lit + "_unc_min"],
                        lit_table[xpar_lit + "_unc_plus"],
                    ),
                    yerr=(
                        lit_table[ypar + "_unc_min"],
                        lit_table[ypar + "_unc_plus"],
                    ),
                    fmt="x",
                    color="green",
                    elinewidth=0.4,
                    label="Gordon+2021",
                )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.92,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )
            axes[j, i].text(
                0.05,
                0.85,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )
            axes[j, i].tick_params(axis="both", labelsize=fs * 0.8)

            # add literature values
            if ypar == "tau/AV":
                xs = np.concatenate((data[xpar], lit_table[xpar_lit]))
                ys = np.concatenate((data[ypar], lit_table[ypar]))
                axes[j, i].text(
                    0.05,
                    0.78,
                    r"$\rho = %.2f$" % spearmanr(xs, ys)[0],
                    transform=axes[j, i].transAxes,
                    fontsize=fs * 0.8,
                    ha="left",
                    color="green",
                )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_norm_AV_A1500_RV.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    labels = ["this work", "Gordon+2021"]
    handles = [handle1, handle2]
    fig.legend(
        handles,
        labels,
        bbox_to_anchor=(0.91, 0.89),
        handletextpad=-0.2,
        borderpad=0,
        fontsize=0.7 * fs,
    )
    fig.savefig(outname, bbox_inches="tight")


def plot_feat_FM90(outpath, feat_name, data, bad_mask):
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

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

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
        r"UV bump amplitude" "\n" r"= C3/$\gamma^2$",
        r"UV bump area" "\n" r"= $\pi$C3/(2$\gamma$)",
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

    ypars = ["tau/AV", "x_0(micron)", "FWHM(micron)", "area/AV"]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"FWHM ($\mu$m)",
        r"area/$A(V)$ ($\mu$m)",
    ]

    # create the figure
    fs = 20
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)
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

            # plot the data
            axes[j, i].errorbar(
                xpar[~bad_mask],
                data[ypar][~bad_mask],
                xerr=xunc[~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            axes[j, i].errorbar(
                xpar[bad_mask],
                data[ypar][bad_mask],
                xerr=xunc[bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$" % spearmanr(xpar[~bad_mask], data[ypar][~bad_mask])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            axes[j, i].text(
                0.05,
                0.82,
                r"$\rho = %.2f$" % spearmanr(xpar, data[ypar])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_FM90.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")


def plot_feat_H(outpath, feat_name, data, bad_mask):
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

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    Returns
    -------
    Plots with feature properties vs. hydrogen measurements
    """
    # define the parameters to be plotted
    # with f(H2)
    xpars = ["logNH", "logNHI", "logNH2", "fH2"]
    xlabels = [
        r"$\text{log}(N(\text{H}))$",
        r"$\text{log}(N(\text{HI}))$",
        r"$\text{log}(N(\text{H}_2))$",
        r"$f(\text{H}_2) = 2N(\text{H}_2)/N(\text{H})$",
    ]
    # with tau/AV, lambda_0 and FWHM, area and area/AV
    # ypars = ["tau", "tau/AV", "x_0(micron)", "FWHM(micron)", "area(micron)", "area/AV"]
    # ylabels = [
    #     r"$\tau(\lambda_0)$",
    #     r"$\tau(\lambda_0)/A(V)$",
    #     r"$\lambda_0$ ($\mu$m)",
    #     r"FWHM ($\mu$m)",
    #     r"area ($\mu$m)",
    #     r"area/$A(V)$ ($\mu$m)",
    # ]
    # only tau
    ypars = ["tau"]
    ylabels = [r"$\tau(\lambda_0)$"]

    # create the figure
    fs = 20
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # add the x-axis label
        if len(ypars) == 1:
            ax = axes[i]
        else:
            ax = axes[-1, i]
        ax.set_xlabel(xlabel, fontsize=fs)

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

            if len(ypars) == 1:
                ax = axes[i]
            else:
                ax = axes[j, i]

            # plot the data
            ax.errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data["e_" + xpar][~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            ax.errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=data["e_" + xpar][bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
            )

            # calculate the Spearman correlation coefficient
            ax.text(
                0.05,
                0.9,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            ax.text(
                0.05,
                0.82,
                r"$\rho = %.2f$" % spearmanr(data[xpar], data[ypar])[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )

            # add the y-axis label (once)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_H.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")


def plot_feat_dcol(outpath, feat_name, data, bad_mask, grmod_table, lit_table):
    """
    Function to plot the feature properties vs. the dust column densities and ratios

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    grmod_table : astropy Table
        Data from grain models

    lit_table : astropy Table
        Literature measurements

    Returns
    -------
    Plots with feature properties vs. dust column densities and ratios
    """
    # define the parameters to be plotted
    # with the ratios
    # xpars = ["N(Mg)_d", "N(Fe)_d", "N(O)_d", "N(Mg)/N(Fe)", "N(Mg)/N(O)", "N(Fe)/N(O)"]
    # xlabels = [
    #     "N(Mg)$_{dust}$",
    #     "N(Fe)$_{dust}$",
    #     "N(O)$_{dust}$",
    #     "N(Mg)/N(Fe)",
    #     "N(Mg)/N(O)",
    #     "N(Fe)/N(O)",
    # ]
    # without the ratios
    xpars = ["N(Mg)_d", "N(Fe)_d", "N(O)_d"]
    xlabels = [
        r"$N(\text{Mg})_{\text{dust}}$",
        r"$N(\text{Fe})_{\text{dust}}$",
        r"$N(\text{O})_{\text{dust}}$",
    ]
    # with lambda_0, FWHM and area
    # ypars = ["tau", "x_0(micron)", "FWHM(micron)", "area(micron)"]
    # ylabels = [
    #     r"$\tau(\lambda_0)$",
    #     r"$\lambda_0$ ($\mu$m)",
    #     r"FWHM ($\mu$m)",
    #     r"area ($\mu$m)",
    # ]
    # only tau
    ypars = ["tau"]
    ylabels = [r"$\tau(\lambda_0)$"]

    # create the figure
    fs = 20
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    # entirely remove bad star when plotting measured data
    del_mask = data["name"] == "HD014434"

    # combine the measured silicate feature properties with the literature dust column densities
    lit_data = join(data, lit_table, keys="star", table_names=["mead", "lit"])
    # grey out the bad star in the literature data
    lit_mask = lit_data["name"] == "HD014434"

    # create a new table
    table_slopes = Table()

    for i, (xpar, xlabel) in enumerate(zip(xpars, xlabels)):
        # add the x-axis label
        if len(ypars) == 1:
            ax = axes[i]
        else:
            ax = axes[-1, i]
        ax.set_xlabel(xlabel, fontsize=fs)

        for j, (ypar, ylabel) in enumerate(zip(ypars, ylabels)):
            # obtain the y-axis uncertainties
            if "(" in ypar:
                index = ypar.find("(")
                yunc = (
                    data[ypar[:index] + "_unc_min" + ypar[index:]],
                    data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
                yunc_lit = (
                    lit_data[ypar[:index] + "_unc_min" + ypar[index:]],
                    lit_data[ypar[:index] + "_unc_plus" + ypar[index:]],
                )
            else:
                yunc = data[ypar + "_unc_min"], data[ypar + "_unc_plus"]
                yunc_lit = lit_data[ypar + "_unc_min"], lit_data[ypar + "_unc_plus"]

            if len(ypars) == 1:
                ax = axes[i]
            else:
                ax = axes[j, i]

            # plot the measured data
            # axes[j, i].errorbar(
            #     data[xpar][~bad_mask],
            #     data[ypar][~bad_mask],
            #     yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
            #     fmt="ok",
            #     markeredgewidth=0,
            #     elinewidth=0.4,
            # )
            # axes[j, i].errorbar(
            #     data[xpar][bad_mask & ~del_mask],
            #     data[ypar][bad_mask & ~del_mask],
            #     yerr=(yunc[0][bad_mask & ~del_mask], yunc[1][bad_mask & ~del_mask]),
            #     fmt="ok",
            #     markeredgewidth=0,
            #     alpha=0.25,
            #     elinewidth=0.4,
            # )

            # plot the literature data
            ax.errorbar(
                lit_data[xpar + "_lit"][~lit_mask],
                lit_data[ypar][~lit_mask],
                xerr=(
                    lit_data[xpar + "_munc"][~lit_mask],
                    lit_data[xpar + "_punc"][~lit_mask],
                ),
                yerr=(yunc_lit[0][~lit_mask], yunc_lit[1][~lit_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            ax.errorbar(
                lit_data[xpar + "_lit"][lit_mask],
                lit_data[ypar][lit_mask],
                xerr=(
                    lit_data[xpar + "_munc"][lit_mask],
                    lit_data[xpar + "_punc"][lit_mask],
                ),
                yerr=(yunc_lit[0][lit_mask], yunc_lit[1][lit_mask]),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
            )

            # calculate the Spearman correlation coefficient for the measured data
            # axes[j, i].text(
            #     0.05,
            #     0.9,
            #     r"$\rho = %.2f$"
            #     % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
            #     transform=axes[j, i].transAxes,
            #     fontsize=fs * 0.8,
            #     ha="left",
            # )
            # axes[j, i].text(
            #     0.05,
            #     0.82,
            #     r"$\rho = %.2f$"
            #     % spearmanr(data[xpar][~del_mask], data[ypar][~del_mask])[0],
            #     transform=axes[j, i].transAxes,
            #     fontsize=fs * 0.8,
            #     ha="left",
            #     alpha=0.25,
            # )
            # calculate the Spearman correlation coefficient for the literature data
            nan_mask = np.isnan(lit_data[xpar + "_lit"])
            ax.text(
                0.05,
                0.9,
                r"$\rho = %.2f$"
                % spearmanr(
                    lit_data[xpar + "_lit"][~nan_mask & ~lit_mask],
                    lit_data[ypar][~nan_mask & ~lit_mask],
                )[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )
            ax.text(
                0.05,
                0.82,
                r"$\rho = %.2f$"
                % spearmanr(
                    lit_data[xpar + "_lit"][~nan_mask],
                    lit_data[ypar][~nan_mask],
                )[0],
                transform=ax.transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )

            # fit the data with a line
            fitter = LinearLSQFitter(calc_uncertainties=True)
            line_mod = Linear1D()
            line_mod_0 = Linear1D(intercept=0, fixed={"intercept": True})

            # measured data
            # xs = data[xpar][~del_mask] * 1e-17
            # ys = data[ypar][~del_mask]
            # literature data
            xs = lit_data[xpar + "_lit"][~nan_mask & ~lit_mask] * 1e-17
            ys = lit_data[ypar][~nan_mask & ~lit_mask]
            fitted_line = fitter(line_mod, xs, ys)
            fitted_line_0 = fitter(line_mod_0, xs, ys)
            # measured data
            # axes[j, i].plot(xs * 1e17, fitted_line(xs), c="k")
            # literature data
            xs_plot = np.arange(np.min(xs) * 0.9, np.max(xs) * 1.1, 0.1)
            ax.plot(xs_plot * 1e17, fitted_line(xs_plot), c="k", label="data")
            ax.plot(
                xs_plot * 1e17,
                fitted_line_0(xs_plot),
                c="grey",
                label="int.=0",
                lw=0.5,
                ls=":",
            )
            ax.text(
                0.4,
                0.9,
                r"y=%.1ex+%.2f"
                % (
                    fitted_line.slope.value * 1e-17,
                    fitted_line.intercept.value,
                ),
                transform=ax.transAxes,
                fontsize=fs * 0.7,
            )
            # ax.text(
            #     0.5,
            #     0.1,
            #     r"%.3f / %.5f "
            #     % (fitted_line.intercept.std, fitter.fit_info["residuals"]),
            #     transform=ax.transAxes,
            #     fontsize=fs * 0.7,
            # )

            # create a list for the slopes
            slope_list = [fitted_line.slope.value * 1e-17]

            # add dust grain models
            for model in grmod_table:
                AVs = np.array([1.2, 2.5])
                ax.plot(
                    model[xpar.replace("_d", "/AV")] * AVs,
                    model[ypar] * AVs,
                    label=model["model"],
                )
                slope_list.append(model[ypar] / model[xpar.replace("_d", "/AV")])

            # save the slopes to the table
            table_slopes[
                ypar.replace("(micron)", "") + "/" + xpar.replace("_d", "")
            ] = slope_list

            # add the y-axis label (once)
            if i == 0:
                ax.set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_dcol.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    if len(ypars) == 1:
        ax = axes[-1]
    else:
        ax = axes[-1, -1]
    ax.legend(loc=4, fontsize=fs * 0.65, labelspacing=0.3)
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")

    # calculate the ratios of the slopes
    table_slopes["N(Mg)/N(Fe)"] = table_slopes["tau/N(Fe)"] / table_slopes["tau/N(Mg)"]
    table_slopes["N(Mg)/N(O)"] = table_slopes["tau/N(O)"] / table_slopes["tau/N(Mg)"]
    table_slopes["N(Fe)/N(O)"] = table_slopes["tau/N(O)"] / table_slopes["tau/N(Fe)"]
    table_slopes["Mg"] = [
        "%.1f" % table_slopes["N(Mg)/N(Fe)"][0],
        str(round(table_slopes["N(Mg)/N(Fe)"][1])),
        str(round(table_slopes["N(Mg)/N(Fe)"][2])),
        "%.1f" % table_slopes["N(Mg)/N(Fe)"][3],
        "%.1f" % table_slopes["N(Mg)/N(Fe)"][4],
    ]
    table_slopes["Fe"] = [1, 1, 1, 1, 1]
    table_slopes["O"] = [
        "%.1f" % (1 / table_slopes["N(Fe)/N(O)"][0]),
        str(round(1 / table_slopes["N(Fe)/N(O)"][1])),
        str(round(1 / table_slopes["N(Fe)/N(O)"][2])),
        "%.1f" % (1 / table_slopes["N(Fe)/N(O)"][3]),
        str(round(1 / table_slopes["N(Fe)/N(O)"][4])),
    ]

    # write the table to a file
    table_slopes.add_column(["data", *grmod_table["model"]], name="Line", index=0)
    table_slopes.write(
        outpath + "stoichiometries.tex",
        format="aastex",
        col_align="l|ccccccr@{ :}c@{: }l",
        latexdict={
            "tabletype": "deluxetable*",
            "caption": r"Slopes of lines in Fig.~\ref{fig:feat_dcol} for the fitted line to the data points, and for the dust grain models, and ratios between slopes. \label{tab:grmods}",
        },
        formats={
            r"$\tau/N$(Mg)": "%.1e",
            r"$\tau/N$(Fe)": "%.1e",
            r"$\tau/N$(O)": "%.1e",
            r"$N$(Mg)/$N$(Fe)": ".2f",
            r"$N$(Mg)/$N$(O)": ".2f",
            r"$N$(Fe)/$N$(O)": ".2f",
        },
        names=[
            "line",
            r"$\tau/N$(Mg)",
            r"$\tau/N$(Fe)",
            r"$\tau/N$(O)",
            r"$N$(Mg)/$N$(Fe)",
            r"$N$(Mg)/$N$(O)",
            r"$N$(Fe)/$N$(O)",
            "Mg :",
            " Fe :",
            "O",
        ],
        overwrite=True,
    )


def plot_feat_norm_dcol(outpath, feat_name, data, bad_mask):
    """
    Function to plot the (normalized) feature properties vs. the normalized dust column densities and ratios

    Parameters
    ----------
    outpath : string
        Path to store the plot

    feat_name : string
        Reference name for the feature

    data : astropy Table
        Data to plot

    bad_mask : numpy.ndarray
        Mask of stars with noisy data

    Returns
    -------
    Plots with (normalized) feature properties vs. normalized dust column densities and ratios
    """
    # define the parameters to be plotted
    xpars = [
        "N(Mg)/AV",
        "N(Fe)/AV",
        "N(O)/AV",
        "N(Mg)/N(Fe)",
        "N(Mg)/N(O)",
        "N(Fe)/N(O)",
    ]
    xlabels = [
        "N(Mg)$_{dust}/A(V)$",
        "N(Fe)$_{dust}/A(V)$",
        "N(O)$_{dust}/A(V)$",
        "N(Mg)/N(Fe)",
        "N(Mg)/N(O)",
        "N(Fe)/N(O)",
    ]
    ypars = ["tau/AV", "x_0(micron)", "FWHM(micron)", "area/AV"]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"FWHM ($\mu$m)",
        r"area/$A(V)$ ($\mu$m)",
    ]

    # create the figure
    fs = 20
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)
    fig, axes = plt.subplots(
        len(ypars),
        len(xpars),
        figsize=(4 * len(xpars), 4 * len(ypars)),
        sharex="col",
        sharey="row",
    )

    # entirely remove bad star
    del_mask = data["name"] == "HD014434"

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

            # plot the data
            axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
                markeredgewidth=0,
                elinewidth=0.4,
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask & ~del_mask],
                data[ypar][bad_mask & ~del_mask],
                yerr=(yunc[0][bad_mask & ~del_mask], yunc[1][bad_mask & ~del_mask]),
                fmt="ok",
                markeredgewidth=0,
                alpha=0.25,
                elinewidth=0.4,
            )

            # calculate the Spearman correlation coefficient
            axes[j, i].text(
                0.05,
                0.9,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~bad_mask], data[ypar][~bad_mask])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
            )

            axes[j, i].text(
                0.05,
                0.82,
                r"$\rho = %.2f$"
                % spearmanr(data[xpar][~del_mask], data[ypar][~del_mask])[0],
                transform=axes[j, i].transAxes,
                fontsize=fs * 0.8,
                ha="left",
                alpha=0.25,
            )

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_norm_dcol.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")


def calc_ave(data):
    """
    Function to calculate and print out some median values

    Parameters
    ----------
    data : astropy Table
        Data for which to calculate medians

    Returns
    -------
    Median values of relevant parameters
    """
    print("Median lambda_0: ", np.median(data["x_0(micron)"]))
    print("Median tau: ", np.median(data["tau"]))
    print(
        "Median and standard deviation FWHM: ",
        np.median(data["FWHM(micron)"]),
        np.std(data["FWHM(micron)"]),
    )
    print("Median area: ", np.median(data["area(micron)"]))
    print("Median alpha: ", np.median(data["shape"]))
    print("Median tau/AV: ", np.median(data["tau/AV"]))
    print(
        "Median and standard deviation AV/tau: ",
        np.median(1 / data["tau/AV"]),
        np.std(1 / data["tau/AV"]),
    )


def main():
    # plotting settings for uniform plots
    # fs = 20
    # plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    # plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)

    # define the data path and the output path
    datapath = "/Users/mdecleir/Documents/MEAD/Extinction/JWST_data/"
    litpath = "/Users/mdecleir/Documents/MEAD/Literature_data/"
    outpath = "/Users/mdecleir/Documents/MEAD/Plots/"

    # obtain the extinction properties
    ext_table = Table.read(litpath + "Gordon+2009_tab2.dat", format="ascii")

    # obtain the FM90 parameters
    fm90_table = Table.read(litpath + "Gordon+2009_tab3.dat", format="ascii")

    # obtain the hydrogen measurements
    hyd_table = Table.read(litpath + "VanDePutte+2023_tab2.dat", format="ascii")

    # obtain the depletion measurements
    dep_table = Table.read(
        "/Users/mdecleir/stis_depletions/depletions.dat", format="ascii"
    )

    # obtain the fitting results of the features
    feat_58 = Table.read(datapath + "fit_results_58.txt", format="ascii")
    feat_10 = Table.read(datapath + "fit_results_10_gauss_skew.txt", format="ascii")

    # merge tables
    joined_ext_58 = join(ext_table, feat_58, keys_left="Name", keys_right="name")
    joined_ext_10 = join(ext_table, feat_10, keys_left="Name", keys_right="name")
    joined_fm90_10 = join(joined_ext_10, fm90_table, keys="Name")
    joined_hyd_10 = join(joined_fm90_10, hyd_table, keys_left="Name", keys_right="Star")
    joined_all_10 = join(joined_hyd_10, dep_table, keys_left="Name", keys_right="star")

    # calculate the uncertainties on A(V) and R(V)
    joined_all_10["AV_unc"] = np.sqrt(
        joined_all_10["AV_runc"] ** 2 + joined_all_10["AV_sunc"] ** 2
    )
    joined_all_10["RV_unc"] = np.sqrt(
        joined_all_10["RV_runc"] ** 2 + joined_all_10["RV_sunc"] ** 2
    )

    # calculate A(1500 Angstrom)
    (joined_all_10["A1500"], joined_all_10["A1500_unc"]) = calc_A1500(joined_all_10)

    # calculate normalized optical depth and area, and uncertainties
    joined_all_10["tau/AV"] = joined_all_10["tau"] / joined_all_10["AV"]
    joined_all_10["area/AV"] = joined_all_10["area(micron)"] / joined_all_10["AV"]
    joined_all_10["tau/AV_unc_min"] = joined_all_10["tau/AV"] * np.sqrt(
        (joined_all_10["tau_unc_min"] / joined_all_10["tau"]) ** 2
        + (joined_all_10["AV_unc"] / joined_all_10["AV"]) ** 2
    )
    joined_all_10["tau/AV_unc_plus"] = joined_all_10["tau/AV"] * np.sqrt(
        (joined_all_10["tau_unc_plus"] / joined_all_10["tau"]) ** 2
        + (joined_all_10["AV_unc"] / joined_all_10["AV"]) ** 2
    )
    joined_all_10["area/AV_unc_min"] = joined_all_10["area/AV"] * np.sqrt(
        (joined_all_10["area_unc_min(micron)"] / joined_all_10["area(micron)"]) ** 2
        + (joined_all_10["AV_unc"] / joined_all_10["AV"]) ** 2
    )
    joined_all_10["area/AV_unc_plus"] = joined_all_10["area/AV"] * np.sqrt(
        (joined_all_10["area_unc_plus(micron)"] / joined_all_10["area(micron)"]) ** 2
        + (joined_all_10["AV_unc"] / joined_all_10["AV"]) ** 2
    )

    # calculate the uncertainty on f(H2)
    joined_all_10["e_fH2"] = (
        joined_all_10["fH2"]
        * np.log(10)
        * np.sqrt(joined_all_10["e_logNH2"] ** 2 + joined_all_10["e_logNH"] ** 2)
    )

    # obtain literature dust column densities
    dcol_table = Table.read(litpath + "lit_dcols.dat", format="ascii")

    # calculate dust column density ratios and normalized dust column densities
    joined_all_10["N(Mg)/N(Fe)"] = joined_all_10["N(Mg)_d"] / joined_all_10["N(Fe)_d"]
    joined_all_10["N(Mg)/N(O)"] = joined_all_10["N(Mg)_d"] / joined_all_10["N(O)_d"]
    joined_all_10["N(Fe)/N(O)"] = joined_all_10["N(Fe)_d"] / joined_all_10["N(O)_d"]
    dcol_table["N(Mg)/N(Fe)"] = dcol_table["N(Mg)_d"] / dcol_table["N(Fe)_d"]
    dcol_table["N(Mg)/N(O)"] = dcol_table["N(Mg)_d"] / dcol_table["N(O)_d"]
    dcol_table["N(Fe)/N(O)"] = dcol_table["N(Fe)_d"] / dcol_table["N(O)_d"]
    joined_all_10["N(Mg)/AV"] = joined_all_10["N(Mg)_d"] / joined_all_10["AV"]
    joined_all_10["N(Fe)/AV"] = joined_all_10["N(Fe)_d"] / joined_all_10["AV"]
    joined_all_10["N(O)/AV"] = joined_all_10["N(O)_d"] / joined_all_10["AV"]

    # obtain literature silicate results
    g21_sil_table = Table.read(litpath + "Gordon+2021_tab6.dat", format="ascii")
    g21_ext_table = Table.read(litpath + "Gordon+2021_tab5.dat", format="ascii")
    g21_table = join(g21_ext_table, g21_sil_table, keys="Name")

    # convert literature A(lambda)/A(V) to tau
    g21_table["tau"] = (
        2.5 * np.log10(math.e) * g21_table["S1x100"] * g21_table["A(V)"] / 100
    )
    g21_table["tau_unc_min"] = g21_table["tau"] * np.sqrt(
        (g21_table["S1_unc_min"] / g21_table["S1x100"]) ** 2
        + (g21_table["A(V)_unc_min"] / g21_table["A(V)"]) ** 2
    )
    g21_table["tau_unc_plus"] = g21_table["tau"] * np.sqrt(
        (g21_table["S1_unc_plus"] / g21_table["S1x100"]) ** 2
        + (g21_table["A(V)_unc_plus"] / g21_table["A(V)"]) ** 2
    )
    g21_table["tau/AV"] = 2.5 * np.log10(math.e) * g21_table["S1x100"] / 100
    g21_table["tau/AV_unc_min"] = 2.5 * np.log10(math.e) * g21_table["S1_unc_min"] / 100
    g21_table["tau/AV_unc_plus"] = (
        2.5 * np.log10(math.e) * g21_table["S1_unc_plus"] / 100
    )

    # calculate dust grain model dust column densities
    grmod_col = Table.read(litpath + "grain_model_abundances.dat", format="ascii")
    grmod_fit = Table.read(datapath + "fit_results_mod.txt", format="ascii")
    grmod_table = join(grmod_col, grmod_fit, keys_left="model", keys_right="name")
    grmod_table["N(Mg)/AV"] = (
        grmod_table["Mg"] * 1e-6 * grmod_table["N(H)/E(B-V)"] / 3.1
    )
    grmod_table["N(Fe)/AV"] = (
        grmod_table["Fe"] * 1e-6 * grmod_table["N(H)/E(B-V)"] / 3.1
    )
    grmod_table["N(O)/AV"] = grmod_table["O"] * 1e-6 * grmod_table["N(H)/E(B-V)"] / 3.1
    grmod_table = grmod_table[[0, 3, 1, 2]]

    # define the stars that should be masked
    bad_stars = ["HD014434", "HD038087"]
    bad_mask = np.isin(feat_10["name"], bad_stars)

    # plot the feature properties against each other
    plot_feat(outpath, "10", joined_all_10, bad_mask)

    # calculate average silicate feature properties
    calc_ave(joined_all_10)

    # plot silicate properties with literature data added
    plot_sil_lit(outpath, joined_all_10, bad_mask, g21_table)

    # create plots vs. A(V), A(1500A) and R(V)
    plot_feat_AV_RV(outpath, "10", joined_all_10, bad_mask)
    plot_feat_AV_RV_lit(outpath, "10", joined_all_10, bad_mask, g21_table)
    plot_feat_norm_AV_RV(outpath, "10", joined_all_10, bad_mask, g21_table)

    # create plots vs. the FM90 parameters
    plot_feat_FM90(outpath, "10", joined_all_10, bad_mask)

    # create plots vs. hydrogen measurements
    plot_feat_H(outpath, "10", joined_all_10, bad_mask)

    # create plots vs. dust column densities
    plot_feat_dcol(outpath, "10", joined_all_10, bad_mask, grmod_table, dcol_table)
    plot_feat_norm_dcol(outpath, "10", joined_all_10, bad_mask)


if __name__ == "__main__":
    main()
