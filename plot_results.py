# plot_results.py: script to plot results

# import the necessary packages
import astropy.units as u
import math
import numpy as np
import os

from astropy.table import Table, join
from dust_extinction.shapes import FM90
from matplotlib import pyplot as plt
from scipy.stats import spearmanr


def calc_A1500(data):
    """
    Function to calculate the absolute extinction at 1500 Angstrom

    Parameters
    ----------
    data : astropy Table
        data including FM90 extinction parameters

    Returns
    -------
    A1500 : numpy.ndarray
        Absolute extinction at 1500 Angstrom for all stars
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
    A1500 = E1500 * data["EBV"].value + data["AV"].value

    return A1500, 0.03 * A1500


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
    xpars = ["x_0(micron)", "tau", "shape"]
    xlabels = [r"$\lambda_0$ ($\mu$m)", r"$\tau(\lambda_0)$", r"$\alpha$"]
    ypars = ["tau", "shape", "area(micron)"]
    ylabels = [r"$\tau(\lambda_0)$", r"$\alpha$", r"area ($\mu$m)"]

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
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=(xunc[0][bad_mask], xunc[1][bad_mask]),
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                alpha=0.25,
                markeredgewidth=0,
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
        data["x_0(micron)"][~bad_mask],
        data["tau"][~bad_mask],
        xerr=(
            data["x_0_unc_min(micron)"][~bad_mask],
            data["x_0_unc_plus(micron)"][~bad_mask],
        ),
        yerr=(data["tau_unc_min"][~bad_mask], data["tau_unc_plus"][~bad_mask]),
        fmt="ok",
        markeredgewidth=0,
        elinewidth=0.4,
        label="this work",
    )
    ax.errorbar(
        data["x_0(micron)"][bad_mask],
        data["tau"][bad_mask],
        xerr=(
            data["x_0_unc_min(micron)"][bad_mask],
            data["x_0_unc_plus(micron)"][bad_mask],
        ),
        yerr=(data["tau_unc_min"][bad_mask], data["tau_unc_plus"][bad_mask]),
        fmt="ok",
        alpha=0.25,
        markeredgewidth=0,
        elinewidth=0.4,
    )

    # add literature values
    ax.errorbar(
        lit_table["lambda_o1"],
        lit_table["tau"],
        xerr=(lit_table["lambda_o1_unc_min"], lit_table["lambda_o1_unc_plus"]),
        yerr=(lit_table["tau_unc_min"], lit_table["tau_unc_plus"]),
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
    ax.set_xlabel(r"$\lambda_0$ ($\mu$m)", fontsize=fs)
    ax.set_ylabel(r"$\tau(\lambda_0)$", fontsize=fs)
    ax.tick_params(axis="both", labelsize=fs * 0.8)
    plt.legend(bbox_to_anchor=(0.5, 0.0, 0.5, 0.87))
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
    xpars = ["AV", "A1500", "RV"]
    xlabels = ["$A(V)$", r"$A(1500\AA)$", "$R(V)$"]
    ypars = ["x_0(micron)", "tau", "FWHM(micron)", "area(micron)"]
    ylabels = [
        r"$\lambda_0$ ($\mu$m)",
        r"$\tau(\lambda_0)$",
        r"FWHM ($\mu$m)",
        r"area ($\mu$m)",
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

            # plot the data
            axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data[xpar + "_unc"][~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=data[xpar + "_unc"][bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                alpha=0.25,
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


def plot_feat_norm_AV_RV(outpath, feat_name, data, bad_mask):
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
    xpars = ["AV", "A1500", "RV"]
    xlabels = ["$A(V)$", r"$A(1500\AA)$", "$R(V)$"]
    ypars = ["tau/AV", "area/AV"]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"area/$A(V)$ ($\mu$m)",
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
            # plot the data
            axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data[xpar + "_unc"][~bad_mask],
                yerr=(
                    data[ypar + "_unc_min"][~bad_mask],
                    data[ypar + "_unc_plus"][~bad_mask],
                ),
                fmt="ok",
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
                alpha=0.25,
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

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_norm_AV_A1500_RV.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
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

    ypars = [
        "tau/AV",
        "x_0(micron)",
        "FWHM(micron)",
        "area/AV",
    ]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"FWHM ($\mu$m)",
        r"area/$A(V)$ ($\mu$m)",
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

            # plot the data
            axes[j, i].errorbar(
                xpar[~bad_mask],
                data[ypar][~bad_mask],
                xerr=xunc[~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
            )
            axes[j, i].errorbar(
                xpar[bad_mask],
                data[ypar][bad_mask],
                xerr=xunc[bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                alpha=0.25,
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
        "tau/AV",
        "x_0(micron)",
        "FWHM(micron)",
        "area/AV",
    ]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"FWHM ($\mu$m)",
        r"area/$A(V)$ ($\mu$m)",
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

            # plot the data
            axes[j, i].errorbar(
                data[xpar][~bad_mask],
                data[ypar][~bad_mask],
                xerr=data["e_" + xpar][~bad_mask],
                yerr=(yunc[0][~bad_mask], yunc[1][~bad_mask]),
                fmt="ok",
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask],
                data[ypar][bad_mask],
                xerr=data["e_" + xpar][bad_mask],
                yerr=(yunc[0][bad_mask], yunc[1][bad_mask]),
                fmt="ok",
                alpha=0.25,
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

            # add the y-axis label (once)
            if i == 0:
                axes[j, 0].set_ylabel(ylabel, fontsize=fs)

    # rename the previous version of the plot
    outname = outpath + feat_name + "_H.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")


def plot_feat_dcol(outpath, feat_name, data, bad_mask):
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

    Returns
    -------
    Plots with feature properties vs. dust column densities and ratios
    """
    # define the parameters to be plotted
    xpars = ["N(Mg)_d", "N(Fe)_d", "N(O)_d", "N(Mg)/N(Fe)", "N(Mg)/N(O)", "N(Fe)/N(O)"]
    xlabels = [
        "N(Mg)$_{dust}$",
        "N(Fe)$_{dust}$",
        "N(O)$_{dust}$",
        "N(Mg)/N(Fe)",
        "N(Mg)/N(O)",
        "N(Fe)/N(O)",
    ]

    ypars = [
        "tau",
        "x_0(micron)",
        "FWHM(micron)",
        "area(micron)",
    ]
    ylabels = [
        r"$\tau(\lambda_0)$",
        r"$\lambda_0$ ($\mu$m)",
        r"FWHM ($\mu$m)",
        r"area ($\mu$m)",
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
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask & ~del_mask],
                data[ypar][bad_mask & ~del_mask],
                yerr=(yunc[0][bad_mask & ~del_mask], yunc[1][bad_mask & ~del_mask]),
                fmt="ok",
                alpha=0.25,
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
    outname = outpath + feat_name + "_dcol.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")

    # finalize and save the figure
    fig.subplots_adjust(hspace=0, wspace=0)
    fig.savefig(outname, bbox_inches="tight")


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

    ypars = [
        "tau/AV",
        "x_0(micron)",
        "FWHM(micron)",
        "area/AV",
    ]
    ylabels = [
        r"$\tau(\lambda_0)/A(V)$",
        r"$\lambda_0$ ($\mu$m)",
        r"FWHM ($\mu$m)",
        r"area/$A(V)$ ($\mu$m)",
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
            )
            axes[j, i].errorbar(
                data[xpar][bad_mask & ~del_mask],
                data[ypar][bad_mask & ~del_mask],
                yerr=(yunc[0][bad_mask & ~del_mask], yunc[1][bad_mask & ~del_mask]),
                fmt="ok",
                alpha=0.25,
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
    joined_all_10["A1500"], joined_all_10["A1500_unc"] = calc_A1500(joined_all_10)

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

    # calculate dust column density ratios and normalized dust column densities
    joined_all_10["N(Mg)/N(Fe)"] = joined_all_10["N(Mg)_d"] / joined_all_10["N(Fe)_d"]
    joined_all_10["N(Mg)/N(O)"] = joined_all_10["N(Mg)_d"] / joined_all_10["N(O)_d"]
    joined_all_10["N(Fe)/N(O)"] = joined_all_10["N(Fe)_d"] / joined_all_10["N(O)_d"]
    joined_all_10["N(Mg)/AV"] = joined_all_10["N(Mg)_d"] / joined_all_10["AV"]
    joined_all_10["N(Fe)/AV"] = joined_all_10["N(Fe)_d"] / joined_all_10["AV"]
    joined_all_10["N(O)/AV"] = joined_all_10["N(O)_d"] / joined_all_10["AV"]

    # obtain literature silicate results
    g21_sil_table = Table.read(
        litpath + "Gordon+2021_tab6.dat",
        format="ascii",
    )
    g21_ext_table = Table.read(
        litpath + "Gordon+2021_tab5.dat",
        format="ascii",
    )
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

    # define the stars that should be masked
    bad_stars = ["HD014434", "HD038087"]
    bad_mask = np.isin(feat_10["name"], bad_stars)

    # plot the feature properties against each other
    plot_feat(outpath, "10", joined_all_10, bad_mask)

    # create plots vs. A(V), A(1500A) and R(V)
    plot_feat_AV_RV(outpath, "10", joined_all_10, bad_mask)
    plot_feat_norm_AV_RV(outpath, "10", joined_all_10, bad_mask)

    # create plots vs. the FM90 parameters
    plot_feat_FM90(outpath, "10", joined_all_10, bad_mask)

    # create plots vs. hydrogen measurements
    plot_feat_H(outpath, "10", joined_all_10, bad_mask)

    # create plots vs. dust column densities
    plot_feat_dcol(outpath, "10", joined_all_10, bad_mask)
    plot_feat_norm_dcol(outpath, "10", joined_all_10, bad_mask)

    # plot silicate properties with literature data added
    plot_sil_lit(outpath, joined_all_10, bad_mask, g21_table)


if __name__ == "__main__":
    main()
