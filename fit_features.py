# fit_features.py: script to fit the features in the JWST MEAD spectra

# import the necessary packages
import numpy as np
import os

from astropy.table import Table
from astropy.modeling.models import custom_model, Drude1D, Gaussian1D, Polynomial1D
from astropy.modeling.fitting import (
    FittingWithOutlierRemoval,
    LevMarLSQFitter,
    LinearLSQFitter,
)
from astropy.stats import sigma_clip
from matplotlib import pyplot as plt
from models_mcmc_extension import EmceeFitter
from scipy import stats


def gamma(x, x_o=1, gamma_o=1, asym=1):
    """
    Gamma function (wavelength dependent width, replacing the FWHM)
    """
    return 2 * gamma_o / (1 + np.exp(asym * (x - x_o)))


def drude_modified(x, scale=1, x_o=1, gamma_o=1, asym=1):
    """
    Modified Drude profile
    """
    y = (
        scale
        * (gamma(x, x_o, gamma_o, asym) / x_o) ** 2
        / ((x / x_o - x_o / x) ** 2 + (gamma(x, x_o, gamma_o, asym) / x_o) ** 2)
    )
    return y


def gauss_skew_func(x, amplitude=0.1, loc=9, scale=1, alpha=2):
    return amplitude * stats.skewnorm.pdf(x, alpha, loc=loc, scale=scale)


def _wavegrid(waverange, resolution):
    """
    Function to define a wavelength grid with a specified resolution between
    a minimum and maximum wavelength

    Parameters
    ----------
    waverange : tuple of floats (min,max)
        Minimum, maximum wavelength of the grid

    resolution : float
        Resolution of grid

    Returns
    -------
    wave_info : tuple of floats (waves, waves_bin_min, waves_bin_max)
        Wavelength grid centers, minimum and maximum bin wavelengths
    """
    # calculate the step size in log space
    # delta_l_log = log(l2) - log(l1) = log(l2/l1) = log((lc+S/2)/(lc-S/2))
    # = log((2R+1)/(2R-1))
    # S = step size in linear space; R = lc/S
    delta_wave_log = np.log10((2 * resolution + 1) / (2 * resolution - 1))

    # create the grid in log space
    wave_log10 = np.arange(
        np.log10(waverange[0]),
        np.log10(waverange[1]),
        delta_wave_log,
    )

    # calculate the grid in linear space
    full_wave_min = 10 ** wave_log10
    full_wave_max = 10 ** (wave_log10 + delta_wave_log)
    full_wave = (full_wave_min + full_wave_max) / 2

    return (full_wave, full_wave_min, full_wave_max)


def rebin_constres(data, waverange, resolution):
    """
    Function to rebin a spectrum to a fixed spectral resolution and within a specified wavelength range

    Parameters
    ----------
    data : Astropy Table
        Spectrum to be rebinned

    waverange : tuple of floats (min,max)
        Minimum, maximum wavelength of the rebinned wavelength grid

    resolution : float
        Spectral resolution of the rebinned spectrum

    Returns
    -------
    Rebinned wavelengths, fluxes and uncertainties
    """
    # read the data
    waves = data["wavelength"]
    fluxes = data["flux"]
    uncs = data["unc"]

    # set up the new wavelength grid
    full_wave, full_wave_min, full_wave_max = _wavegrid(waverange, resolution)
    n_waves = len(full_wave)

    # set up the new rebinned arrays
    new_waves = full_wave
    new_fluxes = np.full((n_waves), np.nan, dtype=float)
    new_uncs = np.full((n_waves), np.nan, dtype=float)

    # rebin using a weighted average
    for k in range(n_waves):
        bin_mask = (waves >= full_wave_min[k]) & (waves < full_wave_max[k])
        if np.sum(bin_mask) > 0:
            weights = 1 / np.square(uncs[bin_mask])
            sweights = np.nansum(weights)
            new_fluxes[k] = np.nansum(weights * fluxes[bin_mask]) / sweights
            new_uncs[k] = 1 / np.sqrt(sweights)

    return (new_waves, new_fluxes, new_uncs)


def fit_cont(waves, fluxes, cont_mask):
    """
    Function to fit the continuum, plot the continuum fit, normalize the spectrum, covert the normalized flux to optical depth, plot the optical depth, and calculate the empirical uncertainty on the optical depth

    Parameters
    ----------
    waves : astropy Table Column
        Wavelengths to fit

    fluxes : astropy Table Column
        Fluxes to fit

    cont_mask : numpy.ndarray
        Mask for data to use in the continuum fitting

    Returns
    -------
    taus : astropy Table Column
        Optical depths (in range_mask)

    unc : numpy.float64
        Empirical uncertainty on the optical depth

    axes : numpy.ndarray
        Plotting axes of the figure
    """
    # fit the continuum
    pol_mod = Polynomial1D(1)
    lin_fitter = LinearLSQFitter()
    out_fitter = FittingWithOutlierRemoval(lin_fitter, sigma_clip, niter=3, sigma=3)

    fit_result_cont, clipmask = out_fitter(
        pol_mod,
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
    )

    # plot the data
    fig, axes = plt.subplots(
        2,
        sharex=True,
        figsize=(9, 10),
        gridspec_kw={"hspace": 0},
    )
    axes[0].plot(waves, fluxes * waves ** 2, c="k", alpha=0.9)

    # plot the data points that were used in the continuum fitting
    axes[0].plot(
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
        "r.",
        markersize=8,
        alpha=0.8,
        label="fit points",
    )

    # cross the sigma clipped data points that were not used in the fitting
    axes[0].plot(
        waves[cont_mask][clipmask],
        fluxes[cont_mask][clipmask] * waves[cont_mask][clipmask] ** 2,
        "x",
        color="gray",
        alpha=0.8,
        label="clipped",
    )

    # plot the continuum fit
    axes[0].plot(
        waves,
        fit_result_cont(waves),
        c="tab:orange",
        label="cont. fit",
    )
    fs = 18
    axes[0].set_ylabel(r"$\lambda^2 F(\lambda)\: [\mu m^2 \:Jy]$", fontsize=fs)

    # normalize the fluxes
    norm_fluxes = fluxes * waves ** 2 / fit_result_cont(waves)

    # convert to optical depth (only exists in range_mask)
    taus = np.log(1 / norm_fluxes)

    # plot the optical depth
    axes[1].plot(waves, taus, c="k", alpha=0.9)
    axes[1].set_xlabel(r"wavelength ($\mu$m)", fontsize=fs)
    axes[1].set_ylabel("optical depth", fontsize=fs)
    axes[1].axhline(ls=":", c="k")

    # calculate the empirical uncertainy on the optical depth in clean regions
    unc = np.std(taus[cont_mask])

    return taus, unc, axes


def fit_feature(
    outpath, plotpath, star, model, waves, taus, uncs, axes, plot_waves, feat_name
):
    """
    Function to fit a feature using the Levenberg-Marquardt algorithm as initial guess and MCMC fitting for refining

    Parameters
    ----------
    outpath : string
        Path to store the output of the MCMC fitting

    plotpath : string
        Path to store the plot of the fitted feature

    star : string
        Star name

    model : astropy model
        Initial model to fit to the data

    waves : astropy.table.column.Column
        Wavelengths to fit

    taus : astropy.table.column.Column
        Optical depths to fit

    uncs : numpy.ndarray
        Uncertainties to use in the fit

    axes : numpy.ndarray
       Plotting axes of the figure

    plot_waves : astropy.table.column.Column

    feat_name : string
        Reference name for the feature

    Returns
    -------
    fit_result_feat_emcee : astropy model
        MCMC fitting results

    chains : numpy.ndarray
        MCMC chains (for amplitude, mean and stddev in that order)
    """
    # fit the feature with the LevMarLSQFitter
    lev_fitter = LevMarLSQFitter()
    fit_result_feat_lev = lev_fitter(
        model,
        waves,
        taus,
        weights=1 / uncs,
        maxiter=10000,
        filter_non_finite=True,
    )

    # fit the feature again with MCMC
    emcee_samples_file = (
        outpath + "Fitting_results/" + star + "_chains_" + feat_name + ".h5"
    )
    nsteps = 5000
    burn = 0.1
    emcee_fitter = EmceeFitter(
        nsteps=nsteps, burnfrac=burn, save_samples=emcee_samples_file
    )
    fit_result_feat_emcee = emcee_fitter(
        fit_result_feat_lev,
        waves,
        taus,
        weights=1 / uncs,
    )

    # calculate chi square
    residuals = fit_result_feat_emcee(waves) - taus
    chi2 = np.sum((residuals / uncs) ** 2)

    # plot the feature fit and save the figure
    axes[1].plot(plot_waves, fit_result_feat_emcee(plot_waves), c="crimson", lw=2)
    outname = plotpath + star + "_" + feat_name + "_fit.pdf"
    # rename the previous version of the plot
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")
    plt.savefig(outname, bbox_inches="tight")

    # obtain the MCMC chains
    chains = emcee_fitter.fit_info["sampler"].get_chain(
        flat=True, discard=np.int32(burn * nsteps)
    )

    # plot the MCMC fitting results
    emcee_fitter.plot_emcee_results(
        fit_result_feat_emcee,
        filebase=outpath + "Fitting_results/" + star + "_" + feat_name,
    )

    return fit_result_feat_emcee, chains, chi2


def fit_34(datapath, star):
    """
    Function to fit the continuum and the feature at 3.4 micron

    Parameters
    ----------
    datapath : string
        Path to the data files

    star : string
        Star name

    Returns
    -------
    fit_result_feat_emcee : astropy model
        MCMC fitting results

    chains : numpy.ndarray
        MCMC chains (for amplitude, mean and stddev in that order)

    Saves plots with data and fitted models
    """
    # obtain the data
    starname = "hd" + star.split("D")[1].strip("0")
    data = Table.read(datapath + "MIRI/v3/" + starname + "_nircam_mrs_merged.fits")
    waves = data["wavelength"]
    fluxes = data["flux"]

    # define masks for the continuum fitting
    rangemask = (waves > 3.25) & (waves <= 3.55)
    feat_reg_mask = (waves > 3.32) & (waves <= 3.5)
    stellar_lines_out = ((waves > 3.28) & (waves <= 3.32)) | (
        (waves > 5.947) & (waves <= 5.969)
    )
    # stellar line masks per star:
    # 5.89-5.913, 5.903-5.913, 5.887-5.922, 5.894-5.918, 5.898-5.926, 5.894-5.911
    # 5.948-5.957, 5.951-5.969, 5.947-5.964
    cont_mask = rangemask & ~feat_reg_mask & ~stellar_lines_out & ~np.isnan(fluxes)

    # fit the continuum
    pol_mod = Polynomial1D(3)
    lin_fitter = LinearLSQFitter()
    fit_result_cont = lin_fitter(
        pol_mod,
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
    )

    # plot the data and the continuum fit
    fig, axes = plt.subplots(
        2,
        sharex=True,
        figsize=(9, 10),
        gridspec_kw={"hspace": 0},
    )
    fs = 18
    axes[0].plot(
        waves[rangemask], fluxes[rangemask] * waves[rangemask] ** 2, c="k", alpha=0.9
    )
    axes[0].plot(
        waves[rangemask],
        fit_result_cont(waves[rangemask]),
        c="tab:orange",
        label="cont. fit",
    )

    # plot the data points that were used in the continuum fitting
    axes[0].plot(
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
        "r.",
        markersize=8,
        alpha=0.8,
        label="fit points",
    )
    axes[0].set_ylabel("flux (Jy)", fontsize=fs)

    # normalize the fluxes
    norm_fluxes = (
        fluxes[rangemask] * waves[rangemask] ** 2 / fit_result_cont(waves[rangemask])
    )

    # convert to optical depth (only exists in rangemask)
    taus = np.log(1 / norm_fluxes)

    # calculate the empirical uncertainy on the optical depth in clean regions
    unc = np.std(taus[cont_mask[rangemask]])

    # mask the stellar lines inside the feature and define the mask for the feature fitting
    stellar_lines_in = ((waves > 5.703) & (waves <= 5.717)) | (
        (waves > 5.823) & (waves <= 5.841)
    )
    # stellar line masks per star:
    # 5.705-5.717, 5.706-5.715, 5.703-716, 5.712-5.717, 5.711-5.717
    # 5.823-5.831, 5.824-5.836, 5.827-5.839, 5.829-5.841, 5.825-5.836, 5.823-5.840
    feat_full_mask = feat_reg_mask & ~stellar_lines_in
    feat_fit_mask = feat_full_mask[rangemask] & ~np.isnan(taus)

    # define the uncertainties
    uncs = np.full(len(taus[feat_fit_mask]), unc)

    # define the initial model to fit the feature
    gauss_mod = models.Gaussian1D(
        amplitude=0.01,
        stddev=0.02,
        mean=3.4,
        bounds={
            "mean": (3.38, 3.44),
            "stddev": (0.05, 0.11),
        },
    )

    # fit the feature
    fit_result_feat_emcee, chains = fit_feature(
        datapath + "MIRI/",
        star,
        gauss_mod,
        waves[rangemask][feat_fit_mask],
        taus[feat_fit_mask],
        uncs,
        "34",
    )

    # plot the feature fits
    axes[1].plot(waves[rangemask], taus, c="k", alpha=0.9)
    axes[1].plot(
        waves[rangemask], fit_result_feat_emcee(waves[rangemask]), c="crimson", lw=2
    )
    axes[1].set_xlabel(r"wavelength ($\mu$m)", fontsize=fs)
    axes[1].set_ylabel("optical depth", fontsize=fs)
    axes[1].axhline(ls=":", c="k")
    plt.savefig(datapath + star + "_34_fit.pdf", bbox_inches="tight")

    return fit_result_feat_emcee, chains


def fit_58(datapath, star):
    """
    Function to fit the continuum and the feature at 5.8 micron

    Parameters
    ----------
    datapath : string
        Path to the data files

    star : string
        Star name

    Returns
    -------
    fit_result_feat_emcee : astropy.modeling.functional_models.Gaussian1D
        MCMC fitting results

    chains : numpy.ndarray
        MCMC chains (for amplitude, mean and stddev in that order)

    Saves plots with data and fitted models
    """
    # obtain the data
    starname = "hd" + star.split("D")[1].strip("0")
    data = Table.read(datapath + "MIRI/v3/" + starname + "_nircam_mrs_merged.fits")
    waves = data["wavelength"]
    fluxes = data["flux"]

    # define masks for the continuum fitting
    rangemask = (waves > 5.6) & (waves <= 6.1)
    feat_reg_mask = (waves > 5.7) & (waves <= 5.9)
    stellar_lines_out = ((waves > 5.887) & (waves <= 5.926)) | (
        (waves > 5.947) & (waves <= 5.969)
    )
    # stellar line masks per star:
    # 5.89-5.913, 5.903-5.913, 5.887-5.922, 5.894-5.918, 5.898-5.926, 5.894-5.911
    # 5.948-5.957, 5.951-5.969, 5.947-5.964
    cont_mask = rangemask & ~feat_reg_mask & ~stellar_lines_out & ~np.isnan(fluxes)

    # fit the continuum
    pol_mod = models.Polynomial1D(3)
    lin_fitter = fitting.LinearLSQFitter()
    fit_result_cont = lin_fitter(
        pol_mod,
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
    )

    # plot the data and the continuum fit
    fig, axes = plt.subplots(
        2,
        sharex=True,
        figsize=(9, 10),
        gridspec_kw={"hspace": 0},
    )
    fs = 18
    axes[0].plot(
        waves[rangemask], fluxes[rangemask] * waves[rangemask] ** 2, c="k", alpha=0.9
    )
    axes[0].plot(
        waves[rangemask],
        fit_result_cont(waves[rangemask]),
        c="tab:orange",
        label="cont. fit",
    )

    # plot the data points that were used in the continuum fitting
    axes[0].plot(
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
        "r.",
        markersize=8,
        alpha=0.8,
        label="fit points",
    )
    axes[0].set_ylabel("flux (Jy)", fontsize=fs)

    # normalize the fluxes
    norm_fluxes = (
        fluxes[rangemask] * waves[rangemask] ** 2 / fit_result_cont(waves[rangemask])
    )

    # convert to optical depth (only exists in rangemask)
    taus = np.log(1 / norm_fluxes)

    # calculate the empirical uncertainy on the optical depth in clean regions
    unc = np.std(taus[cont_mask[rangemask]])

    # mask the stellar lines inside the feature and define the mask for the feature fitting
    stellar_lines_in = ((waves > 5.703) & (waves <= 5.717)) | (
        (waves > 5.823) & (waves <= 5.841)
    )
    # stellar line masks per star:
    # 5.705-5.717, 5.706-5.715, 5.703-716, 5.712-5.717, 5.711-5.717
    # 5.823-5.831, 5.824-5.836, 5.827-5.839, 5.829-5.841, 5.825-5.836, 5.823-5.840
    feat_full_mask = feat_reg_mask & ~stellar_lines_in
    feat_fit_mask = feat_full_mask[rangemask] & ~np.isnan(taus)

    # define the uncertainties
    uncs = np.full(len(taus[feat_fit_mask]), unc)

    # define the initial model to fit the feature
    gauss_mod = models.Gaussian1D(
        amplitude=0.01,
        stddev=0.02,
        mean=5.81,
        bounds={
            "mean": (5.78, 5.84),
            "stddev": (1e-4, 0.15),
        },
    )
    # fit the feature
    fit_result_feat_emcee, chains = fit_feature(
        datapath + "MIRI/",
        star,
        gauss_mod,
        waves[rangemask][feat_fit_mask],
        taus[feat_fit_mask],
        uncs,
        "58",
    )

    # plot the feature fits
    axes[1].plot(waves[rangemask], taus, c="k", alpha=0.9)
    axes[1].plot(
        waves[rangemask], fit_result_feat_emcee(waves[rangemask]), c="crimson", lw=2
    )
    axes[1].set_xlabel(r"wavelength ($\mu$m)", fontsize=fs)
    axes[1].set_ylabel("optical depth", fontsize=fs)
    axes[1].axhline(ls=":", c="k")
    plt.savefig(datapath + star + "_58_fit.pdf", bbox_inches="tight")

    return fit_result_feat_emcee, chains


def fit_62(datapath, star):
    """
    Function to fit the continuum and the feature at 6.2 micron

    Parameters
    ----------
    datapath : string
        Path to the data files

    star : string
        Star name

    Returns
    -------
    fit_result_feat_emcee : astropy model
        MCMC fitting results

    chains : numpy.ndarray
        MCMC chains (for amplitude, mean and stddev in that order)

    Saves plots with data and fitted models
    """
    # obtain the data
    starname = "hd" + star.split("D")[1].strip("0")
    data = Table.read(datapath + "MIRI/v3/" + starname + "_nircam_mrs_merged.fits")
    waves = data["wavelength"]
    fluxes = data["flux"]

    # define masks for the continuum fitting
    rangemask = (waves > 5.9) & (waves <= 6.6)
    feat_reg_mask = (waves > 6.1) & (waves <= 6.44)
    stellar_lines_out = ((waves > 5.887) & (waves <= 5.926)) | (
        (waves > 5.947) & (waves <= 5.969)
    )
    # stellar line masks per star:
    # 5.89-5.913, 5.903-5.913, 5.887-5.922, 5.894-5.918, 5.898-5.926, 5.894-5.911
    # 5.948-5.957, 5.951-5.969, 5.947-5.964
    cont_mask = rangemask & ~feat_reg_mask & ~stellar_lines_out & ~np.isnan(fluxes)

    # fit the continuum
    pol_mod = models.Polynomial1D(3)
    lin_fitter = fitting.LinearLSQFitter()
    fit_result_cont = lin_fitter(
        pol_mod,
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
    )

    # plot the data and the continuum fit
    fig, axes = plt.subplots(
        2,
        sharex=True,
        figsize=(9, 10),
        gridspec_kw={"hspace": 0},
    )
    fs = 18
    axes[0].plot(
        waves[rangemask], fluxes[rangemask] * waves[rangemask] ** 2, c="k", alpha=0.9
    )
    axes[0].plot(
        waves[rangemask],
        fit_result_cont(waves[rangemask]),
        c="tab:orange",
        label="cont. fit",
    )

    # plot the data points that were used in the continuum fitting
    axes[0].plot(
        waves[cont_mask],
        fluxes[cont_mask] * waves[cont_mask] ** 2,
        "r.",
        markersize=8,
        alpha=0.8,
        label="fit points",
    )
    axes[0].set_ylabel("flux (Jy)", fontsize=fs)

    # normalize the fluxes
    norm_fluxes = (
        fluxes[rangemask] * waves[rangemask] ** 2 / fit_result_cont(waves[rangemask])
    )

    # convert to optical depth (only exists in rangemask)
    taus = np.log(1 / norm_fluxes)

    # calculate the empirical uncertainy on the optical depth in clean regions
    unc = np.std(taus[cont_mask[rangemask]])

    # mask the stellar lines inside the feature and define the mask for the feature fitting
    stellar_lines_in = ((waves > 5.703) & (waves <= 5.717)) | (
        (waves > 5.823) & (waves <= 5.841)
    )
    # stellar line masks per star:
    # 5.705-5.717, 5.706-5.715, 5.703-716, 5.712-5.717, 5.711-5.717
    # 5.823-5.831, 5.824-5.836, 5.827-5.839, 5.829-5.841, 5.825-5.836, 5.823-5.840
    feat_full_mask = feat_reg_mask & ~stellar_lines_in
    feat_fit_mask = feat_full_mask[rangemask] & ~np.isnan(taus)

    # define the uncertainties
    uncs = np.full(len(taus[feat_fit_mask]), unc)

    # define the initial model to fit the feature
    gauss_mod = models.Gaussian1D(
        amplitude=0.01,
        stddev=0.02,
        mean=6.2,
        bounds={
            "mean": (6.1, 6.3),
            "stddev": (1e-4, 0.15),
        },
    )

    # fit the feature
    fit_result_feat_emcee, chains = fit_feature(
        datapath + "MIRI/",
        star,
        gauss_mod,
        waves[rangemask][feat_fit_mask],
        taus[feat_fit_mask],
        uncs,
        "62",
    )

    # plot the feature fits
    axes[1].plot(waves[rangemask], taus, c="k", alpha=0.9)
    axes[1].plot(
        waves[rangemask], fit_result_feat_emcee(waves[rangemask]), c="crimson", lw=2
    )
    axes[1].set_xlabel(r"wavelength ($\mu$m)", fontsize=fs)
    axes[1].set_ylabel("optical depth", fontsize=fs)
    axes[1].axhline(ls=":", c="k")
    plt.savefig(datapath + star + "_62_fit.pdf", bbox_inches="tight")

    return fit_result_feat_emcee, chains


def fit_10(datapath, star, profile):
    """
    Function to fit the continuum and the feature at 10 micron

    Parameters
    ----------
    datapath : string
        Path to the data files

    star : string
        Star name

    profile : string
        Profile to fit to the feature

    Returns
    -------
    waves : numpy.ndarray
        Relevant wavelengths

    taus : numpy.ndarray
        Optical depths at wavelengths in waves

    fit_result_feat_emcee : astropy model
        MCMC fitting results

    chains : numpy.ndarray
        MCMC chains (for amplitude, mean and stddev in that order)

    chi2 : float
        Chi square of fitting

    Saves plots with data and fitted models
    """
    # obtain the data
    starname = "hd" + star.split("D")[1].strip("0")
    data = Table.read(datapath + "MIRI/v3/" + starname + "_nircam_mrs_merged.fits")
    waves = data["wavelength"]

    # mask out the stellar lines
    stellar_mask = (
        ((waves > 6.924) & (waves <= 6.967))
        | ((waves > 7.410) & (waves <= 7.559))
        | ((waves > 8.732) & (waves <= 8.775))
        | ((waves > 9.381) & (waves <= 9.403))
        | ((waves > 9.665) & (waves <= 9.741))
        | ((waves > 10.446) & (waves <= 10.575))
        | ((waves > 11.251) & (waves <= 11.387))
        | ((waves > 12.251) & (waves <= 12.453))
        | ((waves > 12.570) & (waves <= 12.618))
        | ((waves > 13.086) & (waves <= 13.189))
    )
    # stellar line masks per star:
    # 7.435-7.522, 7.449-7.471, 7.443-7.538, 7.434-7.520, 7.454-7.467, 7.424-7.522, 7.428-7.473, 7.453-7.462
    # 7.497-7.512, 7.494-7.512
    # 12.570-12.618
    # 8.741-8.772, 8.732-8.775, 8.749-8.766
    # 9.381-9.397, 9.381-9.403
    # 10.447-10.575, 10.488-10.505
    # 11.285-11.321, 11.266-11.314, 11.251-11.373, 11.296-11.314
    # 12.352-12.424, 12.355-12.384, 12.312-12.429, 12.357-12.415, 12.312-12.445, 12.354-12.402, 12.355-12.381

    # mask the suspicious data between 11 and 12 micron
    bad_mask = (waves > 11.15) & (waves <= 12.1)
    # mask per star
    # 11.541-11.947, 11.462-12.013, 11.339-12.080, 11.355-12.096

    # rebin the spectrum, and select the relevant region
    waves, fluxes, uncs = rebin_constres(
        data[(~stellar_mask) & (~bad_mask)], (7.8, 12.9), 500
    )

    # define masks for the continuum fitting
    feat_reg_mask = (waves > 8.3) & (waves <= 12.3)
    # feature region per star:
    # 8.46-12.3, 8.1-12.19, 8.34-12.18, 8.47-12.1, 8.13-12.21, 8.08-12.11, 8.15-12.22, 8.20-12.33
    # 8.12-11.96, 8.07-12.19, 8.35-12.27, 8.42-12.02, 8.19-12.10, 8.15-12.10, 8.15-12.15, 8.31-12.29
    cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)

    # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
    taus, unc, axes = fit_cont(waves, fluxes, cont_mask)

    # define the mask for the feature fitting
    feat_fit_mask = feat_reg_mask & ~np.isnan(taus)

    # define the uncertainties
    emp_uncs = np.full(len(taus[feat_fit_mask]), unc)

    # define the initial model to fit the feature
    if profile == "gauss":
        feat_mod = Gaussian1D(
            amplitude=0.1,
            mean=10,
            stddev=1,
            bounds={
                "mean": (9.5, 10.5),
            },
        )
    elif profile == "gauss_skew":
        gauss_skew_mod = custom_model(gauss_skew_func)
        feat_mod = gauss_skew_mod(
            amplitude=0.1,
            loc=9,
            scale=1,
            alpha=2,
            bounds={
                "loc": (6, 10),
            },
        )
    elif profile == "drude":
        feat_mod = Drude1D(
            amplitude=0.1,
            x_0=10,
            fwhm=2,
            bounds={
                "x_0": (9.5, 10.5),
            },
        )
    elif profile == "drude_modif":
        Drude_mod = custom_model(drude_modified)
        feat_mod = Drude_mod(
            scale=0.1,
            x_o=10,
            gamma_o=2,
            bounds={
                "x_o": (9.5, 10.5),
            },
        )
    else:
        print(
            "This profile is not valid. Please choose between gauss, gauss_skew, drude or drude_modif."
        )

    # fit and plot the feature
    fit_result_feat_emcee, chains, chi2 = fit_feature(
        datapath + "MIRI/",
        datapath,
        star,
        feat_mod,
        waves[feat_fit_mask],
        taus[feat_fit_mask],
        emp_uncs,
        axes,
        waves,
        "10",
    )

    return waves, taus, fit_result_feat_emcee, chains, chi2


def fit_all(datapath, stars, sort_idx):
    """
    Fit all features for all stars

    Parameters
    ----------
    datapath : string
        Path to the data files

    stars : list
        Star names

    sort_idx : list
        Index for plotting order

    Returns
    -------
    Saves tables with fitting results
    """
    # list the features to be fitted
    # features = ["34", "58", "62", "10"]
    features = ["10"]
    profiles = ["gauss_skew"]

    for feat_name, profile in zip(features, profiles):
        print("Fitting " + feat_name + " micron feature")

        # create a table to store the results
        if profile == "gauss":
            names = (
                "name",
                "amplitude",
                "amplitude_unc_min",
                "amplitude_unc_plus",
                "wavelength(micron)",
                "wavelength_unc_min(micron)",
                "wavelength_unc_plus(micron)",
                "std(micron)",
                "std_unc_min(micron)",
                "std_unc_plus(micron)",
                "area(cm-1)",
                "area_unc_min(cm-1)",
                "area_unc_plus(cm-1)",
                "chi2",
            )
        elif profile == "gauss_skew":
            names = (
                "name",
                "amplitude",
                "amplitude_unc_min",
                "amplitude_unc_plus",
                "location(micron)",
                "location_unc_min(micron)",
                "location_unc_plus(micron)",
                "scale(micron)",
                "scale_unc_min(micron)",
                "scale_unc_plus(micron)",
                "alpha",
                "alpha_unc_min",
                "alpha_unc_plus",
                "x_0(micron)",
                "x_0_unc_min(micron)",
                "x_0_unc_plus(micron)",
                # "area(cm-1)",
                # "area_unc_min(cm-1)",
                # "area_unc_plus(cm-1)",
                "chi2",
            )
        dtypes = np.full(len(names), "float64")
        dtypes[0] = "str"
        table_txt = Table(names=names, dtype=dtypes)

        table_lat = Table(
            names=(
                "star",
                r"$\tau$($\lambda_0$)",
                r"$\lambda_0$(\micron)",
                r"$\sigma$(\micron)",
                #  r"$A$(cm$^{-1}$)",
                "gamma",
            ),
            dtype=(
                "str",
                "str",
                "str",
                "str",
                "str",
            ),  # "str"),
        )

        # create a figure to plot the feature
        fig, ax = plt.subplots(figsize=(8, 14))

        # fit the feature for all stars
        for i, star in enumerate(stars):
            print(star)
            func = eval("fit_" + feat_name)
            waves, taus, fit_result, chains, chi2 = func(datapath, star, profile)
            print(fit_result, chi2)

            # obtain the results
            result_list = []
            lat_list = []
            for param_name in fit_result.param_names:
                param = getattr(fit_result, param_name)
                result_list.extend([param.value, param.unc_minus, param.unc_plus])
                lat_list.append(
                    "{:.2f}".format(param.value)
                    + "_{-"
                    + "{:.2f}".format(param.unc_minus)
                    + "}^{+"
                    + "{:.2f}".format(param.unc_plus)
                    + "}"
                )

            if profile == "gauss":
                # convert the standard deviation from units of wavelengths (micron) to wavenumbers (cm^-1)
                # dnu = 1e4 dlambda / lambda^2
                # lambda = mean = chains[:,1]
                # dlambda = stddev = chains[:,2]
                stddev_wavenm = 1e4 * chains[:, 2] / (chains[:, 1] ** 2)

                # calculate the integrated area (in cm^-1)
                # area = amplitude * stddev (in cm^-1)* sqrt(2 * pi)
                # amplitude = chains[:,0]
                area_chain = chains[:, 0] * stddev_wavenm * np.sqrt(2 * np.pi)
                area16, area, area84 = np.percentile(area_chain, [16, 50, 84])
                area_unc_min = area - area16
                area_unc_plus = area84 - area
                result_list.extend(
                    [
                        area,
                        area_unc_min,
                        area_unc_plus,
                    ]
                )
                lat_list.append(
                    "{:.2f}".format(area)
                    + "_{-"
                    + "{:.2f}".format(area_unc_min)
                    + "}^{+"
                    + "{:.2f}".format(area_unc_plus)
                    + "}"
                )

            elif profile == "gauss_skew":
                amplitudes = chains[:, 0]
                locs = chains[:, 1]
                scales = chains[:, 2]
                alphas = chains[:, 3]
                # calculate the mode (i.e. peak wavelength) (in micron)
                # delta = alpha / sqrt(1+alpha^2)
                # m0 = sqrt(2/pi) * delta - (1 - pi/4) * ((sqrt(2/pi)*delta)^3 / (1-2/pi*delta^2)) - sign(alpha)/2 * exp(-2pi/|alpha|))
                # mode = location + scale * m0
                deltas = alphas / (np.sqrt(1 + alphas ** 2))
                m0s = (
                    np.sqrt(2 / np.pi) * deltas
                    - (1 - np.pi / 4)
                    * (np.sqrt(2 / np.pi) * deltas) ** 3
                    / (1 - 2 / np.pi * deltas ** 2)
                    - np.sign(alphas) / 2 * np.exp(-2 * np.pi / np.absolute(alphas))
                )
                mode_chain = locs + scales * m0s
                mode16, mode, mode84 = np.percentile(mode_chain, [16, 50, 84])
                mode_unc_min = mode - mode16
                mode_unc_plus = mode84 - mode
                result_list.extend(
                    [
                        mode,
                        mode_unc_min,
                        mode_unc_plus,
                    ]
                )

            # add the results to the tables
            table_txt.add_row((star, *result_list, chi2))
            table_lat.add_row((star, *lat_list))

            # plot the feature and fitted profile
            ax.plot(waves, taus + sort_idx[i] * 0.07, c="k", alpha=0.9)
            ax.plot(waves, fit_result(waves) + sort_idx[i] * 0.07, c="crimson", lw=2)
            ax.annotate(
                star,
                (waves[0] - 0.15, fit_result(waves[0]) + sort_idx[i] * 0.07 + 0.015),
                fontsize=16,
            )

        # finalize and save the figure
        fs = 20
        ax.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=fs)
        ax.set_ylabel(r"$\tau$($\lambda$) + offset", fontsize=fs)
        ax.set_ylim(-0.05, None)
        outname = datapath + feat_name + "_all.pdf"
        # rename the previous version of the plot
        if os.path.isfile(outname):
            os.rename(outname, outname.split(".")[0] + "_0.pdf")
        fig.savefig(outname, bbox_inches="tight")

        # write the tables to files
        table_txt.write(
            datapath + "fit_results_" + feat_name + ".txt",
            format="ascii",
            overwrite=True,
        )

        table_lat.write(
            datapath + "fit_results_" + feat_name + ".tex",
            format="aastex",
            col_align="l|CCCC",
            latexdict={
                "tabletype": "deluxetable*",
                "caption": r"MCMC fitting results for the 10 \micron\ silicate feature. \label{tab:fit_results}",
            },
            overwrite=True,
        )


def main():
    # plotting settings for uniform plots
    fs = 18
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)

    # define the data path
    datapath = "/Users/mdecleir/Documents/MEAD/extinction/JWST_data/"

    # list the stars
    stars = [
        # "HD014434",
        "HD038087",
        "HD073882",
        "HD147888",
        "HD152249",
        "HD203938",
        "HD206267",
        "HD207198",
        "HD216898",
    ]

    # sort the stars by silicate feature strength by giving them an index. 0=weakest feature.
    sort_idx = [  # "HD014434",
        3,
        7,
        5,
        0,
        4,
        1,
        2,
        6,
    ]

    # obtain and plot a stellar model
    stel_mod = Table.read(datapath + "MIRI/tlusty_z100t25000g400v2_miri_ifu.fits")
    stel_mod = Table.read(datapath + "MIRI/tlusty_z100t23000g400v2_nircam_ss.fits")
    plt.plot(
        stel_mod["WAVELENGTH"] * 1e-4,
        stel_mod["FLUX"] * (stel_mod["WAVELENGTH"] * 1e-4) ** 4,
    )
    # plt.xlim(5.6, 6.1)
    # plt.xlim(7, 14)
    # plt.ylim(3.2e6, 3.4e6)
    # plt.show()

    # fit and plot all features for all stars
    fit_all(datapath, stars, sort_idx)


if __name__ == "__main__":
    main()
