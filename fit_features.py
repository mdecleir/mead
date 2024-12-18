# fit_features.py: script to fit the features in the JWST MEAD spectra

# import the necessary packages
import astropy.units as u
import numpy as np
import os
import scipy

from astropy.modeling.fitting import (
    FittingWithOutlierRemoval,
    LevMarLSQFitter,
    LinearLSQFitter,
)
from astropy.modeling.models import (
    custom_model,
    Drude1D,
    Gaussian1D,
    Polynomial1D,
    Linear1D,
)
from astropy.stats import sigma_clip
from astropy.table import Table

from dust_extinction.grain_models import WD01, D03, ZDA04, J13, HD23, Y24
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


def gauss_skew_func(x, amplitude=0.1, loc=9, scale=1, shape=2):
    return amplitude * stats.skewnorm.pdf(x, shape, loc=loc, scale=scale)


def multi_gauss_model(amplitudes, means, stddevs):
    """
    Function to create an Astropy model with the sum of multiple Gaussians

    Parameters
    ----------
    amplitudes : np.ndarray
        Amplitudes of the Gaussians

    means : np.ndarray
        Central wavelengths of the Gaussians

    stddev : np.ndarray
        Standard deviations of the Gaussians

    Returns
    -------
    Astropy CompoundModel with the sum of the Gaussians
    """
    # create the first Gaussian
    model_sum = Gaussian1D(
        amplitude=amplitudes[0],
        mean=means[0],
        stddev=stddevs[0],
        fixed={"mean": True, "stddev": True},
    )
    # add the rest of the Gaussians
    for i in range(1, len(amplitudes)):
        model_sum += Gaussian1D(
            amplitude=amplitudes[i],
            mean=means[i],
            stddev=stddevs[i],
            fixed={"mean": True, "stddev": True},
        )

    return model_sum


def fwhm_to_stddev(fwhm):
    """
    Function to convert the FWHM of a Gaussian to the standard deviation
    Gaussian1D model stddev: FWHM = 2 * stddev * sqrt(2 * ln(2))
    stddev = FWHM / 2 / sqrt(2 * ln(2))

    Parameters
    ----------
    fwhm : float
        FWHM of the Gaussian

    Returns
    -------
    stddev : float
        standard deviation of the Gaussian
    """
    return fwhm / 2 / np.sqrt(2 * np.log(2))


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


def fit_cont(waves, fluxes, cont_mask, bad_mask=None, mod=False):
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

    bad_mask : numpy.ndarray [default=None]
        Boolean mask for region to be plotted in gray

    Returns
    -------
    taus : astropy Table Column
        Optical depths (in range_mask)

    unc : numpy.float64
        Empirical uncertainty on the optical depth

    axes : numpy.ndarray
        Plotting axes of the figure

    mod : boolean [default=False]
        Whether or not this is a model
    """
    # if this is data, flatten the Rayleigh Jeans tail
    if mod == False:
        fluxes = fluxes * waves ** 2

    # fit the continuum
    lin_mod = Linear1D()
    lin_fitter = LinearLSQFitter()
    out_fitter = FittingWithOutlierRemoval(lin_fitter, sigma_clip, niter=3, sigma=3)

    fit_result_cont, clipmask = out_fitter(
        lin_mod,
        waves[cont_mask],
        fluxes[cont_mask],
    )

    # plot the data
    fig, axes = plt.subplots(
        2,
        sharex=True,
        figsize=(9, 10),
        gridspec_kw={"hspace": 0},
    )
    axes[0].plot(waves, fluxes, c="k", alpha=0.9)

    # plot the data points that were used in the continuum fitting
    axes[0].plot(
        waves[cont_mask],
        fluxes[cont_mask],
        "r.",
        markersize=8,
        alpha=0.8,
        label="fit points",
    )

    # cross the sigma clipped data points that were not used in the fitting
    axes[0].plot(
        waves[cont_mask][clipmask],
        fluxes[cont_mask][clipmask],
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
    norm_fluxes = fluxes / fit_result_cont(waves)

    # convert to optical depth (only exists in range_mask)
    taus = np.log(1 / norm_fluxes)

    # plot the optical depth
    if bad_mask is None:
        bad_mask = np.full(len(taus), False, dtype=bool)
    plot_taus = np.copy(taus)
    plot_taus[bad_mask] = np.nan
    axes[1].plot(waves, plot_taus, c="k", alpha=0.9)

    # plot the extra feature in grey
    axes[1].plot(waves[bad_mask], taus[bad_mask], c="k", alpha=0.3)

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
    nsteps = 10000
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

    # rebin the spectrum, and select the relevant region
    waves, fluxes, uncs = rebin_constres(data[~stellar_mask], (7.9, 12.8), 400)

    # define masks for the continuum fitting
    # mask the extra feature between 11 and 12 micron
    bad_mask = (waves > 11.1) & (waves <= 12.1)
    # mask per star
    # 11.541-11.947, 11.462-12.013, 11.339-12.080, 11.355-12.096
    feat_reg_mask = (waves > 8.1) & (waves <= 12.6)
    # feature region per star:
    # 8.46-12.3, 8.1-12.19, 8.34-12.18, 8.47-12.1, 8.13-12.21, 8.08-12.11, 8.15-12.22, 8.20-12.33
    # 8.12-11.96, 8.07-12.19, 8.35-12.27, 8.42-12.02, 8.19-12.10, 8.15-12.10, 8.15-12.15, 8.31-12.29
    cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)

    # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
    taus, unc, axes = fit_cont(waves, fluxes, cont_mask, bad_mask)

    # define the mask for the feature fitting
    feat_fit_mask = feat_reg_mask & ~np.isnan(taus) & ~bad_mask

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
            shape=2,
            bounds={"loc": (8, 10), "scale": (0.5, 2), "shape": (0, 4)},
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

    return waves, taus, fit_result_feat_emcee, chains, chi2, bad_mask


def fit_grain_mod(datapath):
    """
    Function to fit the continuum and the feature at 10 micron in dust grain models

    Parameters
    ----------
    datapath : string
        Path to store the output plots and tables

    Returns
    -------
    Saves plots with dust grain models and fitted models
    """
    # define the dust grain models to be fitted
    grmods = [WD01, D03, ZDA04, J13, HD23, Y24]
    mod_names = ["MWRV31", "MWRV31", "BARE-GR-S", "MWRV31", "MWRV31", "MWRV31"]

    # define the wavelength grid
    waves = _wavegrid((7.9, 12.8), 400)[0]

    # mask out the stellar lines
    stellar_mask = (
        ((waves > 8.732) & (waves <= 8.775))
        | ((waves > 9.381) & (waves <= 9.403))
        | ((waves > 9.665) & (waves <= 9.741))
        | ((waves > 10.446) & (waves <= 10.575))
        | ((waves > 11.251) & (waves <= 11.387))
        | ((waves > 12.251) & (waves <= 12.453))
        | ((waves > 12.570) & (waves <= 12.618))
    )

    # mask the suspicious data between 11 and 12 micron
    bad_mask = (waves > 11.1) & (waves <= 12.1)

    # define the feature mask
    feat_reg_mask = (waves > 8.1) & (waves <= 12.6)

    # define the initial model to fit the feature
    gauss_skew_mod = custom_model(gauss_skew_func)
    feat_mod = gauss_skew_mod(
        amplitude=0.1,
        loc=9,
        scale=1,
        shape=2,
    )

    # create a table to store the results
    names = (
        "name",
        "tau",
        "tau_unc_min",
        "tau_unc_plus",
        "area(micron)",
        "area_unc_min(micron)",
        "area_unc_plus(micron)",
    )
    dtypes = np.full(len(names), "float64")
    dtypes[0] = "str"
    table_txt = Table(names=names, dtype=dtypes)

    for grmod, mod_name in zip(grmods, mod_names):
        # obtain the model extinction
        ext_model = grmod(mod_name)

        # convert the extinction curve to an extinguished spectrum, assuming a flat intrinsic spectrum (=1)
        fluxes = ext_model.extinguish(waves * u.micron, Av=1)
        fluxes[(stellar_mask) | (bad_mask)] = np.nan

        # define masks for the continuum fitting
        cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)

        # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
        taus, unc, axes = fit_cont(waves, fluxes, cont_mask, bad_mask, mod=True)

        # define the mask for the feature fitting
        feat_fit_mask = feat_reg_mask & ~np.isnan(taus)

        # define the uncertainties
        emp_uncs = np.full(len(taus[feat_fit_mask]), unc)

        # fit and plot the feature
        fit_result_feat_emcee, chains, chi2 = fit_feature(
            datapath + "MIRI/",
            datapath,
            f"{ext_model.__class__.__name__}",
            feat_mod,
            waves[feat_fit_mask],
            taus[feat_fit_mask],
            emp_uncs,
            axes,
            waves,
            "10",
        )

        amplitudes = chains[:, 0]
        locs = chains[:, 1]
        scales = chains[:, 2]
        shapes = chains[:, 3]
        # calculate the mode (i.e. peak wavelength) (in micron)
        # delta = shape / sqrt(1+shape^2)
        # m0 = sqrt(2/pi) * delta - (1 - pi/4) * (sqrt(2/pi)*delta)^3 / (1-2/pi*delta^2) - sgn(shape)/2 * exp(-2pi/|shape|)
        # mode = location + scale * m0
        deltas = shapes / np.sqrt(1 + shapes ** 2)
        m0s = (
            np.sqrt(2 / np.pi) * deltas
            - (1 - np.pi / 4)
            * (np.sqrt(2 / np.pi) * deltas) ** 3
            / (1 - 2 / np.pi * deltas ** 2)
            - np.sign(shapes) / 2 * np.exp(-2 * np.pi / np.absolute(shapes))
        )
        mode_chain = locs + scales * m0s
        mode16, mode, mode84 = np.percentile(mode_chain, [16, 50, 84])
        mode_unc_min = mode - mode16
        mode_unc_plus = mode84 - mode

        # calculate the maximum optical depth (i.e. at the peak wavelength)
        tau_chain = gauss_skew_func(mode_chain, amplitudes, locs, scales, shapes)
        tau16, tau, tau84 = np.percentile(tau_chain, [16, 50, 84])
        tau_unc_min = tau - tau16
        tau_unc_plus = tau84 - tau

        # calculate the area (is equal to the amplitude, given that the pdf of the scipy skewnorm is normalized to 1)
        area16, area, area84 = np.percentile(chains[:, 0], [16, 50, 84])
        area_unc_min = area - area16
        area_unc_plus = area84 - area

        # add the results to the table
        result_list = [
            tau,
            tau_unc_min,
            tau_unc_plus,
            area,
            area_unc_min,
            area_unc_plus,
        ]
        table_txt.add_row((f"{ext_model.__class__.__name__}", *result_list))

    # write the table to a file
    table_txt.write(
        datapath + "fit_results_mod.txt",
        format="ascii",
        overwrite=True,
    )


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
            names_txt = (
                "name",
                "tau",
                "tau_unc_min",
                "tau_unc_plus",
                "x_0(micron)",
                "x_0_unc_min(micron)",
                "x_0_unc_plus(micron)",
                "std(micron)",
                "std_unc_min(micron)",
                "std_unc_plus(micron)",
                "FWHM(micron)",
                "FWHM_unc_min(micron)",
                "FWHM_unc_plus(micron)",
                "area(cm-1)",
                "area_unc_min(cm-1)",
                "area_unc_plus(cm-1)",
                "chi2",
            )
            names_tex = (
                "star",
                r"$\lambda_0$(\micron)",
                r"$\tau$($\lambda_0$)",
                r"FWHM(\micron)",
                r"$A$(cm$^{-1}$)",
            )
        elif profile == "gauss_skew":
            names_txt = (
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
                "shape",
                "shape_unc_min",
                "shape_unc_plus",
                "x_0(micron)",
                "x_0_unc_min(micron)",
                "x_0_unc_plus(micron)",
                "tau",
                "tau_unc_min",
                "tau_unc_plus",
                "FWHM(micron)",
                "FWHM_unc_min(micron)",
                "FWHM_unc_plus(micron)",
                "area(micron)",
                "area_unc_min(micron)",
                "area_unc_plus(micron)",
                "chi2",
            )
            names_tex = (
                "star",
                r"$\lambda_0$(\micron)",
                r"$\tau$($\lambda_0$)",
                r"FWHM(\micron)",
                r"area(\micron) (=$B$)",
                r"$\alpha$",
            )
        dtypes = np.full(len(names_txt), "float64")
        dtypes[0] = "str"
        table_txt = Table(names=names_txt, dtype=dtypes)
        table_tex = Table(names=names_tex, dtype=np.full(len(names_tex), "str"))

        # create a figure to plot the feature
        fig, ax = plt.subplots(figsize=(8, 2 * len(stars)))

        # fit the feature for all stars
        for i, star in enumerate(stars):
            print(star)
            func = eval("fit_" + feat_name)
            waves, taus, fit_result, chains, chi2, bad_mask = func(
                datapath, star, profile
            )
            print(fit_result, chi2)

            # obtain the results
            result_list = []
            for param_name in fit_result.param_names:
                param = getattr(fit_result, param_name)
                result_list.extend([param.value, param.unc_minus, param.unc_plus])

            if profile == "gauss":
                # calculate the FWHM (in micron)
                fwhm_chain = 2 * np.sqrt(2 * np.log(2)) * chains[:, 2]
                fwhm16, fwhm, fwhm84 = np.percentile(fwhm_chain, [16, 50, 84])
                fwhm_unc_min = fwhm - fwhm16
                fwhm_unc_plus = fwhm84 - fwhm

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
                        fwhm,
                        fwhm_unc_min,
                        fwhm_unc_plus,
                        area,
                        area_unc_min,
                        area_unc_plus,
                    ]
                )

                tex_list = [
                    "{:.2f}".format(fit_result.mean.value)
                    + "_{-"
                    + "{:.2f}".format(fit_result.mean.unc_minus)
                    + "}^{+"
                    + "{:.2f}".format(fit_result.mean.unc_plus)
                    + "}",
                    "{:.3f}".format(fit_result.amplitude.value)
                    + "_{-"
                    + "{:.3f}".format(fit_result.amplitude.unc_minus)
                    + "}^{+"
                    + "{:.3f}".format(fit_result.amplitude.unc_plus)
                    + "}",
                    "{:.2f}".format(fwhm)
                    + "_{-"
                    + "{:.2f}".format(fwhm_unc_min)
                    + "}^{+"
                    + "{:.2f}".format(fwhm_unc_plus)
                    + "}",
                    "{:.2f}".format(area)
                    + "_{-"
                    + "{:.2f}".format(area_unc_min)
                    + "}^{+"
                    + "{:.2f}".format(area_unc_plus)
                    + "}",
                ]

            elif profile == "gauss_skew":
                amplitudes = chains[:, 0]
                locs = chains[:, 1]
                scales = chains[:, 2]
                shapes = chains[:, 3]
                # calculate the mode (i.e. peak wavelength) (in micron)
                # delta = shape / sqrt(1+shape^2)
                # m0 = sqrt(2/pi) * delta - (1 - pi/4) * (sqrt(2/pi)*delta)^3 / (1-2/pi*delta^2) - sgn(shape)/2 * exp(-2pi/|shape|)
                # mode = location + scale * m0
                deltas = shapes / np.sqrt(1 + shapes ** 2)
                m0s = (
                    np.sqrt(2 / np.pi) * deltas
                    - (1 - np.pi / 4)
                    * (np.sqrt(2 / np.pi) * deltas) ** 3
                    / (1 - 2 / np.pi * deltas ** 2)
                    - np.sign(shapes) / 2 * np.exp(-2 * np.pi / np.absolute(shapes))
                )
                mode_chain = locs + scales * m0s
                mode16, mode, mode84 = np.percentile(mode_chain, [16, 50, 84])
                mode_unc_min = mode - mode16
                mode_unc_plus = mode84 - mode

                # calculate the maximum optical depth (i.e. at the peak wavelength)
                tau_chain = gauss_skew_func(
                    mode_chain, amplitudes, locs, scales, shapes
                )
                tau16, tau, tau84 = np.percentile(tau_chain, [16, 50, 84])
                tau_unc_min = tau - tau16
                tau_unc_plus = tau84 - tau

                # calculate the FWHM at each chain point (this can also be done using the scipy.signal functions find_peaks and peak_widths, or using the scipy.interpolate splrep and sproot functions)
                fwhm_chain = np.zeros(len(mode_chain))
                xs = np.arange(waves[0], waves[-1], 0.001)

                for j, (tau, amplitude, loc, scale, shape) in enumerate(
                    zip(tau_chain, amplitudes, locs, scales, shapes)
                ):
                    ys = gauss_skew_func(xs, amplitude, loc, scale, shape)
                    peak_idx = np.argmax(ys)
                    left = np.interp(tau / 2, ys[:peak_idx], xs[:peak_idx])
                    right = np.interp(
                        tau / 2, np.flip(ys[peak_idx:]), np.flip(xs[peak_idx:])
                    )
                    fwhm_chain[j] = right - left

                # calculate the FWHM 50th percentile and uncertainties
                fwhm16, fwhm, fwhm84 = np.percentile(fwhm_chain, [16, 50, 84])
                fwhm_unc_min = fwhm - fwhm16
                fwhm_unc_plus = fwhm84 - fwhm

                # calculate the area (is equal to the amplitude, given that the pdf of the scipy skewnorm is normalized to 1)
                area16, area, area84 = np.percentile(amplitudes, [16, 50, 84])
                area_unc_min = area - area16
                area_unc_plus = area84 - area

                # calculate the shape 50th percentile and uncertainties
                shape16, shape, shape84 = np.percentile(shapes, [16, 50, 84])
                shape_unc_min = shape - shape16
                shape_unc_plus = shape84 - shape

                # add the results to the list
                result_list.extend(
                    [
                        mode,
                        mode_unc_min,
                        mode_unc_plus,
                        tau,
                        tau_unc_min,
                        tau_unc_plus,
                        fwhm,
                        fwhm_unc_min,
                        fwhm_unc_plus,
                        area,
                        area_unc_min,
                        area_unc_plus,
                    ]
                )

                tex_list = [
                    "{:.2f}".format(mode)
                    + "_{-"
                    + "{:.2f}".format(mode_unc_min)
                    + "}^{+"
                    + "{:.2f}".format(mode_unc_plus)
                    + "}",
                    "{:.3f}".format(tau)
                    + "_{-"
                    + "{:.3f}".format(tau_unc_min)
                    + "}^{+"
                    + "{:.3f}".format(tau_unc_plus)
                    + "}",
                    "{:.2f}".format(fwhm)
                    + "_{-"
                    + "{:.2f}".format(fwhm_unc_min)
                    + "}^{+"
                    + "{:.2f}".format(fwhm_unc_plus)
                    + "}",
                    "{:.3f}".format(area)
                    + "_{-"
                    + "{:.3f}".format(area_unc_min)
                    + "}^{+"
                    + "{:.3f}".format(area_unc_plus)
                    + "}",
                    "{:.2f}".format(shape)
                    + "_{-"
                    + "{:.2f}".format(shape_unc_min)
                    + "}^{+"
                    + "{:.2f}".format(shape_unc_plus)
                    + "}",
                ]

            # add the results to the tables
            table_txt.add_row((star, *result_list, chi2))
            table_tex.add_row((star, *tex_list))

            # plot the feature
            plot_taus = np.copy(taus)
            plot_taus[bad_mask] = np.nan
            handle1 = ax.plot(
                waves, plot_taus + sort_idx[i] * 0.07, c="k", alpha=0.9, label="data"
            )

            # plot the extra feature in grey
            ax.plot(
                waves[bad_mask], taus[bad_mask] + sort_idx[i] * 0.07, c="k", alpha=0.3
            )
            # plot the fitted profile
            handle2 = ax.plot(
                waves,
                fit_result(waves) + sort_idx[i] * 0.07,
                c="crimson",
                lw=2,
                label="fit",
            )

            # add the star name
            ax.annotate(
                star,
                (waves[0] - 0.38, fit_result(waves[0]) + sort_idx[i] * 0.07),
                fontsize=16,
                rotation=30,
            )

        # finalize and save the figure
        fs = 20
        ax.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=fs)
        ax.set_ylabel(r"$\tau$($\lambda$) + offset", fontsize=fs)
        ax.set_ylim(-0.045, 0.7)
        ax.set_xlim(7.4, 13)
        fig.legend(
            handles=[handle1[0], handle2[0]],
            bbox_to_anchor=(0.88, 0.88),
            fontsize=0.8 * fs,
        )

        outname = datapath + feat_name + "_all.pdf"
        # rename the previous version of the plot
        if os.path.isfile(outname):
            os.rename(outname, outname.split(".")[0] + "_0.pdf")
        fig.savefig(outname, bbox_inches="tight")

        # add the median values to the latex table
        table_tex.add_row(
            [
                "median",
                "{:.2f}".format(np.median(table_txt["x_0(micron)"])),
                "{:.3f}".format(np.median(table_txt["tau"])),
                "{:.2f}".format(np.median(table_txt["FWHM(micron)"])),
                "{:.3f}".format(np.median(table_txt["area(micron)"])),
                "{:.2f}".format(np.median(table_txt["shape"])),
            ],
        )

        # write the tables to files
        table_txt.write(
            datapath + "fit_results_" + feat_name + "_" + profile + ".txt",
            format="ascii",
            overwrite=True,
        )

        table_tex.write(
            datapath + "fit_results_" + feat_name + "_" + profile + ".tex",
            format="aastex",
            col_align="l|CCCCC",
            latexdict={
                "tabletype": "deluxetable*",
                "caption": r"Derived properties of the 10\,\micron\ silicate feature, based on the MCMC fitting results. Median values for this sample are listed in the bottom row. \label{tab:fit_results}",
            },
            overwrite=True,
        )


def stack_spectra_34(datapath, stars, ext_table):
    """
    Function to stack the optical depth around the 3.4 micron feature

    Parameters
    ----------
    datapath : string
        Path to the data files

    stars : list
        Star names

    ext_table : astropy Table
        Extinction properties

    Returns
    -------
    Plot with all optical depths, average optical depths and standard error of the mean optical depths
    """
    # create an empty list to store the optical depths
    tau_list = []

    # create the figure
    fs = 18
    fig, ax = plt.subplots(figsize=(6, 6))

    for star in stars:
        # obtain the data
        data = Table.read(
            datapath + "NIRCam/v4/" + star + "_F322W2_fullSED.dat",
            format="ascii",
            data_start=1,
        )
        data["col1"].name = "wavelength"
        data["col2"].name = "flux"
        data["col3"].name = "unc"
        waves = data["wavelength"]

        # mask out the stellar and HI lines
        stellar_mask = (waves > 3.589) & (waves <= 3.596)

        # stellar lines per star
        # 3.076-3.105; 3.081-3.096; 3.076-3.097; 3.079-3.097; 3.076-3.097
        # 3.590-3.595; 3.590-3.596
        # 3.674-3.684; 3.673-3.687; 3.673-3.686
        # 3.757-3.758?
        # "dips" per star
        # 3.032-3.044; 3.031-3.047; 3.031-3.043; 3.028-3.055; 3.032-3.045; 3.027-3.047; 3.032-3.044; 3.030-3.045; 3.029-3.047
        # 3.287-3.303; 3.280-3.307; 3.282-3.303; 3.282-3.308; 3.282-3.304; 3.289-3.304; 3.284-3.304; 3.282-3.303; 3.282-3.308
        # 3.697-3.709; 3.700-3.709; 3.700-3.709; 3.700-3.710; 3.698-3.709; 3.700-3.709
        # 3.728-3.753; 3.734-3.750; 3.725-3.757; 3.733-3.750; 3.734-3.748; 3.734-3.750

        # rebin the spectrum, and select the relevant region
        waves, fluxes, uncs = rebin_constres(data[~stellar_mask], (3.31, 3.64), 400)

        # define masks for the continuum fitting
        feat_reg_mask = (waves > 3.35) & (waves <= 3.59)

        # feature region per star:
        # 3.32-3.52; 3.39-3.47; 3.37-3.61; 3.34-3.63; 3.73-3.5; 3.35-3.52; 3.34-3.63; 3.32-3.63; 3.32-3.63
        cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)

        # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
        taus, unc, axes = fit_cont(waves, fluxes, cont_mask)

        # rename the previous version of the plot
        outname = datapath + star + "_34_cont.pdf"
        if os.path.isfile(outname):
            os.rename(outname, outname.split(".")[0] + "_0.pdf")
        plt.savefig(outname, bbox_inches="tight")

        # obtain A(V), and normalize the optical depths
        tab_mask = ext_table["Name"] == star
        tau_list.append(taus / ext_table["AV"][tab_mask])

        # plot the normalized optical depths
        ax.plot(waves, taus / ext_table["AV"][tab_mask], lw=1, alpha=0.6)

    # average the normalized optical depths
    ave_taus = np.mean(tau_list, axis=0)
    ax.plot(waves, ave_taus, c="k", lw=3, label="mean")

    # calculate the uncertainties on the mean optical depths
    # standard error of the mean
    ave_uncs = scipy.stats.sem(tau_list, axis=0)
    ax.plot(waves, ave_uncs, c="r", zorder=1, label="std. err. of mean")

    # plot the fitted Chiar+2013 model with 4 Gaussians
    chiar = Table.read(
        "/Users/mdecleir/Documents/MEAD/Literature_data/gcs3_gaussfit.dat",
        format="ascii",
    )
    ax.plot(
        chiar["micron"],
        (chiar["gauss2"] + chiar["gauss3"] + chiar["gauss4"] + chiar["gauss5"])
        / 29
        / 2.5,
        color="darkviolet",
        ls=":",
        lw=3,
        label="Quintuplet (C13) / 2.5",
    )

    # find the peak value and its uncertainty
    max_ind = np.argmax(ave_taus)
    print("Peak wavelength:", waves[max_ind])
    print("Peak tau/AV: ", ave_taus[max_ind], " +- ", ave_uncs[max_ind])
    print(
        "Peak AV/tau: ",
        1 / ave_taus[max_ind],
        1 / ave_taus[max_ind] - 1 / (ave_taus[max_ind] + ave_uncs[max_ind]),
        1 / (ave_taus[max_ind] - ave_uncs[max_ind]) - 1 / ave_taus[max_ind],
    )

    # finalize and save the figure
    ax.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=fs)
    ax.set_ylabel(r"$\tau(\lambda)/A(V)$", fontsize=fs)
    ax.axhline(c="k", ls=":", alpha=0.5)
    ax.set_xlim(3.3, 3.66)
    ax.set_ylim(-0.004, 0.008)
    # rename the previous version of the plot
    outname = datapath + "34_all.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")
    ax.legend(fontsize=0.8 * fs)
    fig.savefig(outname, bbox_inches="tight")


def stack_spectra_62(datapath, stars, ext_table, bad_stars):
    """
    Function to stack the optical depth around the 6.2 micron feature

    Parameters
    ----------
    datapath : string
        Path to the data files

    stars : list
        Star names

    ext_table : astropy Table
        Extinction properties

    bad_stars : list
        Stars that should not be used in the average

    Returns
    -------
    Plot with all optical depths, average optical depths and standard error of the mean optical depths
    """
    # create an empty list to store the optical depths
    tau_list = []

    # create the figure
    fs = 18
    fig, ax = plt.subplots(figsize=(6, 6))

    for star in stars:
        if star in bad_stars:
            continue

        # obtain the data
        starname = "hd" + star.split("D")[1].strip("0")
        data = Table.read(datapath + "MIRI/v3/" + starname + "_nircam_mrs_merged.fits")
        waves = data["wavelength"]

        # mask out the stellar and HI lines
        stellar_mask = (waves > 6.282) & (waves <= 6.299)

        # stellar lines
        # 6.018-6.032
        # 6.455-6.472

        # rebin the spectrum, and select the relevant region
        waves, fluxes, uncs = rebin_constres(data[~stellar_mask], (6.06, 6.45), 400)

        # define masks for the continuum fitting
        feat_reg_mask = (waves > 6.15) & (waves <= 6.36)

        # feature region per star:
        # 6.14-6.39; 6.15-6.29; 5.94-6.41; 5.97-6.42; 5.99-6.36
        cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)

        # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
        taus, unc, axes = fit_cont(waves, fluxes, cont_mask)

        # rename the previous version of the plot
        outname = datapath + star + "_62_cont.pdf"
        if os.path.isfile(outname):
            os.rename(outname, outname.split(".")[0] + "_0.pdf")
        plt.savefig(outname, bbox_inches="tight")

        # obtain A(V), and normalize the optical depths
        tab_mask = ext_table["Name"] == star
        tau_list.append(taus / ext_table["AV"][tab_mask])

        # plot the normalized optical depths
        ax.plot(waves, taus / ext_table["AV"][tab_mask], lw=1, alpha=0.6)

    # average the normalized optical depths
    ave_taus = np.mean(tau_list, axis=0)
    ax.plot(waves, ave_taus, c="k", lw=3, label="mean")

    # calculate the uncertainties on the mean optical depths
    # standard error of the mean
    ave_uncs = scipy.stats.sem(tau_list, axis=0)
    ax.plot(waves, ave_uncs, c="r", zorder=1, label="std. err. of mean")

    # plot the fitted Chiar+2013 model with 2 Gaussians
    chiar = Table.read(
        "/Users/mdecleir/Documents/MEAD/Literature_data/gcs3_6mu_gaussianfit.dat",
        format="ascii",
    )
    ax.plot(
        chiar["micron"],
        (chiar["olefinicgauss"] + chiar["pahgauss"]) / 29,
        color="darkviolet",
        ls=":",
        lw=3,
        label="Quintuplet (C13)",
    )

    # find the peak value and its uncertainty
    max_ind = np.argmax(ave_taus)
    mask = waves < 6.25
    max_ind = np.argmax(ave_taus[mask])
    print("Peak wavelength:", waves[max_ind])
    print("Peak tau/AV: ", ave_taus[max_ind], " +- ", ave_uncs[max_ind])
    print(
        "Peak AV/tau: ",
        1 / ave_taus[max_ind],
        1 / ave_taus[max_ind] - 1 / (ave_taus[max_ind] + ave_uncs[max_ind]),
        1 / (ave_taus[max_ind] - ave_uncs[max_ind]) - 1 / ave_taus[max_ind],
    )

    # finalize and save the figure
    ax.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=fs)
    ax.set_ylabel(r"$\tau(\lambda)/A(V)$", fontsize=fs)
    ax.axhline(c="k", ls=":", alpha=0.5)
    ax.set_xlim(6.05, 6.46)
    ax.set_ylim(-0.004, 0.01)

    # rename the previous version of the plot
    outname = datapath + "62_all.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")
    ax.legend(fontsize=0.8 * fs)
    fig.savefig(outname, bbox_inches="tight")


def stack_spectra_30(datapath, stars, ext_table, exclude):
    """
    Function to stack the optical depth around the 3.0 micron feature

    Parameters
    ----------
    datapath : string
        Path to the data files

    stars : list
        Star names

    ext_table : astropy Table
        Extinction properties

    exclude : list
        Stars to exclude from the average

    Returns
    -------
    Plot with all optical depths, average optical depths and standard error of the mean optical depths
    """
    # create an empty list to store the optical depths
    tau_list = []

    # create the figure
    fs = 18
    fig, ax = plt.subplots(figsize=(6, 6))

    for star in stars:
        # obtain the data
        data = Table.read(
            datapath + "NIRCam/v4/" + star + "_F322W2_fullSED.dat",
            format="ascii",
            data_start=1,
        )
        data["col1"].name = "wavelength"
        data["col2"].name = "flux"
        data["col3"].name = "unc"
        waves = data["wavelength"]

        # mask out the stellar and HI lines
        stellar_mask = (
            (waves > 2.738) & (waves <= 2.778)
            | (waves > 2.852) & (waves <= 2.893)
            | (waves > 3.027) & (waves <= 3.055)
            | ((waves > 3.074) & (waves <= 3.099))
            | ((waves > 3.278) & (waves <= 3.309))
        )
        # stellar lines per star
        # 2.743-2.778; 2.738-2.776; 2.748-2.763; 2.748-2.771; 2.745-2.767; 2.746-2.767; 2.748-2.767
        # 2.865-2.882; 2.852-2.893; 2.864-2.881; 2.852-2.887; 2.858-2.887; 2.865-2.881; 2.858-2.887; 2.862-2.882
        # 3.031-3.044; 3.027-3.047; 3.027-3.055; 3.031-3.046; 3.027-3.047; 3.027-3.046; 3.031-3.045; 3.027-3.052; 3.031-3.045
        # 3.076-3.106; 3.076-3.086; 3.076-3.096; 3.074-3.098; 3.076-3.097; 3.076-3.097; 3.081-3.098; 3.076-3.096
        # 3.285-3.308; 3.278-3.308; 3.278-3.308; 3.282-3.308; 3.281-3.308; 3.283-3.304; 3.289-3.304; 3.282-3.309
        # from stellar model:
        # 2.735-2.781; 2.846-2.897; 3.013-3.060; 3.072-3.091; 3.274-3.323

        # rebin the spectrum, and select the relevant region
        waves, fluxes, uncs = rebin_constres(data[~stellar_mask], (2.7, 3.29), 400)

        # define masks for the continuum fitting
        feat_reg_mask = (waves > 2.74) & (waves <= 3.23)
        cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)

        # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
        taus, unc, axes = fit_cont(waves, fluxes, cont_mask)

        # rename the previous version of the plot
        outname = datapath + star + "_30_cont.pdf"
        if os.path.isfile(outname):
            os.rename(outname, outname.split(".")[0] + "_0.pdf")
        plt.savefig(outname, bbox_inches="tight")

        # obtain A(V), and normalize the optical depths
        tab_mask = ext_table["Name"] == star

        # exclude star from the average, and plot it separately
        if star in exclude:
            fig2, ax2 = plt.subplots(figsize=(6, 6))
            ax2.plot(waves, taus / ext_table["AV"][tab_mask], c="k", label=star)

            max_ind = np.nanargmax(taus / ext_table["AV"][tab_mask])
            print(
                star,
                "Peak wavelength:",
                waves[max_ind],
                "Peak tau/AV: ",
                np.nanmax(taus / ext_table["AV"][tab_mask]),
                " +- ",
                unc / ext_table["AV"][tab_mask].value,
            )

            # add ice profiles from Decleir+2022 for 2 dense sightlines
            Drude_mod = custom_model(drude_modified)
            HD029_profile = Drude_mod(
                scale=0.03305712431358312 / 1.086,
                x_o=3.021603941274231,
                gamma_o=0.4878183911670602,
                asym=-1.316061662998598,
            )
            HD283_profile = Drude_mod(
                scale=0.02643756566455727 / 1.086,
                x_o=3.013890393471433,
                gamma_o=0.4373426133057677,
                asym=-4.382664830198743,
            )
            lin_mod = Linear1D()
            lin_fitter = LinearLSQFitter()
            cont029 = lin_fitter(
                lin_mod, waves[cont_mask], HD029_profile(waves)[cont_mask]
            )
            cont283 = lin_fitter(
                lin_mod, waves[cont_mask], HD283_profile(waves)[cont_mask]
            )
            ax2.plot(
                waves,
                HD029_profile(waves) - cont029(waves),
                color="darkviolet",
                ls=":",
                lw=1.5,
                label="HD029647 (D22)",
            )
            ax2.plot(
                waves,
                HD283_profile(waves) - cont283(waves),
                color="orange",
                ls="--",
                lw=1.5,
                label="HD283809 (D22)",
            )
            ax2.axhline(c="k", ls=":", alpha=0.5)
            ax2.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=fs)
            ax2.set_ylabel(r"$\tau(\lambda)/A(V)$", fontsize=fs)
            ax2.legend(fontsize=0.7 * fs)
            fig2.savefig(datapath + star + "_30.pdf", bbox_inches="tight")
            continue

        # add the normalized optical depth to the list
        tau_list.append(taus / ext_table["AV"][tab_mask])
        # tau_list.append(taus)

        # plot the normalized optical depths
        ax.plot(waves, taus / ext_table["AV"][tab_mask], lw=1, alpha=0.6)
    # ax.plot(waves, taus, lw=1, alpha=0.6)

    # average the normalized optical depths
    ave_taus = np.mean(tau_list, axis=0)
    ax.plot(waves, ave_taus, c="k", lw=3, label="mean")

    # add ice profiles from Decleir+2022 for 2 diffuse sightlines
    HD183_profile = Drude_mod(
        scale=0.0026 / 1.086,
        x_o=3.017747167372832,
        gamma_o=0.46258050223641395,
        asym=-2.8493632465986702,
    )
    HD229_profile = Drude_mod(
        scale=0.0042 / 1.086,
        x_o=3.017747167372832,
        gamma_o=0.46258050223641395,
        asym=-2.8493632465986702,
    )
    cont183 = lin_fitter(lin_mod, waves[cont_mask], HD183_profile(waves)[cont_mask])
    cont229 = lin_fitter(lin_mod, waves[cont_mask], HD229_profile(waves)[cont_mask])

    ax.plot(
        waves,
        HD183_profile(waves) - cont183(waves),
        color="darkviolet",
        ls=":",
        lw=3,
        label="HD183143 (D22)",
    )
    ax.plot(
        waves,
        HD229_profile(waves) - cont229(waves),
        color="orange",
        ls="--",
        lw=3,
        label="HD229238 (D22)",
    )

    # calculate the uncertainties on the mean optical depths
    # standard error of the mean
    ave_uncs = scipy.stats.sem(tau_list, axis=0)
    ax.plot(waves, ave_uncs, c="r", zorder=1, label="std. err. of mean")

    # find the peak value and its uncertainty
    max_ind = np.nanargmax(ave_taus)
    print("Peak wavelength:", waves[max_ind])
    print("Peak: ", ave_taus[max_ind], " +- ", ave_uncs[max_ind])

    # add the calibration star P330-E
    # obtain the data
    data = Table.read(
        datapath + "NIRCam/p330-e_Jy.dat",
        format="ascii",
    )
    data["col1"].name = "wavelength"
    data["col2"].name = "flux"
    waves = data["wavelength"]
    data["unc"] = 0.05 * data["flux"]
    stellar_mask = (
        (waves > 2.738) & (waves <= 2.778)
        | (waves > 2.852) & (waves <= 2.893)
        | (waves > 3.027) & (waves <= 3.055)
        | ((waves > 3.074) & (waves <= 3.099))
        | ((waves > 3.278) & (waves <= 3.309))
    )
    # rebin the spectrum, and select the relevant region
    waves, fluxes, uncs = rebin_constres(data[~stellar_mask], (2.77, 3.29), 400)
    # define masks for the continuum fitting
    feat_reg_mask = (waves > 2.8) & (waves <= 3.26)
    cont_mask = ~feat_reg_mask & ~np.isnan(fluxes)
    # fit and plot the continuum, normalize the spectrum, calculate the optical depth and its uncertainty
    taus, unc, axes = fit_cont(waves, fluxes, cont_mask)
    # rename the previous version of the plot
    outname = datapath + "P330-E_30_cont.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")
    plt.savefig(outname, bbox_inches="tight")
    # add to the main figure
    # ax.plot(waves, taus / 0.133, c="blue", zorder=1, label="P330-E")
    # ax.plot(waves, taus, c="blue", zorder=1, label="P330-E")

    # finalize and save the figure
    ax.set_xlabel(r"$\lambda$ ($\mu$m)", fontsize=fs)
    ax.set_ylabel(r"$\tau(\lambda)/A(V)$", fontsize=fs)
    # ax.set_ylabel(r"$\tau(\lambda)$", fontsize=fs)
    ax.axhline(c="k", ls=":", alpha=0.5)
    ax.set_ylim(-0.004, 0.01)

    # rename the previous version of the plot
    outname = datapath + "30_all.pdf"
    if os.path.isfile(outname):
        os.rename(outname, outname.split(".")[0] + "_0.pdf")
    ax.legend(fontsize=0.7 * fs)
    fig.savefig(outname, bbox_inches="tight")


def main():
    # plotting settings for uniform plots
    fs = 18
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)

    # define the data path
    datapath = "/Users/mdecleir/Documents/MEAD/extinction/JWST_data/"

    # list the stars
    stars = [
        "HD014434",
        "HD038087",
        "HD073882",
        "HD147888",
        "HD152249",
        "HD203938",
        "HD206267",
        "HD207198",
        "HD216898",
    ]

    # define the stars that have noisy MIRI spectra
    bad_stars = ["HD014434", "HD038087"]

    # sort the stars by silicate feature strength by giving them an index. 0=weakest feature.
    sort_idx = [
        3,
        5,
        4,
        7,
        0,
        6,
        1,
        2,
        8,
    ]

    # obtain and plot a stellar model
    # stel_mod = Table.read(datapath + "MIRI/tlusty_z100t25000g400v2_miri_ifu.fits")
    # stel_mod = Table.read(datapath + "MIRI/tlusty_z100t23000g400v2_nircam_ss.fits")
    # plt.plot(
    #     stel_mod["WAVELENGTH"] * 1e-4,
    #     stel_mod["FLUX"] * (stel_mod["WAVELENGTH"] * 1e-4) ** 4,
    # )
    # plt.show()

    # fit and plot all features for all stars
    fit_all(datapath, stars, sort_idx)

    # fit the feature for some dust grain models
    fit_grain_mod(datapath)

    # obtain the extinction properties
    ext_table = Table.read(
        "/Users/mdecleir/Documents/MEAD/Literature_data/Gordon+2009_tab2.dat",
        format="ascii",
    )

    # stack the spectra around 3.4 micron
    stack_spectra_34(datapath, stars, ext_table)

    # stack the spectra around 6.2 micron
    stack_spectra_62(datapath, stars, ext_table, bad_stars)

    # define stars to be excluded from the stack
    exclude = ["HD073882"]

    # stack the spectra around 3.0 micron
    stack_spectra_30(datapath, stars, ext_table, exclude)


if __name__ == "__main__":
    main()
