# fit_features.py: script to fit the features in the JWST MEAD spectra

# import the necessary packages
import numpy as np

from astropy.table import Table
from astropy.modeling import models, fitting
from matplotlib import pyplot as plt
from models_mcmc_extension import EmceeFitter


def fit_58(datapath, outpath, star):
    """
    Function to fit the continuum and the feature at 5.8 micron

    Parameters
    ----------
    datapath : string
        Path to the data files

    outpath : string
        Path to store the output plots

    stars : string
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
    data = Table.read(datapath + starname + "_nircam_mrs_merged.fits")
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
    axes[0].plot(waves[rangemask], fluxes[rangemask] * waves[rangemask] ** 2)
    axes[0].scatter(waves[cont_mask], fluxes[cont_mask] * waves[cont_mask] ** 2, c="r")
    axes[0].plot(waves[rangemask], fit_result_cont(waves[rangemask]))
    axes[0].set_ylabel("flux (Jy)", fontsize=fs)

    # normalize the fluxes
    norm_fluxes = (
        fluxes[rangemask] * waves[rangemask] ** 2 / fit_result_cont(waves[rangemask])
    )

    # convert to optical depth (only exists in rangemask)
    tau = np.log(1 / norm_fluxes)

    # calculate the empirical uncertainy on the optical depth in clean regions
    unc = np.std(tau[cont_mask[rangemask]])

    # mask the stellar lines inside the feature and define the mask for the feature fitting
    stellar_lines_in = ((waves > 5.703) & (waves <= 5.717)) | (
        (waves > 5.823) & (waves <= 5.841)
    )
    # stellar line masks per star:
    # 5.705-5.717, 5.706-5.715, 5.703-716, 5.712-5.717, 5.711-5.717
    # 5.823-5.831, 5.824-5.836, 5.827-5.839, 5.829-5.841, 5.825-5.836, 5.823-5.840
    feat_full_mask = feat_reg_mask & ~stellar_lines_in
    feat_fit_mask = feat_full_mask[rangemask] & ~np.isnan(tau)

    # define the uncertainties
    uncs = np.full(len(tau[feat_fit_mask]), unc)

    # fit the feature with the LevMarLSQFitter
    gauss_mod = models.Gaussian1D(
        amplitude=0.01,
        stddev=0.02,
        mean=5.81,
        bounds={
            "mean": (5.78, 5.84),
            "stddev": (1e-4, 0.15),
        },
    )
    lev_fitter = fitting.LevMarLSQFitter()
    fit_result_feat_lev = lev_fitter(
        gauss_mod,
        waves[rangemask][feat_fit_mask],
        tau[feat_fit_mask],
        weights=1.0 / uncs,
        maxiter=10000,
        filter_non_finite=True,
    )

    # fit the feature again with MCMC
    emcee_samples_file = outpath + "Fitting_results/" + star + "_chains_58.h5"
    nsteps = 5000
    burn = 0.1
    emcee_fitter = EmceeFitter(
        nsteps=nsteps, burnfrac=burn, save_samples=emcee_samples_file
    )
    fit_result_feat_emcee = emcee_fitter(
        fit_result_feat_lev,
        waves[rangemask][feat_fit_mask],
        tau[feat_fit_mask],
        weights=1.0 / uncs,
    )

    # obtain the MCMC chains
    chains = emcee_fitter.fit_info["sampler"].get_chain(
        flat=True, discard=np.int32(burn * nsteps)
    )

    # plot the MCMC fitting results
    emcee_fitter.plot_emcee_results(
        fit_result_feat_emcee,
        filebase=outpath + "Fitting_results/" + star + "_58",
    )

    # plot the feature fits
    axes[1].plot(waves[rangemask], tau)
    axes[1].plot(waves[rangemask], fit_result_feat_lev(waves[rangemask]), c="green")
    axes[1].plot(waves[rangemask], fit_result_feat_emcee(waves[rangemask]), c="pink")
    axes[1].set_xlabel(r"wavelength ($\mu$m)", fontsize=fs)
    axes[1].set_ylabel("optical depth", fontsize=fs)
    axes[1].axhline(ls=":", c="k")
    plt.savefig(outpath + star + "_58_fit.pdf", bbox_inches="tight")

    return fit_result_feat_emcee, chains


def fit_all_58(datapath, outpath, stars, ext_table):
    """
    Fit the 5.8 micron feature for all stars

    Parameters
    ----------
    datapath : string
        Path to the data files

    outpath : string
        Path to store the output plots and tables

    stars : list
        Star names

    ext_table : Astropy Table
        Table with extinction measurements

    Returns
    -------
    Tables with fitting results
    """
    # create a table to store the results
    result_tab = Table(
        names=(
            "name",
            "amplitude",
            "amp_unc_min",
            "amp_unc_plus",
            "wavelength(micron)",
            "wavelength_unc_min(micron)",
            "wavelength_unc_plus(micron)",
            "std(micron)",
            "std_unc_min(micron)",
            "std_unc_plus(micron)",
            "area(cm-1)",
            "area_unc_min(cm-1)",
            "area_unc_plus(cm-1)",
            "A(V)",
            "R(V)",
        ),
        dtype=(
            "str",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
        ),
    )

    # fit the 5.8 micron feature for all stars
    for star in stars:
        fit_result, chains = fit_58(datapath, outpath, star)

        # obtain the results
        amp = fit_result.amplitude.value
        amp_unc_plus = fit_result.amplitude.unc_plus
        amp_unc_min = fit_result.amplitude.unc_minus
        amp_unc = fit_result.amplitude.unc
        mean = fit_result.mean.value
        mean_unc_plus = fit_result.mean.unc_plus
        mean_unc_min = fit_result.mean.unc_minus
        mean_unc = fit_result.mean.unc
        stddev = fit_result.stddev.value
        stddev_unc_plus = fit_result.stddev.unc_plus
        stddev_unc_min = fit_result.stddev.unc_minus
        stddev_unc = fit_result.stddev.unc

        # convert the standard deviation from units of wavelengths (micron) to wavenumbers (cm^-1)
        # dnu = 1/(lambda**2) * dlambda * 1e4
        # lambda = mean = chains[:,1]
        # dlambda = stddev = chains[:,2]
        stddev_wavenm = 1.0 / (chains[:, 1] ** 2) * chains[:, 2] * 1e4

        # calculate the integrated area (in cm^-1)
        # area = amplitude * stddev (in cm^-1)* sqrt(2 * pi)
        # amplitude = chains[:,0]
        area_chain = chains[:, 0] * stddev_wavenm * np.sqrt(2 * np.pi)
        area16, area, area84 = np.percentile(area_chain, [16, 50, 84])
        area_unc_min = area - area16
        area_unc_plus = area84 - area

        # obtain A(V) and R(V)
        mask = ext_table["Name"] == star
        AV = ext_table[mask]["AV"]
        RV = ext_table[mask]["RV"]

        # add the results to a table
        result_tab.add_row(
            (
                star,
                amp,
                amp_unc_min,
                amp_unc_plus,
                mean,
                mean_unc_min,
                mean_unc_plus,
                stddev,
                stddev_unc_min,
                stddev_unc_plus,
                area,
                area_unc_min,
                area_unc_plus,
                AV,
                RV,
            )
        )

    # write the table to a file
    result_tab.write(outpath + "fit_results_58.txt", format="ascii", overwrite=True)


def fit_10(datapath, outpath, star):
    """
    Function to fit the continuum and the feature at 10 micron

    Parameters
    ----------
    datapath : string
        Path to the data files

    outpath : string
        Path to store the output plots

    stars : string
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
    data = Table.read(datapath + starname + "_nircam_mrs_merged.fits")
    waves = data["wavelength"]
    fluxes = data["flux"]

    # define masks for the continuum fitting
    rangemask = (waves > 7) & (waves <= 14)
    feat_reg_mask = (waves > 8) & (waves <= 12.5)
    stellar_lines_out = ((waves > 7.424) & (waves <= 7.538)) | (
        (waves > 12.570) & (waves <= 12.618)
    )
    # feature region per star:
    # 8.12-11.96, 8.07-12.19, 8.35-12.27, 8.42-12.02, 8.19-12.10, 8.15-12.10, 8.15-12.15, 8.31-12.29
    # stellar line masks per star:
    # 7.435-7.522, 7.449-7.471, 7.443-7.538, 7.434-7.520, 7.454-7.467, 7.424-7.522, 7.428-7.473, 7.453-7.462
    # 7.497-7.512, 7.494-7.512
    # 12.570-12.618

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
    axes[0].plot(waves[rangemask], fluxes[rangemask] * waves[rangemask] ** 2)
    axes[0].scatter(waves[cont_mask], fluxes[cont_mask] * waves[cont_mask] ** 2, c="r")
    axes[0].plot(waves[rangemask], fit_result_cont(waves[rangemask]))
    axes[0].set_ylabel("flux (Jy)", fontsize=fs)

    # normalize the fluxes
    norm_fluxes = (
        fluxes[rangemask] * waves[rangemask] ** 2 / fit_result_cont(waves[rangemask])
    )

    # convert to optical depth (only exists in rangemask)
    tau = np.log(1 / norm_fluxes)

    # calculate the empirical uncertainy on the optical depth in clean regions
    unc = np.std(tau[cont_mask[rangemask]])

    # mask the stellar lines inside the feature and define the mask for the feature fitting
    stellar_lines_in = (
        ((waves > 8.732) & (waves <= 8.775))
        | ((waves > 9.381) & (waves <= 9.403))
        | ((waves > 10.447) & (waves <= 10.575))
        | ((waves > 11.251) & (waves <= 11.373))
        | ((waves > 12.312) & (waves <= 12.445))
    )
    # stellar line masks per star:
    # 8.741-8.772, 8.732-8.775, 8.749-8.766
    # 9.381-9.397, 9.381-9.403
    # 10.447-10.575, 10.488-10.505
    # 11.285-11.321, 11.266-11.314, 11.251-11.373, 11.296-11.314
    # 12.352-12.424, 12.355-12.384, 12.312-12.429, 12.357-12.415, 12.312-12.445, 12.354-12.402, 12.355-12.381
    feat_full_mask = feat_reg_mask & ~stellar_lines_in
    feat_fit_mask = feat_full_mask[rangemask] & ~np.isnan(tau)

    # define the uncertainties
    uncs = np.full(len(tau[feat_fit_mask]), unc)

    # fit the feature with the LevMarLSQFitter
    gauss_mod = models.Gaussian1D(
        amplitude=0.1,
        stddev=1,
        mean=10,
        bounds={
            "mean": (9.5, 10.5),
        },
    )
    lev_fitter = fitting.LevMarLSQFitter()
    fit_result_feat_lev = lev_fitter(
        gauss_mod,
        waves[rangemask][feat_fit_mask],
        tau[feat_fit_mask],
        weights=1.0 / uncs,
        maxiter=10000,
        filter_non_finite=True,
    )

    # fit the feature again with MCMC
    emcee_samples_file = outpath + "Fitting_results/" + star + "_chains_10.h5"
    nsteps = 5000
    burn = 0.1
    emcee_fitter = EmceeFitter(
        nsteps=nsteps, burnfrac=burn, save_samples=emcee_samples_file
    )
    fit_result_feat_emcee = emcee_fitter(
        fit_result_feat_lev,
        waves[rangemask][feat_fit_mask],
        tau[feat_fit_mask],
        weights=1.0 / uncs,
    )

    # obtain the MCMC chains
    chains = emcee_fitter.fit_info["sampler"].get_chain(
        flat=True, discard=np.int32(burn * nsteps)
    )

    # plot the MCMC fitting results
    emcee_fitter.plot_emcee_results(
        fit_result_feat_emcee,
        filebase=outpath + "Fitting_results/" + star + "_10",
    )

    # plot the feature fits
    axes[1].plot(waves[rangemask], tau)
    axes[1].plot(waves[rangemask], fit_result_feat_lev(waves[rangemask]), c="green")
    axes[1].plot(waves[rangemask], fit_result_feat_emcee(waves[rangemask]), c="pink")
    axes[1].set_xlabel(r"wavelength ($\mu$m)", fontsize=fs)
    axes[1].set_ylabel("optical depth", fontsize=fs)
    axes[1].axhline(ls=":", c="k")
    plt.savefig(outpath + star + "_10_fit.pdf", bbox_inches="tight")

    return fit_result_feat_emcee, chains


def fit_all_10(datapath, outpath, stars, ext_table):
    """
    Fit the 10 micron feature for all stars

    Parameters
    ----------
    datapath : string
        Path to the data files

    outpath : string
        Path to store the output plots and tables

    stars : list
        Star names

    ext_table : Astropy Table
        Table with extinction measurements

    Returns
    -------
    Tables with fitting results
    """
    # create a table to store the results
    result_tab = Table(
        names=(
            "name",
            "amplitude",
            "amp_unc_min",
            "amp_unc_plus",
            "wavelength(micron)",
            "wavelength_unc_min(micron)",
            "wavelength_unc_plus(micron)",
            "std(micron)",
            "std_unc_min(micron)",
            "std_unc_plus(micron)",
            "area(cm-1)",
            "area_unc_min(cm-1)",
            "area_unc_plus(cm-1)",
            "A(V)",
            "R(V)",
        ),
        dtype=(
            "str",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
            "float64",
        ),
    )

    # fit the 10 micron feature for all stars
    for star in stars:
        fit_result, chains = fit_10(datapath, outpath, star)

        # obtain the results
        amp = fit_result.amplitude.value
        amp_unc_plus = fit_result.amplitude.unc_plus
        amp_unc_min = fit_result.amplitude.unc_minus
        amp_unc = fit_result.amplitude.unc
        mean = fit_result.mean.value
        mean_unc_plus = fit_result.mean.unc_plus
        mean_unc_min = fit_result.mean.unc_minus
        mean_unc = fit_result.mean.unc
        stddev = fit_result.stddev.value
        stddev_unc_plus = fit_result.stddev.unc_plus
        stddev_unc_min = fit_result.stddev.unc_minus
        stddev_unc = fit_result.stddev.unc

        # convert the standard deviation from units of wavelengths (micron) to wavenumbers (cm^-1)
        # dnu = 1/(lambda**2) * dlambda * 1e4
        # lambda = mean = chains[:,1]
        # dlambda = stddev = chains[:,2]
        stddev_wavenm = 1.0 / (chains[:, 1] ** 2) * chains[:, 2] * 1e4

        # calculate the integrated area (in cm^-1)
        # area = amplitude * stddev (in cm^-1)* sqrt(2 * pi)
        # amplitude = chains[:,0]
        area_chain = chains[:, 0] * stddev_wavenm * np.sqrt(2 * np.pi)
        area16, area, area84 = np.percentile(area_chain, [16, 50, 84])
        area_unc_min = area - area16
        area_unc_plus = area84 - area

        # obtain A(V) and R(V)
        mask = ext_table["Name"] == star
        AV = ext_table[mask]["AV"]
        RV = ext_table[mask]["RV"]

        # add the results to a table
        result_tab.add_row(
            (
                star,
                amp,
                amp_unc_min,
                amp_unc_plus,
                mean,
                mean_unc_min,
                mean_unc_plus,
                stddev,
                stddev_unc_min,
                stddev_unc_plus,
                area,
                area_unc_min,
                area_unc_plus,
                AV,
                RV,
            )
        )

    # write the table to a file
    result_tab.write(outpath + "fit_results_10.txt", format="ascii", overwrite=True)


def main():
    # plotting settings for uniform plots
    fs = 18
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)

    # define the data path
    datapath = "/Users/mdecleir/Documents/MEAD/JWST_ext/"

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

    # obtain and plot a stellar model
    stel_mod = Table.read(datapath + "MIRI/tlusty_z100t25000g400v2_miri_ifu.fits")
    plt.plot(
        stel_mod["WAVELENGTH"] * 1e-4,
        stel_mod["FLUX"] * (stel_mod["WAVELENGTH"] * 1e-4) ** 4,
    )
    # plt.xlim(5.6, 6.1)
    plt.xlim(7, 14)
    # plt.ylim(3.2e6, 3.4e6)
    # plt.show()

    # obtain the table with extinction properties
    ext_table = Table.read(datapath + "Gordon+2009_tab2.dat", format="ascii")

    # fit the 5.8 micron feature for all stars
    fit_all_58(datapath + "MIRI/v3/", datapath + "MIRI/", stars, ext_table)

    # fit the 10 micron feature for all stars
    fit_all_10(datapath + "MIRI/v3/", datapath + "MIRI/", stars, ext_table)


if __name__ == "__main__":
    main()
