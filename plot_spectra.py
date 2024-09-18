import numpy as np

from astropy.table import Table
from matplotlib import pyplot as plt

from fit_features import rebin_constres


def plot_spectra(datapath, stars, sort_idx):
    """
    Function to plot all spectra

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
    Plot with all spectra
    """
    # create the figure
    fs = 20
    plt.rc("xtick", top=True, direction="in", labelsize=fs * 0.8)
    plt.rc("ytick", right=True, direction="in", labelsize=fs * 0.8)
    fig, ax = plt.subplots(figsize=(16, 2 * len(stars)))

    for i, star in enumerate(stars):
        # obtain the data
        starname = "hd" + star.split("D")[1].strip("0")
        data = Table.read(datapath + "MIRI/v3/" + starname + "_nircam_mrs_merged.fits")

        # rebin the spectra, and select the relevant region
        waves, fluxes, uncs = rebin_constres(data, (2.3, 20), 400)

        # normalize the spectra
        norm_mask = (waves > 5) & (waves < 7.5)
        average = np.nanmean(fluxes[norm_mask] * waves[norm_mask] ** 2)
        fluxes = fluxes / average

        # plot the spectra
        p = plt.plot(waves, fluxes * waves ** 2 + 0.2 * sort_idx[i])

        # add the star name
        plt.annotate(
            star,
            (1.8, fluxes[30] * waves[30] ** 2 + 0.2 * sort_idx[i]),
            fontsize=fs * 0.8,
            rotation=30,
            color=p[0].get_color(),
        )

    # finalize and save the figure
    plt.ylim(0.8, 3.2)
    plt.xlabel(r"$\lambda (\mu m)$", fontsize=fs)
    plt.ylabel(r"normalized flux * $\lambda^2 (\mu m^2)$ + offset", fontsize=fs)
    plt.savefig(datapath + "all_spectra.pdf", bbox_inches="tight")


def main():
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

    # sort the stars by steepness of their spectra by giving them an index. 0=flattest.
    sort_idx = [8, 2, 6, 1, 7, 5, 4, 3, 0]

    # plot all spectra
    plot_spectra(datapath, stars, sort_idx)


if __name__ == "__main__":
    main()
