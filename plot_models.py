# plot_models.py: script to plot some example models

# import the necessary packages
import numpy as np

from astropy.modeling.models import custom_model
from matplotlib import pyplot as plt
from scipy import stats

from astropy.modeling.models import Gaussian1D
from astropy.table import Table


def gauss_skew_func(x, amplitude=0.1, loc=9, scale=1, shape=2):
    return amplitude * stats.skewnorm.pdf(x, shape, loc=loc, scale=scale)


def plot_skew_gauss():
    # plot some example models
    x = np.arange(6, 14, 0.01)

    model1 = gauss_skew_func(
        x,
        amplitude=0.15,
        loc=9,
        scale=1,
        shape=3,
    )
    model2 = gauss_skew_func(
        x,
        amplitude=0.15,
        loc=9,
        scale=2,
        shape=3,
    )
    model3 = gauss_skew_func(
        x,
        amplitude=0.15,
        loc=10,
        scale=1,
        shape=3,
    )
    model4 = gauss_skew_func(
        x,
        amplitude=0.1,
        loc=9,
        scale=1,
        shape=3,
    )
    model5 = gauss_skew_func(
        x,
        amplitude=0.15,
        loc=9,
        scale=1,
        shape=-5,
    )

    plt.plot(x, model1, label="amp=0.15, loc=9, scale=1, shape=3")
    plt.plot(x, model2, ls="--", label="amp=0.15,loc=9, scale=2, shape=3")
    plt.plot(x, model3, ls=":", label="amp=0.15, loc=10, scale=1, shape=3")
    plt.plot(x, model4, ls="-.", label="amp=0.1,loc=9, scale=1, shape=3")
    plt.plot(x, model5, label="amp=0.15, loc=9, scale=1, shape=-5")

    plt.xlabel("x")
    plt.legend()
    plt.savefig("/Users/mdecleir/Documents/MEAD/Extinction/gauss_skewed.pdf")


def gamma(x, x_o=1, gamma_o=1, asym=1):
    """
    Gamma function (wavelength dependent width, replacing the FWHM)
    """
    return 2 * gamma_o / (1 + np.exp(asym * (x - x_o)))


def gauss_modified_func(x, amp=1, x_o=1, gamma_o=1, asym=1):
    return amp * np.exp(-((x - x_o) ** 2) / (2 * gamma(x, x_o, gamma_o, asym) ** 2))


def plot_modif_gauss():
    x = np.arange(6, 14, 0.01)

    model1 = gauss_modified_func(x, amp=0.15, x_o=9, gamma_o=0.5, asym=0.3)
    model2 = gauss_modified_func(x, amp=0.1, x_o=9, gamma_o=0.5, asym=0.3)
    model3 = gauss_modified_func(x, amp=0.15, x_o=10, gamma_o=0.5, asym=0.3)
    model4 = gauss_modified_func(x, amp=0.15, x_o=9, gamma_o=1, asym=0.3)
    model5 = gauss_modified_func(x, amp=0.15, x_o=9, gamma_o=0.5, asym=-0.5)

    plt.clf()
    plt.plot(x, model1, label="amp=0.15, xo=9, gammao=0.5, asym=0.3")
    plt.plot(x, model2, ls="--", label="amp=0.1, xo=9, gammao=0.5, asym=0.3")
    plt.plot(x, model3, ls=":", label="amp=0.15, xo=10, gammao=0.5, asym=0.3")
    plt.plot(x, model4, ls="-.", label="amp=0.15, xo=9, gammao=1, asym=0.3")
    plt.plot(x, model5, label="amp=0.15, xo=9, gammao=0.5, asym=-0.5")

    plt.xlabel("x")
    plt.legend()
    plt.savefig("/Users/mdecleir/Documents/MEAD/Extinction/gauss_modified.pdf")


def plot_gauss3():
    waves = np.arange(3.1, 3.6, 0.001)
    chiar = Table.read(
        "/Users/mdecleir/Documents/MEAD/Literature_data/gcs3_gaussfit.dat",
        format="ascii",
    )

    plt.plot(chiar["micron"], chiar["gauss1"], color="k")
    g1 = Gaussian1D(
        amplitude=0.044,
        mean=3.289,
        stddev=81.8 * (3.289 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g1(waves), ls=":")

    plt.plot(chiar["micron"], chiar["gauss2"], color="k")
    g2 = Gaussian1D(
        amplitude=0.132,
        mean=3.376,
        stddev=47.2 * (3.376 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g2(waves), ls=":")

    plt.plot(chiar["micron"], chiar["gauss3"], color="k")
    g3 = Gaussian1D(
        amplitude=0.149,
        mean=3.420,
        stddev=42.8 * (3.42 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g3(waves), ls=":")

    plt.plot(chiar["micron"], chiar["gauss4"], color="k")
    g4 = Gaussian1D(
        amplitude=0.115,
        mean=3.474,
        stddev=41 * (3.474 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g4(waves), ls=":")

    plt.plot(chiar["micron"], chiar["gauss5"], color="k")
    g5 = Gaussian1D(
        amplitude=0.065,
        mean=3.52,
        stddev=40.4 * (3.52 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g5(waves), ls=":")

    plt.show()


def plot_gauss6():
    waves = np.arange(5.9, 6.5, 0.001)
    chiar = Table.read(
        "/Users/mdecleir/Documents/MEAD/Literature_data/gcs3_6mu_gaussianfit.dat",
        format="ascii",
    )

    plt.plot(chiar["micron"], chiar["olefinicgauss"], color="k")
    g1 = Gaussian1D(
        amplitude=0.07,
        mean=6.19,
        stddev=15 * (6.19 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g1(waves), ls=":")

    plt.plot(chiar["micron"], chiar["pahgauss"], color="k")
    g2 = Gaussian1D(
        amplitude=0.03,
        mean=6.25,
        stddev=40 * (6.25 ** 2) / 10 ** 4 / 2 / np.sqrt(2 * np.log(2)),
        fixed={"mean": True, "stddev": True},
    )
    plt.plot(waves, g2(waves), ls=":")

    plt.show()


def main():
    # plot skewed Gaussian models
    plot_skew_gauss()

    # plot modified Gaussian models
    plot_modif_gauss()

    # plot Gaussian models around 3.4 micron from Chiar+2013
    plot_gauss3()

    # plot Gaussian models around 6.2 micron from Chiar+2013
    plot_gauss6()


if __name__ == "__main__":
    main()
