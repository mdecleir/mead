# plot_models.py: script to plot some example models

# import the necessary packages
import numpy as np

from astropy.modeling.models import custom_model
from matplotlib import pyplot as plt
from scipy import stats


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


def main():
    # plot skewed Gaussian models
    plot_skew_gauss()

    # plot modified Gaussian models
    plot_modif_gauss()


if __name__ == "__main__":
    main()
