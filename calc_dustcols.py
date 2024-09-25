import numpy as np

from astropy.table import Table, join


def calc_Ndust(lit_table, element, solar, solar_unc):
    """
    Function to calculate the dust column density for one element

    Parameters
    ----------
    lit_table : Astropy Table
        Table with the literature gas and hydrogen column densities

    element : string
        Name of the element

    solar : float
        Logarithm of the solar abundance of this element

    solar_unc : float
        Uncertainty on the logarithm of the solar abundance of this element

    Returns
    -------
    Dust column density and uncertainties
    """
    # calculate the dust column density
    # Ndust = Ntot - Ngas
    Ntot = 10 ** solar * 10 ** lit_table["logNH"]
    Ngas = 10 ** lit_table["N(" + element + ")"]
    Ndust = Ntot - Ngas

    # calculate the uncertainty on the dust column density
    # Ndust_unc = sqrt(Ntot_unc**2 + Ngas_unc**2)
    # Ngas_unc = ln(10) * Ngas * logN_unc
    # Ntot_unc = ln(10) * Ntot * sqrt(solar_unc**2 + logH_unc**2 )
    Ngas_munc = np.log(10) * Ngas * lit_table["N(" + element + ")_munc"]
    Ngas_punc = np.log(10) * Ngas * lit_table["N(" + element + ")_punc"]
    Ntot_unc = np.log(10) * Ntot * np.sqrt(solar_unc ** 2 + lit_table["e_logNH"] ** 2)
    Ndust_munc = np.sqrt(Ntot_unc ** 2 + Ngas_munc ** 2)
    Ndust_punc = np.sqrt(Ntot_unc ** 2 + Ngas_punc ** 2)

    return Ndust, Ndust_munc, Ndust_punc


def calc_all_Ndust(H_table, gcol_table):
    """
    Function to calculate the dust column densities for all elements

    Parameters
    ----------
    H_table : Astropy Table
        Table with the literature hydrogen column densities

    gcol_table : Astropy Table
        Table with the literature gas column densities

    Returns
    -------
    Table with literature dust column densities
    """
    # join both literature tables
    lit_table = join(H_table, gcol_table, keys_left="Star", keys_right="star")

    # calculate the dust column densities for Mg, Fe and O
    (
        gcol_table["N(Mg)_d"],
        gcol_table["N(Mg)_d_munc"],
        gcol_table["N(Mg)_d_punc"],
    ) = calc_Ndust(lit_table, "Mg", 7.62 - 12, 0.02)
    (
        gcol_table["N(Fe)_d"],
        gcol_table["N(Fe)_d_munc"],
        gcol_table["N(Fe)_d_punc"],
    ) = calc_Ndust(lit_table, "Fe", 7.54 - 12, 0.03)
    (
        gcol_table["N(O)_d"],
        gcol_table["N(O)_d_munc"],
        gcol_table["N(O)_d_punc"],
    ) = calc_Ndust(lit_table, "O", 8.76 - 12, 0.05)

    return gcol_table


def main():
    # define the literature data path
    litpath = "/Users/mdecleir/Documents/MEAD/Literature_data/"

    # obtain the literature (Ritchey+2023 and Jenkins 2009) gas column densities
    gcol_table = Table.read(litpath + "lit_dcols.dat", format="ascii")

    # obtain the hydrogen column densities, from table 2 in Van De Putte+2023
    H_table = Table.read(litpath + "VanDePutte+2023_tab2.dat", format="ascii")[
        "Star", "logNH", "e_logNH"
    ]

    # calculate the dust column densities
    result_table = calc_all_Ndust(H_table, gcol_table)

    # write the output table to a file
    result_table.write(litpath + "lit_dcols.dat", format="ascii", overwrite=True)


if __name__ == "__main__":
    main()
