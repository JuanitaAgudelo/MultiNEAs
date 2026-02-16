#############################################################
# Astropy Constants
#############################################################
list_constants = [
    "G",
    "N_A",
    "R",
    "Ryd",
    "a0",
    "alpha",
    "atm",
    "b_wien",
    "c",
    "e",
    "eps0",
    "g0",
    "h",
    "hbar",
    "k_B",
    "m_e",
    "m_n",
    "m_p",
    "mu0",
    "muB",
    "sigma_T",
    "sigma_sb",
    "u",
    "GM_earth",
    "GM_jup",
    "GM_sun",
    "L_bol0",
    "L_sun",
    "M_earth",
    "M_jup",
    "M_sun",
    "R_earth",
    "R_jup",
    "R_sun",
    "au",
    "kpc",
    "pc",
]
exec("from astropy.constants import " + ", ".join(list_constants))
for constant in list_constants:
    exec(f"{constant} = {constant}.value")

#############################################################
# OTHER CONSTANTS
#############################################################
import numpy as np

# Angles
rad = 180 / np.pi
deg = 1 / rad

# Time
day = 86400
year = 365.25 * day

# WSG84
rearth = 6378137.0  # meters
fe = 1 / 298.257223563

# Rates
# DIFFERENTIAL RATES OF TCG AND TCB, SEE USNO CIRCULAR 179, OCT 2006
# TDB TIME AT 01/01/01 1977 00:00:00 TAI, 01/01/01 1977 00:00:32.184 TDB, 12/31/1976 23:59:45 UTC
T0 = -725803167.816
LG = 6.969290134e-10
LB = 1.55e-8


#############################################################
# CONSTANTES ACTUALIZADAS
#############################################################
mu_mercury = 22031.868551e9  # m^3/s^2, SPICE Kernels DE441
mu_venus = 324858.592000e9  # m^3/s^2, SPICE Kernels DE441
mu_earth = 398600.435507e9  # m^3/s^2, SPICE Kernels DE441
mu_emb = 403503.235625e9  # m^3/s^2, SPICE Kernels DE441
mu_mars = 42828.375816e9  # m^3/s^2, SPICE Kernels DE441
mu_jupiter = 126712764.100000e9  # m^3/s^2, SPICE Kernels DE441
mu_saturn = 37940584.841800e9  # m^3/s^2, SPICE Kernels DE441
mu_uranus = 5794556.400000e9  # m^3/s^2, SPICE Kernels DE441
mu_neptune = 6836527.100580e9  # m^3/s^2, SPICE Kernels DE441
mu_pluto = 975.500000e9  # m^3/s^2, SPICE Kernels DE441
mu_sun = 132712440041.279419e9  # m^3/s^2, SPICE Kernels DE441
mu_moon = 4902.800118e9  # m^3/s^2, SPICE Kernels DE441
