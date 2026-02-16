from astropy.time import Time
import spiceypy as spy
import numpy as np
import rebound as rb
from datetime import datetime, timedelta, timezone
from time import time
import calendar
import pandas as pd
import os
import re
import sys
import json
import warnings
from urllib import request
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Point
from typing import List

from sklearn.linear_model import LinearRegression
import requests
import gzip

####################################################
# Directories
####################################################
pack_dir = os.path.abspath(os.path.dirname(__file__))
data_dir = "data"
objects_dir = f"{data_dir}/objects"
log_dir = "log"
object_suf = "object"

####################################################
# Asteroid classes
####################################################
# Source: https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html
asteroid_classes = {
    "IEO": {
        "name": "Atira",
        "type": "asteroid",
        "description": "An asteroid orbit contained entirely within the orbit of the Earth (Q < 0.983 au). Also known as an Interior Earth Object",
    },
    "ATE": {
        "name": "Aten",
        "type": "asteroid",
        "description": "Near-Earth asteroid orbits similar to that of 2062 Aten (a < 1.0 au; Q > 0.983 au)",
    },
    "APO": {
        "name": "Apollo",
        "type": "asteroid",
        "description": "Near-Earth asteroid orbits which cross the Earth’s orbit similar to that of 1862 Apollo (a > 1.0 au; q < 1.017 au)",
    },
    "AMO": {
        "name": "Amor",
        "type": "asteroid",
        "description": "Near-Earth asteroid orbits similar to that of 1221 Amor (1.017 au < q < 1.3 au)",
    },
    "MCA": {
        "name": "Mars-crossing Asteroid",
        "type": "asteroid",
        "description": "Asteroids that cross the orbit of Mars constrained by (1.3 au < q < 1.666 au; a < 3.2 au)",
    },
    "IMB": {
        "name": "Inner Main-belt Asteroid",
        "type": "asteroid",
        "description": "Asteroids with orbital elements constrained by (a < 2.0 au; q > 1.666 au)",
    },
    "MBA": {
        "name": "Main-belt Asteroid",
        "type": "asteroid",
        "description": "Asteroids with orbital elements constrained by (2.0 au < a < 3.2 au; q > 1.666 au)",
    },
    "OMB": {
        "name": "Outer Main-belt Asteroid",
        "type": "asteroid",
        "description": "Asteroids with orbital elements constrained by (3.2 au < a < 4.6 au)",
    },
    "TJN": {
        "name": "Jupiter Trojan",
        "type": "asteroid",
        "description": "Asteroids trapped in Jupiter’s L4/L5 Lagrange points (4.6 au < a < 5.5 au; e < 0.3)",
    },
    "AST": {
        "name": "Asteroid",
        "type": "asteroid",
        "description": "Asteroid orbit not matching any defined orbit class",
    },
    "CEN": {
        "name": "Centaur",
        "type": "",
        "description": "Objects with orbits between Jupiter and Neptune (5.5 au < a < 30.1 au)",
    },
    "TNO": {
        "name": "TransNeptunian Object",
        "type": "",
        "description": "Objects with orbits outside Neptune (a > 30.1 au)",
    },
    "PAA": {
        "name": "Parabolic “Asteroid”",
        "type": "",
        "description": "“Asteroids” (objects other than comets) on parabolic orbits (e = 1.0)",
    },
    "HYA": {
        "name": "Hyperbolic “Asteroid”",
        "type": "",
        "description": "“Asteroids” (objects other than comets) on hyperbolic orbits (e > 1.0)",
    },
    "ETc": {
        "name": "Encke-type Comet",
        "type": "comet",
        "description": "Encke-type comet, as defined by Levison and Duncan (Tj > 3; a < aJ)",
    },
    "JFc": {
        "name": "Jupiter-family Comet",
        "type": "comet",
        "description": "Jupiter-family comet, as defined by Levison and Duncan (2 < Tj < 3)",
    },
    "JFC": {
        "name": "Jupiter-family Comet*",
        "type": "comet",
        "description": "Jupiter-family comet, classical definition (P < 20 y)",
    },
    "CTc": {
        "name": "Chiron-type Comet",
        "type": "comet",
        "description": "Chiron-type comet, as defined by Levison and Duncan (Tj > 3; a > aJ)",
    },
    "HTC": {
        "name": "Halley-type Comet*",
        "type": "comet",
        "description": "Halley-type comet, classical definition (20 y < P < 200 y)",
    },
    "PAR": {
        "name": "Parabolic Comet",
        "type": "comet",
        "description": "Comets on parabolic orbits (e = 1.0)",
    },
    "HYP": {
        "name": "Hyperbolic Comet",
        "type": "comet",
        "description": "Comets on hyperbolic orbits (e > 1.0)",
    },
    "COM": {
        "name": "Comet",
        "type": "comet",
        "description": "Comet orbit not matching any defined orbit class",
    },
}

####################################################
# Command line options
####################################################
opts = dict()
for arg in sys.argv[1:]:
    if "=" in arg:
        key, val = arg.split("=")
        opts[key] = val


def get_ipython():
    class foo:
        def run_line_magic(self, *args):
            pass

    return foo


def check_opts(key, default=None):
    if key in opts.keys():
        return opts[key]
    else:
        if default is not None:
            return default
        else:
            return None


def dict2json(dictionary):
    output = "{"
    for key, value in dictionary.items():
        if isinstance(value, str):
            output += f'"{key}":"{value}",'
        elif isinstance(value, dict):
            output += f'"{key}":{dict2json(value)},'
        else:
            output += f'"{key}":{value},'
    output = output.strip(",")
    output += "}"
    return output


def dec2sex(dec, sep=None, day=False, secfmt="%02.3f"):
    if day:
        fac = 24
    else:
        fac = 60
    H = np.floor(dec)
    mm = (dec - H) * fac
    M = np.floor(mm)
    ss = (mm - M) * 60
    S = np.floor(ss)
    if np.abs(ss - 60) < 1e-4:
        M += 1
        ss = 0

    if not sep is None:
        fmt = "%%02d%%s%%02d%%s%s" % secfmt
        return fmt % (int(H), sep[0], int(M), sep[1], ss)
    return H, M, ss


def sex2dec(sex, sep=":"):
    if sep == ":":
        sex = [float(s) for s in sex.split(":")]
    dec = sex[0] + sex[1] / 60.0 + sex[2] / 3600.0
    return dec


####################################################
# Core routines
####################################################
def get_mus_cov(orbit):
    Cov = np.array(orbit["orbit"]["covariance"]["data"], dtype=float)
    Cov_label = orbit["orbit"]["covariance"]["labels"]
    t = float(orbit["orbit"]["epoch"])
    nlen = len(orbit["orbit"]["elements"])
    elements = dict()
    for i in range(nlen):
        element = orbit["orbit"]["elements"][i]
        elements[element["name"]] = dict()
        for prop in element.keys():
            try:
                elements[element["name"]][prop] = float(element[prop])
            except:
                pass
    mus_name = ["e", "q", "tp", "om", "w", "i"]
    mus = [elements[name]["value"] for name in mus_name]

    # Check and filter covariance matrix if it contains extra parameters (e.g. A1, A2)
    # This logic is duplicated from get_latest_orbit to ensure consistency
    if Cov.shape[0] > 6:
        # Mapping from our means order to JPL labels
        standard_labels = ["e", "q", "tp", "node", "peri", "i"]

        # Find indices
        indices = []
        for lbl in standard_labels:
            try:
                indices.append(Cov_label.index(lbl))
            except ValueError:
                # Fallback: try direct name match if 'node'/'peri' not used
                mapping = {"node": "om", "peri": "w"}
                alt_lbl = mapping.get(lbl, lbl)
                if alt_lbl in Cov_label:
                    indices.append(Cov_label.index(alt_lbl))

        if len(indices) == 6:
            Cov = Cov[np.ix_(indices, indices)]
        else:
            # Fallback to simple slicing if matching fails (assuming standard order)
            Cov = Cov[:6, :6]

    return mus, Cov, Cov_label


def get_latest_orbit(id=None, verbose=True):
    """Get the latest orbital elements for an asteroid from JPL's Small-Body Database Browser

    Ejemplo:
        >>> mus,cov = get_latest_orbit(id='2024yr4',verbose=True)

    Notas:
      - Based on the code originally developed by Leonard Gómez, Astronomía UdeA (2022)
    """
    url = f"https://ssd-api.jpl.nasa.gov/sbdb.api?sstr={id}&cov=mat&full-prec=true"
    if verbose:
        print(f"Downloading data about {id.upper()} from {url}...")

    html = request.urlopen(url)
    html_text = html.read().decode()

    # Read data from HTML
    json_data = json.loads(html_text)
    orbit_id = json_data["object"]["orbit_id"]
    print(f"Orbit id: {orbit_id}")

    # Get elements
    t0 = float(json_data["orbit"]["epoch"])
    if verbose:
        print(f"Elements epoch (JDTDB): {t0}")

    if verbose:
        print(f"Extracting covariance matrix...")
    Cov = np.array(json_data["orbit"]["covariance"]["data"], dtype=float)
    Cov_label = json_data["orbit"]["covariance"]["labels"]
    t = float(json_data["orbit"]["epoch"])

    if verbose:
        print(f"Extracting element values and its errors...")
    nlen = len(json_data["orbit"]["elements"])
    elnames = []
    elements = dict()
    for i in range(nlen):
        element = json_data["orbit"]["elements"][i]
        elements[element["name"]] = dict()
        for prop in element.keys():
            try:
                elements[element["name"]][prop] = float(element[prop])
            except:
                pass

    for elname in elements.keys():
        element = elements[elname]
        print(f"Element {elname} = {element['value']:.7e} +/- {element['sigma']:.7e}")

    means = [
        elements["e"]["value"],
        elements["q"]["value"],
        elements["tp"]["value"],
        elements["om"]["value"],
        elements["w"]["value"],
        elements["i"]["value"],
    ]
    if verbose:
        print(f"Order of the downloaded orbital elements: {Cov_label}")

    # Check and filter covariance matrix if it contains extra parameters (e.g. A1, A2)
    if Cov.shape[0] > 6:
        if verbose:
            print(f"Filtering covariance matrix from {Cov.shape} to (6, 6)")

        # Mapping from our means order to JPL labels
        # means = [e, q, tp, om, w, i]
        # JPL labels typically: ['e', 'q', 'tp', 'node', 'peri', 'i', ...]
        standard_labels = ["e", "q", "tp", "node", "peri", "i"]

        # Find indices
        indices = []
        for lbl in standard_labels:
            try:
                indices.append(Cov_label.index(lbl))
            except ValueError:
                # Fallback: try direct name match if 'node'/'peri' not used
                mapping = {"node": "om", "peri": "w"}
                alt_lbl = mapping.get(lbl, lbl)
                if alt_lbl in Cov_label:
                    indices.append(Cov_label.index(alt_lbl))
                else:
                    if verbose:
                        print(
                            f"Warning: Could not find label {lbl} in covariance matrix"
                        )

        if len(indices) == 6:
            Cov = Cov[np.ix_(indices, indices)]
        else:
            # Fallback to simple slicing if matching fails (assuming standard order)
            if verbose:
                print(
                    "Warning: Could not match all labels, falling back to first 6 elements"
                )
            Cov = Cov[:6, :6]

    return json_data, t0, means, Cov


def free_fall(t, y, mu=1, R=1):
    r = y[:3]
    v = y[3:]
    drdt = v
    dvdt = -r * mu / np.linalg.norm(r) ** 3
    return np.concatenate([drdt, dvdt])


def hit_planet(t, y, mu=1, R=1):
    return np.linalg.norm(y[:3]) - R


hit_planet.terminal = True


def impact_on_planet(X0, mu, R):

    # Computing maximuma free-fall time
    r = np.linalg.norm(X0[:3])
    v = np.linalg.norm(X0[3:])
    a = 1 / (2 / r - v**2 / mu)  # vis viva equation
    tmax = 2 * np.pi * np.sqrt(abs(a) ** 3 / mu)  # orbital period proxy

    # Solve the free-fall problem
    solution = solve_ivp(
        free_fall,
        [0, tmax],
        X0,
        events=hit_planet,
        args=(mu, R),
        rtol=1e-12,
        atol=1e-12,
    )
    X_solution = solution.y.T

    # Verify if there were an impact
    if len(solution.t_events[0]) > 0:
        t_impact = solution.t_events[0][0]
        X_impact = solution.y_events[0][0]
    else:
        t_impact = None
        X_impact = None

    return t_impact, X_impact, X_solution


####################################################
# Common routines
####################################################
def load_json(filename):
    with open(filename) as json_file:
        print("Loading json data from", filename)
        obj = json.load(json_file)
    return obj


def save_json(obj, filename):
    with open(filename, "w") as json_file:
        print("Saving object information to", filename)
        json.dump(obj, json_file)


def convert_date_jd(date):
    date_parts = date.split("-")
    year_month = "-".join(date_parts[:2])
    day_frac = date_parts[2]
    time = Time(
        datetime.strptime(year_month, "%Y-%m") + timedelta(days=float(day_frac) - 1),
        scale="utc",
    )
    return time.isot, time.jd


def load_orbit(file_orbit):
    orbit = np.loadtxt(file_orbit)
    jd_0 = orbit[0, 0]
    mus = orbit[1]
    cov = orbit[2:]
    return jd_0, mus, cov


def read_orbit(obj_id, orbit_id, data_dir=data_dir):
    obj_suf = f"obj_{obj_id}-orbit_{orbit_id}"
    data_dir = f"{data_dir}/{obj_id}"
    orbit = load_json(f"{data_dir}/orbit-{obj_suf}.json")
    return orbit


def sphere_surface(R):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = R * np.outer(np.cos(u), np.sin(v))
    y_sphere = R * np.outer(np.sin(u), np.sin(v))
    z_sphere = R * np.outer(np.ones(np.size(u)), np.cos(v))
    return x_sphere, y_sphere, z_sphere


####################################################
# Logging
####################################################
class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


####################################################
# NEO database
####################################################
NEO_DB_DF_FILE = f"neo_db.csv.gz"
NEO_DB_JSON_FILE = f"neo_db.json.gz"


def update_neo_db(dir=data_dir, force=False):
    dir += "/"
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist")

    if os.path.exists(dir + NEO_DB_DF_FILE) and not force:
        print(f"NEO database already exists in {dir}")
        return

    # API URL
    base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"

    # Properties
    identification_prop = [
        "spkid",  # ID  of an object in JPL SBDB
        "pdes",  # Principal description
        "full_name",  # Full name, eg. 433 Eros (A898 PA)
        "kind",  # a: asteroid, c: comet, +n: numbered, +u: unnumbered
        "class",  # See: https://ssd-api.jpl.nasa.gov/doc/sbdb_filter.html
    ]
    orbital_prop = [
        # Time and reference frame
        "equinox",
        "epoch",
        # Orbital elements
        "e",
        "a",
        "q",
        "i",
        "om",
        "w",
        "tp",
        "ma",
    ]
    observational_prop = [
        # Brightness
        "H",
        "G",  # Absolute magnitude and slope parameter
        # Color
        "BV",
        "UB",
        "IR",
        # Spectral type: Tholen and SMASSII respectively
        "spec_T",
        "spec_B",
    ]
    physical_prop = [
        # Estimated albedo
        "albedo",
        # Diameter and tri(or bi)-axial body dimensions
        "diameter",
        "extent",
        "diameter_sigma",  # km
        # Gravitational parameter in km3/s2
        "GM",
        # Average density
        "density",  # g/cm3
        # Synodic rotational period
        "rot_per",  # hours
        # Non-gravitational parameters
        "A1",
        "A2",
        "A3",  # Adimensional
    ]
    fields = ",".join(
        identification_prop + observational_prop + orbital_prop + physical_prop
    )

    # Download the database
    base_url = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
    params = {
        "fields": fields,
        "sb-group": "neo",
        # 'sb-class':'IEO' # For testing purposes
    }
    # Make the API request
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        neo_db = response.json()
        print(f"Number of objects recovered: {neo_db['count']}")
    else:
        raise AssertionError(f"Error: {response.status_code}")

    # Save the JSON gzipped file
    json_file = dir + NEO_DB_JSON_FILE
    with gzip.open(json_file, "wt", encoding="utf-8") as f:
        json.dump(neo_db, f)
    print(f"neo_db with {neo_db['count']} objects has been saved to {json_file}")

    # Save the DF gzipped file
    data = neo_db["data"]
    fields = neo_db["fields"]
    df = pd.DataFrame(data, columns=fields)
    df.set_index("spkid", inplace=True)

    df_file = dir + NEO_DB_DF_FILE
    df.to_csv(df_file, index=True, compression="gzip")

    # Backup file
    # df_bak_file = dir + f"{NEO_DB_DF_FILE.split('.')[0]}_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv.gz"
    # df_bak_file = dir + f"{NEO_DB_DF_FILE.split('.')[0]}_{datetime.now().strftime('%Y_%m_%d')}.csv.gz"
    df_bak_file = dir + f"{NEO_DB_DF_FILE.split('.')[0]}_prev.csv.gz"
    if not os.path.exists(df_bak_file):
        df.to_csv(df_bak_file, index=False, compression="gzip")
        print(f"File saved as {df_bak_file}")

    print(f"File saved as {df_file}")


def read_neo_db(dir=data_dir, type="df", verbose=True):
    dir += "/"
    if not os.path.exists(dir):
        raise FileNotFoundError(f"Directory {dir} does not exist")

    if type == "json":
        # Read the JSON gzipped file
        json_file = dir + NEO_DB_JSON_FILE
        with gzip.open(json_file, "rt", encoding="utf-8") as f:
            neo_db = json.load(f)
        if verbose:
            print(f"Loaded neo_db with {neo_db['count']} objects from {json_file}")
    elif type == "df":
        # Read the DataFrame gzipped file
        df_file = dir + NEO_DB_DF_FILE
        neo_db = pd.read_csv(df_file, index_col=0, compression="gzip", low_memory=False)
        if verbose:
            print(f"Loaded neo_db with {len(neo_db)} objects from {df_file}")
    else:
        raise ValueError(f"Unknown type: {type}")

    return neo_db


C_std = 1329.22  # km
a_std = 0.2


def Hmag2diameter(H, albedo=1.0, C=C_std, a=a_std):
    """Convert from absolute magnitude H to diameter in km

    Args:
        H: float or nd.array (size=n_h, units: mag)
            Absolute magnitude.

        albedo: float or iterable (tuple, list, nd.array) (size=n_a, units: admin)
            Geometric albedo.

    Optional arguments:
        C: float [default = C_std] (units: km)
            Size parameter in the law: D = C 10^(-aH)/sqrt(p)

        a: float [default = a_str] (units: 1/mag)
            Exponent parameter in the law: D = C 10^(-aH)/sqrt(p)

    Return:
        diameters: float or nd.array (n_h x n_a, units: km)
            Diameter(s)

    Examples:
        >>> Hmag2diameter(30,albedo=[0.1,1])
        >>> Hmag2diameter(np.array([10,15,20]),albedo=[0.1,1])
    """
    if isinstance(albedo, (tuple, list, np.ndarray)):
        albedos = np.array(albedo)
        diameters = np.array(
            [C * 10 ** (-a * H) / np.sqrt(albedo) for albedo in albedos]
        )
    else:
        diameters = C * 10 ** (-a * H) / np.sqrt(albedo)
    return diameters


def diameter2Hmag(diameter, albedo=1.0, C=C_std, a=a_std):
    """Convert from diameter in km to absolute magnitude H

    Args:
        diameters: float or nd.array (size=n_h, units: km)
            Diameter(s)

        albedo: float or iterable (tuple, list, nd.array) (size=n_a, units: admin)
            Geometric albedo.

    Optional arguments:
        C: float [default = C_std] (units: km)
            Size parameter in the law: D = C 10^(-aH)/sqrt(p)

        a: float [default = a_str] (units: 1/mag)
            Exponent parameter in the law: D = C 10^(-aH)/sqrt(p)

    Return:
        H: float or nd.array (n_h x n_a, units: mag)
            Absolute magnitude.

    Examples:
        >>> diameter2Hmag(0.5,albedo=[0.1,1])
    """
    if isinstance(albedo, (tuple, list, np.ndarray)):
        albedos = np.array(albedo)
        Hmags = np.array(
            [-np.log10(diameter * np.sqrt(albedo) / C) / a for albedo in albedos]
        )
    else:
        Hmags = -np.log10(diameter * np.sqrt(albedo) / C) / a
    return Hmags


def linear_fit_Hmag(Hs, Fs, F_threshold=10, H_max=None, tail=[]):

    # Thresholds
    Hmin = Hs[Fs > F_threshold][0]
    Hmax = H_max
    cond = (Hs >= Hmin) & (Hs <= Hmax)

    Hs = Hs[cond]
    Fs = Fs[cond]

    if len(tail) > 0:
        Hs = np.concatenate((Hs, tail[0]))
        Fs = np.concatenate((Fs, tail[1]))

    # Linear fit
    Hs_corte = Hs.reshape(-1, 1)[:-1]
    Fs_corte = Fs.reshape(-1, 1)[:-1]
    model = LinearRegression()
    model.fit(Hs_corte, np.log10(Fs_corte))

    # Coefficients
    a = model.coef_[0][0]
    b = model.intercept_[0]

    # Function
    FH_function = lambda H: 10 ** (a * H + b)

    chi2 = chi2_cumsum(FH_function, Hs, Fs)

    return Hs, Fs, a, b, FH_function, chi2


def chi2_cumsum(FH_function, Hs, Fs, tail=[]):

    # Add tail
    if len(tail) > 0:
        Hs = np.concatenate((Hs, tail[0]))
        Fs = np.concatenate((Fs, tail[1]))

    # Compute chi-square
    chi2 = ((np.log10(Fs) - np.log10(FH_function(Hs))) ** 2).sum() / np.log10(Fs.sum())

    return chi2


def pympact_watermark(ax, enlarge=1, alpha=0.5):
    """Add a water mark to a 2d or 3d plot.

    Parameters:

        ax: Class axes:
            Axe where the pryngles mark will be placed.
    """
    # Get the height of axe
    axh = (
        ax.get_window_extent()
        .transformed(ax.get_figure().dpi_scale_trans.inverted())
        .height
    )
    fig_factor = axh / 4

    # Options of the water mark
    args = dict(
        rotation=270,
        ha="left",
        va="top",
        transform=ax.transAxes,
        color="pink",
        fontsize=8 * fig_factor * enlarge,
        zorder=100,
        alpha=alpha,
    )

    # Text of the water mark
    mark = f"[PY]mpact Team"

    # Choose the according to the fact it is a 2d or 3d plot
    try:
        ax.add_collection3d
        plt_text = ax.text2D
    except:
        plt_text = ax.text

    text = plt_text(1, 1, mark, **args)
    return text


class UtilPlot(object):
    """
    This abstract class contains useful methods for the module plot
    """

    def points_inside(
        points: List[List[float]], shp_path: str, plot_map: bool = False
    ) -> gpd.GeoDataFrame:  # -> List[bool]:
        """
        Verifica si uno o varios puntos se encuentran dentro del borde definido en el
        shapefile.

        Parameters
        ----------
        points : List[List[float]]
            Lista de pares [lon, lat], por ejemplo: [[lon1, lat1], [lon2, lat2], ...].
        shp_path : str
            Ruta al shapefile que contiene el borde. Este debe tener el polígono en
            formato WGS84.
        plot_map : bool, optional
            Si True, se muestra un mapa con los puntos y el borde.

        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame con una columna 'inside' que indica True si el punto está
            dentro y False si está fuera.


        Examples:
        --------
        shp_path = "Colombia/borde_colombia.shp"  # Ruta al shapefile del borde
        test_points = [(-72,2.5), (-73, 4), (-74, 5), (-80,0)]
        results = points_inside(test_points, shp_path, plot_map=True)
        """
        # Cargar el shapefile
        border_gdf = gpd.read_file(shp_path)
        border_union = border_gdf.unary_union

        # Crear el GeoDataFrame con los puntos
        gdf_points = gpd.GeoDataFrame(
            {"geometry": [Point(lon, lat) for lon, lat in points]}, crs=border_gdf.crs
        )

        # Evaluar si cada punto se encuentra dentro del polígono
        gdf_points["inside"] = gdf_points.geometry.apply(
            lambda p: border_union.contains(p)
        )

        if plot_map:
            fig, ax = plt.subplots()
            border_gdf.plot(ax=ax, color="lightgray")

            # Extraer subconjuntos
            inside_points = gdf_points[gdf_points.inside]
            outside_points = gdf_points[~gdf_points.inside]

            if not inside_points.empty:
                inside_points.plot(ax=ax, color="blue", marker="o", label="Inside")
            if not outside_points.empty:
                outside_points.plot(ax=ax, color="red", marker="x", label="Outside")

            plt.legend()
            plt.show()

        return gdf_points

        ## Si se desea es una lista de booleanos
        # return gdf_points.inside.to_list()

    def mantisaExp(x):
        """
        Calculate the mantisa and exponent of a number.

        Parameters:
            x: number, float.

        Return:
            man: mantisa, float
            exp: exponent, float.

        Examples:
            m,e=mantisaExp(234.5), returns m=2.345, e=2
            m,e=mantisaExp(-0.000023213), return m=-2.3213, e=-5
        """
        xa = np.abs(x)
        s = np.sign(x)
        try:
            exp = int(np.floor(np.log10(xa)))
            man = s * xa / 10 ** (exp)
        except OverflowError as e:
            man = exp = 0
        return man, exp
