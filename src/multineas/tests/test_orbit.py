import os
import numpy as np
import pytest
from multineas.orbit import (
    AU_KM,
    Orbit,
    MU_SUN_AU,
    compute_jacobian_x_to_e,
    compute_orbital_period,
    geo_to_eclip,
    geo_to_rectangular,
    get_orbit_impactor,
    get_pre_impact_orbit,
    get_velocity_ecliptic,
    state_au_per_day_to_km_per_s,
    state_km_per_s_to_au_per_day,
    transformation_e_to_x,
    transformation_x_to_e,
)

# Shared test data (no fixtures, to avoid injection issues)
SAMPLE_ORBITAL_ELEMENTS = dict(
    q=1.0,
    e=0.2,
    i=np.deg2rad(40),
    Omega=np.deg2rad(10),
    w=np.deg2rad(60),
    M=np.deg2rad(25),
)

SAMPLE_STATE_VECTOR = dict(
    x=1.0,
    y=0.2,
    z=0.1,
    vx=0.0,
    vy=6.28,
    vz=0.0,
)


# Software unit tests
def test_transformation_e_to_x_output_shape():
    r, v = transformation_e_to_x(mu=MU_SUN_AU, **SAMPLE_ORBITAL_ELEMENTS)
    assert np.asarray(r).shape == (3,)
    assert np.asarray(v).shape == (3,)


def test_transformation_x_to_e_output_shape():
    elements = transformation_x_to_e(mu=MU_SUN_AU, **SAMPLE_STATE_VECTOR)
    elements = np.asarray(elements)
    assert elements.shape == (7,)


def test_transformation_x_to_e_eccentricity_output_values():

    with pytest.raises(ValueError):
        transformation_e_to_x(
            q=1.0, e=-0.1, i=0.0, Omega=0.0, w=0.0, M=0.0, mu=MU_SUN_AU
        )


def test_state_unit_conversion_roundtrip():
    """State conversion km/km/s <-> AU/AU/day preserves shape and roundtrips."""
    pos_km = np.array([1e8, 2e7, 5e6])
    vel_km_s = np.array([10.0, -5.0, 1.0])
    pos_au, vel_au_day = state_km_per_s_to_au_per_day(pos_km, vel_km_s)
    assert pos_au.shape == (3,)
    assert vel_au_day.shape == (3,)
    pos2, vel2 = state_au_per_day_to_km_per_s(pos_au, vel_au_day)
    np.testing.assert_allclose(pos2, pos_km, rtol=1e-12)
    np.testing.assert_allclose(vel2, vel_km_s, rtol=1e-12)


def test_compute_orbital_period_output():
    """Orbital period is positive and ~365 days for a = 1 AU."""
    period = compute_orbital_period(1.0, MU_SUN_AU)
    assert isinstance(period, (float, np.floating))
    assert period > 0
    assert 0.9 < period < 1.2  # Earth ~1 year


def test_compute_jacobian_x_to_e_shape():
    """Jacobian of state -> elements has shape (6, 6)."""
    a = 1.0
    e = 0.2
    i = np.deg2rad(10)
    Omega = np.deg2rad(20)
    w = np.deg2rad(30)
    M = np.deg2rad(15)
    J = compute_jacobian_x_to_e(a, e, i, Omega, w, M, MU_SUN_AU)
    assert J.shape == (6, 6)


def test_orbit_from_state_vector_shapes():
    """Orbit built from state_vector has .state_vector (6,) and .elements (6 or 7)."""
    vec = np.array(
        [
            SAMPLE_STATE_VECTOR["x"],
            SAMPLE_STATE_VECTOR["y"],
            SAMPLE_STATE_VECTOR["z"],
            SAMPLE_STATE_VECTOR["vx"],
            SAMPLE_STATE_VECTOR["vy"],
            SAMPLE_STATE_VECTOR["vz"],
        ]
    )
    orb = Orbit(mu=MU_SUN_AU, state_vector=vec)
    assert orb.state_vector.shape == (6,)
    assert len(orb.elements) in (6, 7)


def test_orbit_validation_exactly_one_input():
    """Orbit raises if both state_vector and elements are provided."""
    vec = np.array([1.0, 0.0, 0.0, 0.0, 6.28, 0.0])
    el = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="exactly one"):
        Orbit(mu=MU_SUN_AU, state_vector=vec, elements=el)


# -----------------------------------------------------------------------------
# Scientific test: specific energy
# -----------------------------------------------------------------------------
def test_specific_energy():
    r, v = transformation_e_to_x(mu=MU_SUN_AU, **SAMPLE_ORBITAL_ELEMENTS)
    r = np.asarray(r)
    v = np.asarray(v)
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)
    q = SAMPLE_ORBITAL_ELEMENTS["q"]
    e = SAMPLE_ORBITAL_ELEMENTS["e"]
    a = q / (1 - e)

    specific_energy = -0.5 * MU_SUN_AU / a
    expected_specific_energy = v_norm**2 / 2 - MU_SUN_AU / r_norm

    assert np.isclose(specific_energy, expected_specific_energy, rtol=1e-10, atol=1e-10)


# -----------------------------------------------------------------------------
# Scientific test: Chelyabinsk impact orbit vs literature
# -----------------------------------------------------------------------------
# Chelyabinsk impact data (2013-02-15)
CHELYABINSK_IMPACT_SITE = [
    60.09285,
    55.07815,
    178.4583,
]  # lon [deg], lat [deg], alt [km]
CHELYABINSK_IMPACT_VELOCITY = [12.8, -13.3, -2.4]  # vx, vy, vz [km/s] Earth-fixed
CHELYABINSK_IMPACT_DATE = "2013-02-15 03:20:33"  # UTC

# Literature ranges from Zuluaga & Ferrin (arXiv), Borovicka et al. (IAU), Zuluaga/Ferrin/Geens (arXiv)
# Columns: Q/aphelion, q/perihelion, a, e, i [deg], Omega [deg], omega [deg]
LIT_Q_AU = (0.71, 0.83)  # perihelion
LIT_A_AU = (1.26, 1.75)  # semi-major axis
LIT_E = (0.44, 0.57)  # eccentricity
LIT_I_DEG = (2.98, 4.5)  # inclination
LIT_OMEGA_DEG = (326.4, 326.8)  # longitude of ascending node
LIT_OMEGA_ARG_DEG = (95.5, 121)  # argument of periapsis omega


# Kernel path: src/multineas/data/kernels (relative to repo root)
_KERNEL_DIR = os.path.join(os.path.dirname(__file__), "kernels")


@pytest.mark.skipif(
    not os.path.exists(os.path.join(_KERNEL_DIR, "naif0012.tls")),
    reason="SPICE kernels not found (src/multineas/data/kernels/)",
)
def test_chelyabinsk_impact_orbit_vs_literature():
    """
    Scientific test: compute impactor orbit from Chelyabinsk bolide data and
    compare orbital elements with literature (Zuluaga & Ferrin, Borovicka et al.,
    Zuluaga/Ferrin/Geens). Loads SPICE kernels from src/multineas/data/kernels/.
    """
    import spiceypy as spy

    kernel_dir = _KERNEL_DIR
    kernels = [
        os.path.join(kernel_dir, "naif0012.tls"),
        os.path.join(kernel_dir, "pck00010.tpc"),
        os.path.join(kernel_dir, "earth_fixed.tf"),
        os.path.join(kernel_dir, "earth_720101_230601.bpc"),
        os.path.join(kernel_dir, "earth_latest_high_prec.bpc"),
    ]
    for k in kernels:
        if not os.path.exists(k):
            pytest.skip(f"SPICE kernel not found: {k}")
    spy.furnsh(kernels)

    try:
        lon, lat, alt = CHELYABINSK_IMPACT_SITE
        vx, vy, vz = CHELYABINSK_IMPACT_VELOCITY
        date = CHELYABINSK_IMPACT_DATE

        # get_orbit_impactor returns (q, e, i, Omega, w, M, a) in AU and radians
        elements = get_orbit_impactor(lon, lat, alt, vx, vy, vz, date, MU_SUN_AU)
        elements = np.asarray(elements)
        assert elements.shape == (7,), "get_orbit_impactor should return 7 elements"

        q, e, a = elements[0], elements[1], elements[6]
        i_rad, Omega_rad, w_rad = elements[2], elements[3], elements[4]
        i_deg = np.rad2deg(i_rad)
        Omega_deg = np.rad2deg(Omega_rad)
        w_deg = np.rad2deg(w_rad)

        # Compare with literature ranges
        assert LIT_Q_AU[0] <= q <= LIT_Q_AU[1], (
            f"q={q:.3f} AU outside literature {LIT_Q_AU}"
        )
        assert LIT_A_AU[0] <= a <= LIT_A_AU[1], (
            f"a={a:.3f} AU outside literature {LIT_A_AU}"
        )
        assert LIT_E[0] <= e <= LIT_E[1], f"e={e:.3f} outside literature {LIT_E}"
        assert LIT_I_DEG[0] <= i_deg <= LIT_I_DEG[1], (
            f"i={i_deg:.2f}° outside literature {LIT_I_DEG}"
        )
        assert LIT_OMEGA_DEG[0] <= Omega_deg <= LIT_OMEGA_DEG[1], (
            f"Omega={Omega_deg:.2f}° outside literature {LIT_OMEGA_DEG}"
        )
        assert LIT_OMEGA_ARG_DEG[0] <= w_deg <= LIT_OMEGA_ARG_DEG[1], (
            f"omega={w_deg:.2f}° outside literature {LIT_OMEGA_ARG_DEG}"
        )
    finally:
        for k in kernels:
            try:
                spy.unload(k)
            except Exception:
                pass
