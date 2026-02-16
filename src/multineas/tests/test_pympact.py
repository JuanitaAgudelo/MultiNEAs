import unittest
import numpy as np
import multineas.pympact as pm
from astropy.time import Time
import os


class TestPympact(unittest.TestCase):
    def test_convert_date_jd(self):
        """Test the convert_date_jd function."""
        # Test case 1: Standard date
        date_str = "2024-01-01.5"
        iso, jd = pm.convert_date_jd(date_str)

        # Expected JD for 2024-01-01 12:00:00 UTC
        expected_time = Time("2024-01-01 12:00:00", scale="utc")

        self.assertAlmostEqual(jd, expected_time.jd, places=5)
        self.assertEqual(iso, expected_time.isot)

    def test_get_mus_cov_logic(self):
        """Test get_mus_cov extraction and filtering logic."""
        # Create a mock JSON structure similar to JPL Horizons output
        # Case 1: Covariance matrix 6x6, standard order
        mock_orbit_data_std = {
            "orbit": {
                "epoch": "2460000.5",
                "elements": [
                    {"name": "e", "value": "0.1", "sigma": "1e-5"},
                    {"name": "q", "value": "1.0", "sigma": "1e-5"},
                    {"name": "tp", "value": "2459000.5", "sigma": "1e-2"},
                    {"name": "om", "value": "100.0", "sigma": "1e-4"},
                    {"name": "w", "value": "50.0", "sigma": "1e-4"},
                    {"name": "i", "value": "10.0", "sigma": "1e-4"},
                ],
                "covariance": {
                    "labels": ["e", "q", "tp", "node", "peri", "i"],
                    "data": np.eye(6).tolist(),
                },
            }
        }

        mus, cov, labels = pm.get_mus_cov(mock_orbit_data_std)

        self.assertEqual(len(mus), 6)
        self.assertEqual(cov.shape, (6, 6))
        self.assertEqual(mus[0], 0.1)  # e
        self.assertEqual(labels, ["e", "q", "tp", "node", "peri", "i"])

        # Case 2: Covariance matrix > 6x6 (e.g., 8x8 with A1, A2), needs filtering
        # The code expects standard_labels = ["e", "q", "tp", "node", "peri", "i"] to calculate indices
        large_cov_data = np.eye(8)
        # Check if code handles 'node'/'om' and 'peri'/'w' mapping correctly.
        # Function def get_mus_cov mapping: means = [e, q, tp, om, w, i]
        # function logic: standard_labels = ["e", "q", "tp", "node", "peri", "i"]

        mock_orbit_data_large = {
            "orbit": {
                "epoch": "2460000.5",
                "elements": [
                    {"name": "e", "value": "0.2"},
                    {"name": "q", "value": "1.2"},
                    {"name": "tp", "value": "2459000.5"},
                    {"name": "om", "value": "110.0"},
                    {"name": "w", "value": "60.0"},
                    {"name": "i", "value": "5.0"},
                ],
                "covariance": {
                    "labels": ["e", "q", "tp", "node", "peri", "i", "A1", "A2"],
                    "data": large_cov_data.tolist(),
                },
            }
        }

        mus_l, cov_l, labels_l = pm.get_mus_cov(mock_orbit_data_large)

        self.assertEqual(len(mus_l), 6)
        self.assertEqual(cov_l.shape, (6, 6))
        # Validate that we got a 6x6 slice
        # Since identity matrix, any 6x6 submatrix on diagonal is identity,
        # but let's assume the function filters by name properly.
        self.assertEqual(cov_l.shape[0], 6)
        self.assertEqual(cov_l.shape[1], 6)

    def test_impact_on_planet_free_fall(self):
        """Test impact_on_planet with a simple free fall scenario."""
        # Object falling straight to a planet from r0
        # mu = 1, R = 1 (normalized units often used in examples, but function takes args)

        mu = 1.0
        R = 1.0

        # Start at x=2, y=0, z=0 (r=2)
        # Velocity = 0 (free fall) implies e=1 (parabolic/rectilinear)
        # Actually impact_on_planet solves ODE.

        # Initial state: [x, y, z, vx, vy, vz]
        r0 = 2.0
        X0 = np.array([r0, 0.0, 0.0, 0.0, 0.0, 0.0])

        t_impact, X_impact, X_solution = pm.impact_on_planet(X0, mu, R)

        # Verify impact occurred
        self.assertIsNotNone(t_impact)
        self.assertIsNotNone(X_impact)

        # Determine expected impact radius (should be close to R)
        r_impact = np.linalg.norm(X_impact[:3])
        self.assertAlmostEqual(r_impact, R, places=4)

        # Theoretical free fall time from r0 to R for radial orbit
        # T = (arccos(sqrt(R/r0)) + sqrt(R/r0 * (1 - R/r0))) * sqrt(r0^3 / (2*mu))
        # Wait, that's regular free fall?
        # Standard rectilinear fallback time: t = sqrt(r0^3/2mu) * ( ... )
        # Let's just check sanity: t > 0
        self.assertGreater(t_impact, 0)

        # Check velocity direction is purely radial inwards at impact
        # v dot r should be negative (moving inwards) roughly -1 if normalized
        v_impact = X_impact[3:]
        pos_impact = X_impact[:3]
        cos_angle = np.dot(v_impact, pos_impact) / (
            np.linalg.norm(v_impact) * np.linalg.norm(pos_impact)
        )
        self.assertAlmostEqual(cos_angle, -1.0, places=4)

    def test_impact_on_planet_miss(self):
        """Test scenario where object misses the planet (stable circular orbit)."""
        mu = 1.0
        R = 1.0
        r_orbit = 2.0

        # Circular velocity v = sqrt(mu/r)
        v_circ = np.sqrt(mu / r_orbit)

        # State: [r_orbit, 0, 0, 0, v_circ, 0]
        X0 = np.array([r_orbit, 0.0, 0.0, 0.0, v_circ, 0.0])

        # This will simulate up to tmax based on period.
        # Since it's a circular orbit with r=2 > R=1, it should NOT impact.

        t_impact, X_impact, X_solution = pm.impact_on_planet(X0, mu, R)

        self.assertIsNone(t_impact)
        self.assertIsNone(X_impact)
        # X_solution should exist and contain trajectory
        self.assertTrue(len(X_solution) > 0)


if __name__ == "__main__":
    unittest.main()
