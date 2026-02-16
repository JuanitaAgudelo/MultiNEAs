import unittest
import numpy as np
import multineas.legacyorb as leg
import spiceypy as spy


class TestLegacyOrb(unittest.TestCase):
    def test_numerical_jacobian_polar_cartesian(self):
        """Test numerical Jacobian with Polar <-> Cartesian transformation."""

        # E to X (Polar to Cartesian)
        def EtoX(E):
            r, q = E
            x = r * np.cos(q)
            y = r * np.sin(q)
            X = np.array([x, y])
            return X

        # Value of E
        r = 2
        q = np.pi / 3
        E = [r, q]
        dE = [1e-3, 1e-3]

        # Analytical Jacobian JXoE
        # [[cos(q), -r*sin(q)],
        #  [sin(q),  r*cos(q)]]
        JXoE_analytical = np.array(
            [[np.cos(q), -r * np.sin(q)], [np.sin(q), r * np.cos(q)]]
        )

        # Compute numerical
        X, JXoE_numerical = leg.computeNumericalJacobian(EtoX, E, dE)

        # Check values
        np.testing.assert_allclose(X, [r * np.cos(q), r * np.sin(q)], rtol=1e-5)
        np.testing.assert_allclose(JXoE_numerical, JXoE_analytical, rtol=1e-5)

        # X to E (Cartesian to Polar)
        def XtoE(X_val):
            x, y = X_val
            r_val = (x**2 + y**2) ** 0.5
            q_val = np.arctan2(y, x)
            E_val = np.array([r_val, q_val])
            return E_val

        dX = [1e-3, 1e-3]

        # Compute numerical inverse Jacobian
        E_calc, JEoX_numerical = leg.computeNumericalJacobian(XtoE, X, dX)

        # Verify strict inverse relationship JXoE * JEoX ~ I
        identity_approx = np.dot(JXoE_numerical, JEoX_numerical)
        np.testing.assert_allclose(identity_approx, np.eye(2), atol=1e-4)

    def test_numerical_jacobian_keplerian(self):
        """Test numerical Jacobian with Cartesian -> Keplerian elements using SpiceyPy."""

        def X2E(X_val, mu):
            elts = spy.oscelt(X_val, 0, mu)
            E_val = elts[:6]
            return E_val

        mu = 1
        X = np.array([1, 1, 1, -0.1, -0.1, 1])
        dX = np.array([1e-3] * 6)
        args = dict(mu=mu)

        # We just verify it runs and returns correct shape (6,6)
        E, JEoX = leg.computeNumericalJacobian(X2E, X, dX, **args)

        self.assertEqual(len(E), 6)
        self.assertEqual(JEoX.shape, (6, 6))

        # Basic sanity check on determinant (should not be zero for valid transformation)
        det = np.linalg.det(JEoX)
        self.assertNotAlmostEqual(det, 0)


if __name__ == "__main__":
    unittest.main()
