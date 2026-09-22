import unittest

import numpy as np

from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu, normalise_physical


class SynthesisTests(unittest.TestCase):
    def test_overlapping_discharges_add(self):
        muaps = np.ones((1, 1, 1, 96))
        emg = generate_emg_mu(muaps, [0, 64], 160)
        np.testing.assert_array_equal(emg[0, 0, 64:96], np.full(32, 2.0))

    def test_physical_normalisation_round_trip(self):
        from BioMime.utils.params import coeff_a, coeff_b

        physical = np.array([3.63636364])
        normalized = normalise_physical(physical, "cv")
        np.testing.assert_allclose(normalized / coeff_b["cv"] - coeff_a["cv"], physical)


if __name__ == "__main__":
    unittest.main()
