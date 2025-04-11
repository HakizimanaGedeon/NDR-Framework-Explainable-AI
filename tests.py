import unittest
from ndr.framework import NDRFramework

class TestNDRFramework(unittest.TestCase):

    def setUp(self):
        self.ndr = NDRFramework()
        self.ndr.add_law("Law1", "If Mean Radius is high, the likelihood of cancer increases.")
        self.ndr.add_antecedent("Antecedent1", "Mean Radius > 15")

    def test_generate_explanation(self):
        features = {'mean_radius': 16, 'smoothness': 0.2, 'mean_texture': 22}
        explanation = self.ndr.generate_explanation(features)
        self.assertIn("Law1", explanation)
        self.assertIn("Likelihood of cancer", explanation)

if __name__ == '__main__':
    unittest.main()
