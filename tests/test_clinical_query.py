import unittest

from clinical_query import build_clinical_query, build_risk_query


from risk_config import RISK_THRESHOLD


class ClinicalQueryTest(unittest.TestCase):
    def test_build_clinical_query_includes_prediction_and_threshold(self):
        query = build_clinical_query(
            0.42,
            {"label_en": "High Risk"},
        )

        self.assertIn("0.420000", query)
        self.assertIn(str(RISK_THRESHOLD), query)
        self.assertIn("High Risk", query)
        self.assertIn("early invasive", query)

    def test_build_risk_query_remains_compatible(self):
        query = build_risk_query({"label_en": "Low Risk"})
        self.assertIn("low risk", query)
        self.assertIn("follow-up", query)


if __name__ == "__main__":
    unittest.main()
