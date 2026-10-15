import unittest

import numpy as np
from sklearn.preprocessing import LabelEncoder

from app.backend.model_service import RiskModelService


class ConstantClassifier:
    def predict(self, matrix):
        return np.zeros(len(matrix), dtype=int)

    def predict_proba(self, matrix):
        return np.tile([1.0, 0.0], (len(matrix), 1))


def make_service():
    service = RiskModelService.__new__(RiskModelService)
    service.required_features = [
        "AGENCE",
        "SECTEUR ACTIVITE",
        "TYPE DE CREDIT",
        "ANNEE OUVERTURE",
        "MOIS OUVERTURE",
        "Age",
        "Nombre de Crédit antérieur",
        "SOLDE_CREDIT",
        "MONTANT_DEBOURSE",
    ]
    service.encoders = {
        "AGENCE": LabelEncoder().fit(["AGENCE_A", "AGENCE_B"]),
        "SECTEUR ACTIVITE": LabelEncoder().fit(["SECTEUR_A", "SECTEUR_B"]),
        "TYPE DE CREDIT": LabelEncoder().fit([" TYPE_A", "TYPE_B"]),
    }
    service.median_credits = 3.0
    service.model = ConstantClassifier()
    service.mode = "production"
    service.metrics = None
    return service


class RiskModelServiceTests(unittest.TestCase):
    def test_history_fills_missing_credit_count_and_skips_unknown_outcomes(self):
        service = make_service()
        base = {
            "AGENCE": "AGENCE_A",
            "SECTEUR ACTIVITE": "SECTEUR_A",
            "TYPE DE CREDIT": " TYPE_A",
            "ANNEE OUVERTURE": "2023",
            "MOIS OUVERTURE": "5",
            "Age": "",
            "Nombre de Crédit antérieur": "",
            "SOLDE_CREDIT": "NaN",
            "MONTANT_DEBOURSE": "300000",
        }

        result = service.evaluate_history([
            {**base, "RETARD": "0 JOUR"},
            {**base, "RETARD": ""},
            {**base, "RETARD": "30 JOUR", "Nombre de Crédit antérieur": None},
        ])

        self.assertEqual(result["records"], 2)
        self.assertEqual(result["confusion_matrix"], [[1, 0], [1, 0]])

    def test_encoder_labels_are_preserved_and_unknown_categories_rejected(self):
        service = make_service()
        record = {
            "AGENCE": "AGENCE_A",
            "SECTEUR ACTIVITE": "SECTEUR_A",
            "TYPE DE CREDIT": " TYPE_A",
        }
        self.assertEqual(service._transform_record(record).shape, (1, 9))

        with self.assertRaisesRegex(ValueError, "Catégorie inconnue"):
            service._transform_record({**record, "TYPE DE CREDIT": "UNKNOWN"})


if __name__ == "__main__":
    unittest.main()