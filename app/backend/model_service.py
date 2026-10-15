from __future__ import annotations

import csv
import json
import logging
import math
import os
import pickle
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT_PATH = ROOT / "artifacts" / "model_credit_risk_artifacts.pkl"
ROOT_ARTIFACT_PATH = ROOT / "model_credit_risk_artifacts.pkl"
logger = logging.getLogger(__name__)


class RiskModelService:
    def __init__(self, artifact_path: str | Path = ARTIFACT_PATH):
        self.artifact_path = Path(artifact_path)
        self.model = None
        self.encoders = {}
        self.median_credits = None
        self.required_features = []
        self.metrics = None
        self.mode = "demo"
        self._load_model()

    def _load_model(self):
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        for path in dict.fromkeys((self.artifact_path, ROOT_ARTIFACT_PATH)):
            if not path.exists():
                continue
            try:
                with open(path, "rb") as f:
                    artifacts = pickle.load(f)
                self.model = artifacts.get("model")
                self.encoders = artifacts.get("encoders", {})
                self.median_credits = artifacts.get("median_credits")
                self.required_features = artifacts.get("features", [])
                self.metrics = artifacts.get("metrics")
                if self.model is not None and self.required_features:
                    self.artifact_path = path
                    self.mode = "production"
                    return
                raise ValueError("L'artefact doit contenir un modèle et la liste des variables.")
            except Exception as exc:
                logger.warning("Impossible de charger l'artefact modèle %s: %s", path, exc)
                continue

        if os.environ.get("CREDIT_RISK_REQUIRE_ARTIFACT", "0").lower() in {"1", "true", "yes"}:
            raise RuntimeError(
                f"Artefact de production absent ou invalide. Fichier attendu: {self.artifact_path}"
            )
        self._load_demo_model()

    def _load_demo_model(self):
        self.mode = "demo"
        self.required_features = [
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

        data = [
            {
                "AGENCE": "MECYD",
                "SECTEUR ACTIVITE": "CONSTRUCTION",
                "TYPE DE CREDIT": "CONSOMMATION",
                "ANNEE OUVERTURE": 2022,
                "MOIS OUVERTURE": 5,
                "Age": 28,
                "Nombre de Crédit antérieur": 2,
                "SOLDE_CREDIT": 120000,
                "MONTANT_DEBOURSE": 300000,
                "CIBLE_RETARD": 0,
            },
            {
                "AGENCE": "DLA",
                "SECTEUR ACTIVITE": "TIC",
                "TYPE DE CREDIT": "IMMOBILIER",
                "ANNEE OUVERTURE": 2021,
                "MOIS OUVERTURE": 11,
                "Age": 35,
                "Nombre de Crédit antérieur": 4,
                "SOLDE_CREDIT": 260000,
                "MONTANT_DEBOURSE": 700000,
                "CIBLE_RETARD": 1,
            },
            {
                "AGENCE": "MECYD",
                "SECTEUR ACTIVITE": "AGROALIMENTAIRE",
                "TYPE DE CREDIT": "CONSOMMATION",
                "ANNEE OUVERTURE": 2023,
                "MOIS OUVERTURE": 2,
                "Age": 42,
                "Nombre de Crédit antérieur": 1,
                "SOLDE_CREDIT": 95000,
                "MONTANT_DEBOURSE": 180000,
                "CIBLE_RETARD": 0,
            },
            {
                "AGENCE": "DLA",
                "SECTEUR ACTIVITE": "CONSTRUCTION",
                "TYPE DE CREDIT": "IMMOBILIER",
                "ANNEE OUVERTURE": 2020,
                "MOIS OUVERTURE": 8,
                "Age": 47,
                "Nombre de Crédit antérieur": 7,
                "SOLDE_CREDIT": 370000,
                "MONTANT_DEBOURSE": 900000,
                "CIBLE_RETARD": 1,
            },
            {
                "AGENCE": "MECYD",
                "SECTEUR ACTIVITE": "TIC",
                "TYPE DE CREDIT": "CONSOMMATION",
                "ANNEE OUVERTURE": 2024,
                "MOIS OUVERTURE": 1,
                "Age": 31,
                "Nombre de Crédit antérieur": 3,
                "SOLDE_CREDIT": 145000,
                "MONTANT_DEBOURSE": 540000,
                "CIBLE_RETARD": 0,
            },
            {
                "AGENCE": "DLA",
                "SECTEUR ACTIVITE": "TRANSPORT",
                "TYPE DE CREDIT": "IMMOBILIER",
                "ANNEE OUVERTURE": 2019,
                "MOIS OUVERTURE": 9,
                "Age": 52,
                "Nombre de Crédit antérieur": 8,
                "SOLDE_CREDIT": 420000,
                "MONTANT_DEBOURSE": 1000000,
                "CIBLE_RETARD": 1,
            },
        ]

        X = np.vstack([self._transform_record(record)[0] for record in data])
        y = np.array([record["CIBLE_RETARD"] for record in data], dtype=int)

        self.model = XGBClassifier(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        )
        self.model.fit(X, y)
        self.median_credits = float(np.median([d["Nombre de Crédit antérieur"] for d in data]))

    def _normalize_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        credits_default = self.median_credits if self.median_credits is not None else 0
        categorical_defaults = {
            key: str(encoder.classes_[0])
            for key, encoder in self.encoders.items()
            if len(encoder.classes_)
        }
        defaults = {
            "AGENCE": categorical_defaults.get("AGENCE", "MECYD"),
            "SECTEUR ACTIVITE": categorical_defaults.get("SECTEUR ACTIVITE", "TIC"),
            "TYPE DE CREDIT": categorical_defaults.get("TYPE DE CREDIT", "CONSOMMATION"),
            "ANNEE OUVERTURE": 2024,
            "MOIS OUVERTURE": 1,
            "Age": 30,
            "Nombre de Crédit antérieur": credits_default,
            "SOLDE_CREDIT": 0,
            "MONTANT_DEBOURSE": 0,
        }
        data = {k: raw.get(k, defaults.get(k, 0)) for k in self.required_features}
        for key in categorical_defaults.keys() | {"AGENCE", "SECTEUR ACTIVITE", "TYPE DE CREDIT"}:
            value = data.get(key)
            data[key] = categorical_defaults.get(key, defaults.get(key)) if self._is_missing(value) else str(value)

        numeric_defaults = {
            "ANNEE OUVERTURE": 2024,
            "MOIS OUVERTURE": 1,
            "Age": 30,
            "Nombre de Crédit antérieur": credits_default,
            "SOLDE_CREDIT": 0,
            "MONTANT_DEBOURSE": 0,
        }
        for key, default in numeric_defaults.items():
            if key not in self.required_features:
                continue
            value = default if self._is_missing(data.get(key)) else data[key]
            try:
                numeric_value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Valeur numérique invalide pour '{key}'.") from exc
            if not math.isfinite(numeric_value):
                numeric_value = float(default)
            data[key] = int(numeric_value) if key in {"ANNEE OUVERTURE", "MOIS OUVERTURE", "Age"} else numeric_value
        return data

    @staticmethod
    def _is_missing(value: Any) -> bool:
        if value is None:
            return True
        if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
            return True
        try:
            return not math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    def get_feature_options(self) -> Dict[str, List[str]]:
        demo_options = {
            "AGENCE": ["MECYD", "DLA"],
            "SECTEUR ACTIVITE": ["CONSTRUCTION", "TIC", "AGROALIMENTAIRE", "TRANSPORT"],
            "TYPE DE CREDIT": ["CONSOMMATION", "IMMOBILIER"],
        }
        return {
            key: [str(value) for value in self.encoders[key].classes_]
            if key in self.encoders
            else options
            for key, options in demo_options.items()
        }

    def _transform_record(self, record: Dict[str, Any]) -> np.ndarray:
        normalized = self._normalize_record(record)
        mapping = {
            "AGENCE": {"MECYD": 0, "DLA": 1},
            "SECTEUR ACTIVITE": {"CONSTRUCTION": 0, "TIC": 1, "AGROALIMENTAIRE": 2, "TRANSPORT": 3},
            "TYPE DE CREDIT": {"CONSOMMATION": 0, "IMMOBILIER": 1},
        }

        values = []
        for key in self.required_features:
            value = normalized.get(key, 0)
            if key in mapping:
                encoder = self.encoders.get(key)
                if encoder is not None:
                    try:
                        value = int(encoder.transform([str(value)])[0])
                    except ValueError as exc:
                        raise ValueError(f"Catégorie inconnue pour '{key}': {value}") from exc
                else:
                    value = mapping[key].get(str(value).strip().upper(), 0)
            values.append(float(value))
        return np.array([values], dtype=float)

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        matrix = self._transform_record(record)
        probability = float(self.model.predict_proba(matrix)[0][1])
        prediction = int(self.model.predict(matrix)[0])
        status = "Risque élevé" if prediction == 1 else "Risque faible"
        return {
            "prediction_code": prediction,
            "decision": status,
            "probability": round(probability * 100, 2),
            "level": "High" if prediction == 1 else "Low",
        }

    def _coerce_history_rows(self, rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        cleaned = []
        for row in rows:
            if not row:
                continue
            source = {str(key).strip(): value for key, value in row.items() if key is not None}
            normalized = {key: source[key] for key in self.required_features if key in source}
            target = source.get("CIBLE_RETARD")
            if self._is_missing(target):
                retard = source.get("RETARD")
                if not self._is_missing(retard):
                    target = 0 if str(retard).strip().upper() == "0 JOUR" else 1
            if not self._is_missing(target):
                try:
                    numeric_target = float(target)
                except (TypeError, ValueError) as exc:
                    raise ValueError("La cible 'CIBLE_RETARD' doit valoir 0 ou 1.") from exc
                if numeric_target not in {0.0, 1.0}:
                    raise ValueError("La cible 'CIBLE_RETARD' doit valoir 0 ou 1.")
                target_value = int(numeric_target)
                normalized["CIBLE_RETARD"] = target_value
            if normalized:
                cleaned.append(normalized)
        return cleaned

    def evaluate_history(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not rows:
            return {
                "error": "Aucune donnée historique reçue.",
                "records": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        cleaned = self._coerce_history_rows(rows)
        if not cleaned:
            return {
                "error": "Aucune donnée exploitable.",
                "records": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        target_column_present = any(
            str(key).strip() in {"CIBLE_RETARD", "RETARD"}
            for row in rows if row
            for key in row
        )
        if not target_column_present:
            return {
                "error": "Aucune colonne cible détectée. Ajoutez 'CIBLE_RETARD' ou 'RETARD'.",
                "records": len(cleaned),
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }

        labeled_rows = [row for row in cleaned if "CIBLE_RETARD" in row]
        if not labeled_rows:
            return {
                "error": "Aucune cible exploitable. Les lignes sans retard connu ne sont pas évaluées.",
                "records": 0,
                "accuracy": 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "confusion_matrix": [[0, 0], [0, 0]],
            }
        X_array = np.vstack([self._transform_record(row)[0] for row in labeled_rows])
        y_true = [int(row["CIBLE_RETARD"]) for row in labeled_rows]
        y_true_array = np.array(y_true, dtype=int)
        y_pred = self.model.predict(X_array)
        cm = confusion_matrix(y_true_array, y_pred, labels=[0, 1]).tolist()
        return {
            "records": int(len(labeled_rows)),
            "accuracy": round(float(accuracy_score(y_true_array, y_pred)), 4),
            "precision": round(float(precision_score(y_true_array, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true_array, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true_array, y_pred, zero_division=0)), 4),
            "confusion_matrix": cm,
        }

    def get_demo_summary(self) -> Dict[str, Any]:
        if self.mode == "production":
            if self.metrics:
                return self.metrics
            return {
                "records": 0,
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "auc_roc": None,
                "confusion_matrix": [[0, 0], [0, 0]],
                "error": "Métriques de test absentes de l'artefact.",
            }

        history = [
            {"AGENCE": "MECYD", "SECTEUR ACTIVITE": "CONSTRUCTION", "TYPE DE CREDIT": "CONSOMMATION", "ANNEE OUVERTURE": 2022, "MOIS OUVERTURE": 5, "Age": 28, "Nombre de Crédit antérieur": 2, "SOLDE_CREDIT": 120000, "MONTANT_DEBOURSE": 300000, "CIBLE_RETARD": 0},
            {"AGENCE": "DLA", "SECTEUR ACTIVITE": "TIC", "TYPE DE CREDIT": "IMMOBILIER", "ANNEE OUVERTURE": 2021, "MOIS OUVERTURE": 11, "Age": 35, "Nombre de Crédit antérieur": 4, "SOLDE_CREDIT": 260000, "MONTANT_DEBOURSE": 700000, "CIBLE_RETARD": 1},
            {"AGENCE": "MECYD", "SECTEUR ACTIVITE": "AGROALIMENTAIRE", "TYPE DE CREDIT": "CONSOMMATION", "ANNEE OUVERTURE": 2023, "MOIS OUVERTURE": 2, "Age": 42, "Nombre de Crédit antérieur": 1, "SOLDE_CREDIT": 95000, "MONTANT_DEBOURSE": 180000, "CIBLE_RETARD": 0},
            {"AGENCE": "DLA", "SECTEUR ACTIVITE": "CONSTRUCTION", "TYPE DE CREDIT": "IMMOBILIER", "ANNEE OUVERTURE": 2020, "MOIS OUVERTURE": 8, "Age": 47, "Nombre de Crédit antérieur": 7, "SOLDE_CREDIT": 370000, "MONTANT_DEBOURSE": 900000, "CIBLE_RETARD": 1},
            {"AGENCE": "MECYD", "SECTEUR ACTIVITE": "TIC", "TYPE DE CREDIT": "CONSOMMATION", "ANNEE OUVERTURE": 2024, "MOIS OUVERTURE": 1, "Age": 31, "Nombre de Crédit antérieur": 3, "SOLDE_CREDIT": 145000, "MONTANT_DEBOURSE": 540000, "CIBLE_RETARD": 0},
        ]
        return self.evaluate_history(history)


service = RiskModelService()
