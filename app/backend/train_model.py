from __future__ import annotations

import argparse
import math
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


FEATURES = [
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
CATEGORICAL_FEATURES = ["AGENCE", "SECTEUR ACTIVITE", "TYPE DE CREDIT"]


def train(data_path: Path, artifact_path: Path) -> dict:
    if not data_path.is_file():
        raise FileNotFoundError(f"Fichier de données introuvable : {data_path}")

    df = pd.read_excel(data_path)
    required_columns = set(FEATURES + ["RETARD", "NUMERO_CREDIT", "NUMERO_MEMBRE"])
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"Colonnes manquantes dans le fichier Excel : {', '.join(missing)}")

    cleaned = df.dropna(subset=["RETARD"]).copy()
    target = cleaned["RETARD"].apply(
        lambda value: 0 if str(value).strip() == "0 JOUR" else 1
    )
    X = cleaned[FEATURES].copy()
    if target.nunique() != 2:
        raise ValueError("L'entraînement nécessite des dossiers des deux classes de retard.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, target, test_size=0.2, random_state=42, stratify=target
    )

    median_credits = float(X_train["Nombre de Crédit antérieur"].median())
    if not math.isfinite(median_credits):
        raise ValueError("Impossible de calculer la médiane des crédits antérieurs du train.")
    X_train["Nombre de Crédit antérieur"] = X_train["Nombre de Crédit antérieur"].fillna(median_credits)
    X_test["Nombre de Crédit antérieur"] = X_test["Nombre de Crédit antérieur"].fillna(median_credits)

    encoders = {}
    for column in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        X_train[column] = encoder.fit_transform(X_train[column].astype(str))
        class_indexes = {label: index for index, label in enumerate(encoder.classes_)}
        X_test[column] = X_test[column].astype(str).map(class_indexes).fillna(-1).astype(int)
        encoders[column] = encoder

    model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=42,
        n_jobs=-1,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = {
        "records": int(len(y_test)),
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "auc_roc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "confusion_matrix": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
    }

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = artifact_path.with_suffix(f"{artifact_path.suffix}.tmp")
    with temporary_path.open("wb") as artifact_file:
        pickle.dump(
            {
                "model": model,
                "encoders": encoders,
                "median_credits": median_credits,
                "features": FEATURES,
                "metrics": metrics,
                "artifact_version": 1,
            },
            artifact_file,
        )
    temporary_path.replace(artifact_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entraîne et exporte le modèle XGBoost du notebook.")
    parser.add_argument("data", type=Path, help="Chemin vers DataSet_Credit.xlsx")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "artifacts" / "model_credit_risk_artifacts.pkl",
        help="Chemin de sortie de l'artefact modèle",
    )
    args = parser.parse_args()
    metrics = train(args.data, args.output)
    print(f"Artefact enregistré : {args.output.resolve()}")
    print(f"Métriques de test : {metrics}")


if __name__ == "__main__":
    main()