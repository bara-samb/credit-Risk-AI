# Credit Risk AI

Application FastAPI pour le modèle XGBoost de risque de crédit défini dans `creditRisk.ipynb`. L’interface sépare les performances, la prédiction individuelle et l’évaluation d’un historique.

## Pages

- `/` : métriques du jeu de test et matrice de confusion.
- `/prediction` : évaluation d’un dossier individuel.
- `/historique` : évaluation de dossiers étiquetés depuis CSV ou JSON.

L’API expose aussi `/api/health`, `/api/summary`, `/api/options`, `/api/predict` et `/api/test-historical`.

## Prérequis

- Python 3.12 ou 3.14 pour l’exécution locale.
- Docker Desktop pour construire et démarrer le conteneur.
- Le fichier source `DataSet_Credit.xlsx` pour réentraîner le modèle. Le classeur n’est volontairement pas versionné.

## Installation locale

Dans PowerShell :

```powershell
py -3.14 -m venv .venv
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Le modèle de production est fourni dans `artifacts/model_credit_risk_artifacts.pkl`. Pour le réentraîner depuis le classeur source :

```powershell
python -m app.backend.train_model "C:\chemin\vers\DataSet_Credit.xlsx"
```

L’entraînement sépare les données en train/test avant de calculer la médiane et d’ajuster les encodeurs, afin d’éviter la fuite de données. L’artefact contient le modèle, les encodeurs, les métadonnées et les métriques du test.

Démarrer le serveur :

```powershell
python -m uvicorn app.backend.main:app --host 0.0.0.0 --port 8000
```

Puis ouvrir `http://localhost:8000`.

## Déploiement Docker local

L’image copie l’artefact dans le conteneur. Le classeur d’entraînement et le venv local ne sont pas inclus. Docker est configuré pour échouer au démarrage si l’artefact de production est absent ou invalide.

Construire et lancer :

```powershell
docker compose up --build -d
```

Vérifier la santé et les logs :

```powershell
docker compose ps
Invoke-RestMethod http://localhost:8000/api/health
docker compose logs -f credit-risk-app
```

Arrêter l’application :

```powershell
docker compose down
```

Le conteneur écoute sur le port `PORT` fourni par l’environnement, avec `8000` comme valeur locale par défaut. Il refuse le démarrage si l’artefact de production est absent ou invalide.

## Déploiement Render

Le dépôt contient un Blueprint `render.yaml` qui configure un service Web Docker, un contrôle de santé sur `/api/health` et le déploiement automatique depuis `main`.

1. Connecte-toi à Render et choisis **New + → Blueprint**.
2. Sélectionne le dépôt `bara-samb/credit-Risk-AI` et la branche `main`.
3. Vérifie les paramètres détectés : région Frankfurt, plan Free, Dockerfile à la racine.
4. Lance la création du service et attends que le premier déploiement passe le contrôle de santé.

Render attribuera alors une URL HTTPS de la forme `https://credit-risk-ai.onrender.com`. L’URL exacte est affichée dans le tableau de bord Render après création ; elle ne peut pas être connue avant que Render provisionne le service.

L’artefact `artifacts/model_credit_risk_artifacts.pkl` est inclus dans le dépôt et copié dans l’image. Si le modèle est réentraîné, reconstruis et pousse le changement pour déclencher un nouveau déploiement. Ne publie pas le classeur source ni des données client.

Le plan gratuit peut mettre le service en veille après une période d’inactivité ; la première requête suivante peut alors prendre plus de temps.

## Format des tests historiques

L’upload accepte un CSV avec ligne d’en-têtes ou un JSON contenant un objet ou une liste d’objets. Chaque ligne évaluée doit inclure `RETARD` (`0 JOUR` ou une valeur de retard) ou `CIBLE_RETARD` (`0` ou `1`). Les lignes sans cible sont ignorées. Les catégories doivent correspondre aux valeurs proposées par le modèle. Taille maximale : 10 Mo.

## Tests

```powershell
python -m unittest discover -s tests -v
```

## Publier sur GitHub

Après avoir vérifié le contenu qui sera envoyé :

```powershell
git init
git branch -M main
git remote add origin https://github.com/bara-samb/credit-Risk-AI.git
git add .
git status --short
git commit -m "Initial commit: credit risk AI app"
git push -u origin main
```

Le `.gitignore` exclut le venv, les fichiers d’environnement et les classeurs/CSV locaux, mais conserve l’artefact de production nécessaire au déploiement.