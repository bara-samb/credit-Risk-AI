from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

from app.backend.model_service import service

app = FastAPI(title="Credit Risk AI Dashboard", version="1.0.0")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

ROOT = Path(__file__).resolve().parent.parent
templates = Jinja2Templates(directory=str(ROOT / "frontend" / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "frontend" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={"active_page": "dashboard", "page_title": "Performances"},
    )


@app.get("/prediction", response_class=HTMLResponse)
async def prediction_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="prediction.html",
        context={"active_page": "prediction", "page_title": "Prédiction"},
    )


@app.get("/historique", response_class=HTMLResponse)
async def history_page(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="history.html",
        context={"active_page": "history", "page_title": "Historique"},
    )


@app.get("/api/health")
async def health() -> Dict[str, Any]:
    return {"status": "ok", "model_mode": service.mode, "model_name": "XGBoost"}


@app.get("/api/summary")
async def get_summary() -> Dict[str, Any]:
    summary = service.get_demo_summary()
    return {
        "model_mode": service.mode,
        "model_name": "XGBoost",
        "metrics": summary,
    }


@app.get("/api/options")
async def get_options() -> Dict[str, List[str]]:
    return service.get_feature_options()


@app.post("/api/predict")
async def predict(payload: Dict[str, Any]):
    try:
        result = service.predict_single(payload)
        return result
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/api/test-historical")
async def test_historical(file: UploadFile = File(...)):
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Le fichier dépasse la limite de 10 Mo.")
    name = (file.filename or "").lower()

    try:
        if name.endswith(".csv"):
            text = content.decode("utf-8-sig")
            reader = csv.DictReader(io.StringIO(text))
            if not reader.fieldnames:
                raise HTTPException(status_code=400, detail="Le CSV ne contient pas d'en-têtes.")
            reader.fieldnames = [field.strip() if field else field for field in reader.fieldnames]
            data = [
                {key.strip(): value for key, value in row.items() if key is not None}
                for row in reader
            ]
        elif name.endswith((".json", ".txt")):
            text = content.decode("utf-8-sig")
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                data = [parsed]
            elif isinstance(parsed, list) and all(isinstance(row, dict) for row in parsed):
                data = parsed
            else:
                raise HTTPException(status_code=400, detail="Le JSON doit contenir un objet ou une liste d'objets.")
        else:
            raise HTTPException(status_code=400, detail="Format non pris en charge. Utilisez CSV ou JSON.")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Fichier invalide : {exc}") from exc

    try:
        metrics = service.evaluate_history(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return metrics


@app.get("/api/metrics")
async def metrics() -> Dict[str, Any]:
    return service.get_demo_summary()
