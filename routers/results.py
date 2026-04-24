import csv
import io
import json
from pathlib import Path

import yfinance as yf
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..db.database import get_db
from ..db.models import AnalysisRun
from ..iut.pipeline import LOOKBACK_DAYS

_TEMPLATES = Path(__file__).parents[1] / "templates"

router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATES))


@router.get("/results", response_class=HTMLResponse)
async def list_results(request: Request, db: Session = Depends(get_db)):
    runs = db.query(AnalysisRun).order_by(AnalysisRun.created_at.desc()).all()
    items = [
        {
            "id": r.id,
            "tickers": r.tickers,
            "result_count": r.result_count,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            "status": r.status,
        }
        for r in runs
    ]
    return templates.TemplateResponse("results_list.html", {"request": request, "runs": items})


@router.get("/results/{run_id}", response_class=HTMLResponse)
async def get_result(run_id: int, request: Request, db: Session = Depends(get_db)):
    run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="分析結果が見つかりません")

    results = json.loads(run.results_json)
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "run_id": run.id,
            "tickers": run.tickers,
            "created_at": run.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            "result_count": run.result_count,
            "results": results,
            "results_json": run.results_json,
            "status": run.status,
            "error_message": run.error_message,
            "lookback_days": LOOKBACK_DAYS,
        },
    )


@router.get("/results/{run_id}/export.csv")
async def export_csv(run_id: int, db: Session = Depends(get_db)):
    run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
    if not run:
        raise HTTPException(status_code=404, detail="分析結果が見つかりません")

    results = json.loads(run.results_json)
    output = io.StringIO()
    fieldnames = [
        "ticker",
        "distortion",
        "entropy",
        "purity",
        "resonance_status",
        "reconstructed_mean",
        "recon_entropy",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename=iut_run_{run_id}.csv"},
    )


@router.get("/api/price-history/{ticker}")
async def price_history(ticker: str):
    try:
        df = yf.download(
            ticker.upper(),
            period=f"{LOOKBACK_DAYS}d",
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return JSONResponse({"dates": [], "prices": []})
        dates = df.index.strftime("%Y-%m-%d").tolist()
        prices = df["Close"].to_numpy(dtype=float).flatten().tolist()
        return JSONResponse({"dates": dates, "prices": prices})
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=500)
