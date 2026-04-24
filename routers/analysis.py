import json
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..db.database import SessionLocal, get_db
from ..db.models import AnalysisRun
from ..iut.pipeline import run_analysis_pipeline

_TEMPLATES = Path(__file__).parents[1] / "templates"

router = APIRouter()
templates = Jinja2Templates(directory=str(_TEMPLATES))


def _pipeline_task(run_id: int, ticker_list: list[str]) -> None:
    """バックグラウンドで分析パイプラインを実行し結果を DB に保存する。"""
    db = SessionLocal()
    try:
        run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
        if not run:
            return
        run.status = "running"
        db.commit()

        results = run_analysis_pipeline(ticker_list)

        run.status = "completed"
        run.result_count = len(results)
        run.results_json = json.dumps(results)
        db.commit()
    except Exception as exc:
        run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
        if run:
            run.status = "failed"
            run.error_message = str(exc)
            db.commit()
    finally:
        db.close()


@router.get("/", response_class=HTMLResponse)
async def index(request: Request, db: Session = Depends(get_db)):
    runs = db.query(AnalysisRun).order_by(AnalysisRun.created_at.desc()).limit(10).all()
    recent = [
        {
            "id": r.id,
            "tickers": r.tickers,
            "result_count": r.result_count,
            "created_at": r.created_at.strftime("%Y-%m-%d %H:%M UTC"),
            "status": r.status,
        }
        for r in runs
    ]
    return templates.TemplateResponse("index.html", {"request": request, "recent": recent})


@router.post("/analyze", response_class=HTMLResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    tickers: str = Form(...),
    db: Session = Depends(get_db),
):
    ticker_list = [t.strip().upper() for t in tickers.replace(",", " ").split() if t.strip()]

    run = AnalysisRun(
        tickers=", ".join(ticker_list),
        result_count=0,
        results_json="[]",
        status="pending",
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    background_tasks.add_task(_pipeline_task, run.id, ticker_list)

    return RedirectResponse(url=f"/results/{run.id}", status_code=303)


@router.get("/api/run/{run_id}/status")
async def run_status(run_id: int, db: Session = Depends(get_db)):
    run = db.query(AnalysisRun).filter(AnalysisRun.id == run_id).first()
    if not run:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(
        {
            "status": run.status,
            "result_count": run.result_count,
            "error_message": run.error_message,
        }
    )
