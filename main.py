from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .db.database import Base, engine, migrate_add_columns
from .routers import analysis, results

_HERE = Path(__file__).parent

# DB テーブル作成（初回起動時）
Base.metadata.create_all(bind=engine)
# 既存 DB への新規カラム追加
migrate_add_columns()

app = FastAPI(title="IUT Market Oracle", version="0.1.0")

app.mount("/static", StaticFiles(directory=str(_HERE / "static")), name="static")

app.include_router(analysis.router)
app.include_router(results.router)
