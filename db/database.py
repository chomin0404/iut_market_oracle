from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

ARTIFACTS_DIR = Path(__file__).parents[4] / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

DATABASE_URL = f"sqlite:///{ARTIFACTS_DIR / 'iut_results.db'}"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def migrate_add_columns() -> None:
    """既存 DB に新規カラムを追加する（idempotent）。"""
    new_columns = [
        "ALTER TABLE analysis_runs ADD COLUMN status VARCHAR DEFAULT 'completed'",
        "ALTER TABLE analysis_runs ADD COLUMN error_message TEXT",
    ]
    with engine.connect() as conn:
        for stmt in new_columns:
            try:
                conn.execute(text(stmt))
                conn.commit()
            except Exception:
                pass  # カラムが既に存在する場合はスキップ
