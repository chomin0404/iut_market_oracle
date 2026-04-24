from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from .database import Base


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"

    id = Column(Integer, primary_key=True, index=True)
    tickers = Column(String, nullable=False)  # カンマ区切り
    result_count = Column(Integer, nullable=False, default=0)  # 通過した銘柄数
    results_json = Column(Text, nullable=False, default="[]")  # JSON 文字列
    status = Column(String, nullable=False, default="pending")  # pending|running|completed|failed
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
