from datetime import datetime

from pydantic import BaseModel


class TickerResult(BaseModel):
    ticker: str
    distortion: float
    entropy: float
    resonance_status: str
    purity: float
    reconstructed_mean: float
    recon_entropy: float


class RunSummary(BaseModel):
    id: int
    tickers: list[str]
    result_count: int
    created_at: datetime


class AnalysisDetail(BaseModel):
    id: int
    tickers: list[str]
    result_count: int
    created_at: datetime
    results: list[TickerResult]
