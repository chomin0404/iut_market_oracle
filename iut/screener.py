import numpy as np
import yfinance as yf

DISTORTION_THRESHOLD = 1.5


def screen_iut_distortion(ticker_list: list[str]) -> list[dict]:
    """
    IUT的歪み（|log P/B|）でスクリーニングする。

    distortion = |log(currentPrice) - log(bookValue)| = |log(P/B ratio)|
    """
    results = []

    for ticker in ticker_list:
        try:
            info = yf.Ticker(ticker).info
        except Exception as exc:
            print(f"[WARN] {ticker}: データ取得失敗 ({exc})")
            continue

        book_value: float | None = info.get("bookValue")
        current_price: float | None = info.get("currentPrice")

        if not book_value or not current_price:
            print(f"[WARN] {ticker}: bookValue または currentPrice が未取得")
            continue
        if book_value <= 0 or current_price <= 0:
            print(f"[WARN] {ticker}: 負または零の値 (book={book_value}, price={current_price})")
            continue

        distortion = float(np.abs(np.log(current_price) - np.log(book_value)))

        if distortion > DISTORTION_THRESHOLD:
            results.append({"ticker": ticker, "distortion": distortion})

    return results
