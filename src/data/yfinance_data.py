from src.data.news_data_range import get_news_data_range
import yfinance as yf
from datetime import datetime, timedelta
from typing import List

class YFinanceData:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def get_data(self):
        start_iso, end_iso = get_news_data_range()
        start_date = datetime.fromisoformat(start_iso).date()
        end_date = datetime.fromisoformat(end_iso).date()

        # yfinance の end は非包含のため、最終日を含めるために +1 日する
        df = yf.Ticker(self.ticker).history(
            start=start_date,
            end=end_date + timedelta(days=1),
            interval="1d",
        )
        if df.empty:
            # ニュース期間が休場や週末に当たる等でデータが空のことがあるためフォールバック
            buffer_days = 14
            df = yf.Ticker(self.ticker).history(
                start=start_date - timedelta(days=buffer_days),
                end=end_date + timedelta(days=buffer_days + 1),
                interval="1d",
            )
        # 終値のみ返す（Date index, Close 列）
        return df[["Close"]]

    def summarize_closing_trend(self, max_points: int = 10) -> str:
        """終値・前日差分・変化率(%)の推移をテキスト化して返す。

        例: "2025-01-05 39000.00 (Δ+200.00, +0.52%)"
        """
        df = self.get_data()
        if df.empty:
            return "価格データ無し"

        # 差分と変化率を計算
        df = df.copy()
        df["Diff"] = df["Close"].diff()
        df["Pct"] = df["Close"].pct_change() * 100.0

        # 直近から max_points 件取得（古い→新しい順で見やすく）
        tail_df = df.tail(max_points)

        lines: List[str] = []
        for idx, row in tail_df.iterrows():
            date_str = idx.date().isoformat()
            close_val = float(row["Close"]) if row["Close"] == row["Close"] else 0.0
            diff_val = row["Diff"]
            pct_val = row["Pct"]

            if diff_val == diff_val:  # not NaN
                diff_sign = "+" if diff_val > 0 else ("" if diff_val == 0 else "-")
                diff_text = f"{diff_sign}{abs(float(diff_val)):.2f}"
            else:
                diff_text = "-"

            if pct_val == pct_val:  # not NaN
                pct_sign = "+" if pct_val > 0 else ("" if pct_val == 0 else "-")
                pct_text = f"{pct_sign}{abs(float(pct_val)):.2f}%"
            else:
                pct_text = "-"

            lines.append(f"{date_str} {close_val:.2f} (Δ{diff_text}, {pct_text})")

        return "\n".join(lines)

if __name__ == "__main__":
    data = YFinanceData("^N225")
    print(data.summarize_closing_trend())