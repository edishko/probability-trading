from backtesting import Strategy
import pandas as pd


class PercentileStrategy(Strategy):  
    period = 100
    percentile_high = 0.5
    percentile_low = 0.5
    mode = 'long'

    @staticmethod
    def default_func(series: pd.Series, q: float) -> pd.Series:
        return series.rolling(window=PercentileStrategy.period, min_periods=10).quantile(q)

    func = default_func  # can be overridden

    def init(self):
        open_ = self.data.Open
        high_ = self.data.High
        low_ = self.data.Low

        # Relative intrabar move from open
        high_move = (high_ - open_) / open_
        low_move = (open_ - low_) / open_

        # Compute thresholds using the custom quantile function
        self.threshold_high = self.I(self.func, pd.Series(high_move), self.percentile_high)
        self.threshold_low  = self.I(self.func, pd.Series(low_move),  self.percentile_low)

    def next(self):
        if len(self.data) < 10:  # Ensure minimum 10 candles
            return
            
        p = self.data.Close[-1]
        h = self.threshold_high[-1]
        l = self.threshold_low[-1]

        if pd.isna(h) or pd.isna(l):
            return
            
        tp, sl = p * (1 + h), p * (1 - l)

        if not self.position:
            if self.mode == 'long' and sl < p < tp:
                self.buy(tp=tp, sl=sl, size=1)  # Reduced size for safety
            elif self.mode == 'short' and tp < p < sl:
                self.sell(tp=tp, sl=sl, size=1)


if __name__ == "__main__":
    import yfinance as yf
    from backtesting import Backtest
    import time
    import math

    tickers = ["GC=F", "BTC-USD"]
    period = "2y"
    interval = "1h"
    percentile_high = 0.5
    percentile_low  = 0.5

    print("Downloading data...")
    data = yf.download(
        tickers=tickers, 
        period=period, 
        interval=interval, 
        group_by='ticker',
        progress=False
    )
    
    start = time.perf_counter()

    for ticker in tickers:
        print(f"\nBacktesting {ticker}...")
        df = data[ticker].copy().dropna()

        bt = Backtest(df, PercentileStrategy, cash=math.inf)
        stats = bt.run(
            percentile_high=percentile_high,
            percentile_low=percentile_low,
        )

        print(stats)

    print(f"\nTotal execution time: {time.perf_counter() - start:.2f} seconds")
