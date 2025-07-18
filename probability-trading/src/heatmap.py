# src/heatmap.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_heatmap(heatmap: pd.Series, ticker: str) -> None:
    """Plot optimization heatmap for a single asset."""
    df = heatmap.reset_index()
    df.columns = ['high', 'low', 'Win Rate']
    pivot = df.pivot(index='high', columns='low', values='Win Rate')

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(pivot, cmap='viridis', annot=True, fmt=".1f",
                     annot_kws={"size": 8}, cbar_kws={'label': 'Win Rate [%]'})
    
    best_high, best_low = heatmap.idxmax()
    best_win = heatmap.max()
    ax.scatter(
        x=pivot.columns.get_loc(best_low) + 0.5,
        y=pivot.index.get_loc(best_high) + 0.5,
        color='red', s=200, marker='*',
        label=f'Best: High={best_high:.2f}, Low={best_low:.2f}\nWin={best_win:.1f}%'
    )
    
    ax.set_title(f'{ticker} Win Rate Optimization', fontsize=14)
    ax.set_xlabel('Stop-loss Percentile (Low)', fontsize=10)
    ax.set_ylabel('Take-profit Percentile (High)', fontsize=10)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(f'{ticker}_heatmap.png', dpi=100)
    plt.close()
    print(f"Saved {ticker}_heatmap.png")


if __name__ == "__main__":
    from backtesting import Backtest
    from strategy import PercentileStrategy
    import yfinance as yf
    import numpy as np
    import math
    import time

    # Configuration
    tickers = ["GC=F", "BTC-USD"]
    period = "2y"
    interval = "1h"
    resolution = 10
    percentiles = list(np.round(np.linspace(0.01, 1, resolution), 2))

    # Download data
    data = yf.download(tickers=tickers, period=period, interval=interval, 
                       group_by='ticker', auto_adjust=True, progress=False)
    
    # Run optimization
    start = time.perf_counter()
    for ticker in tickers:
        print(f"\nOptimizing {ticker}...")
        df = data[ticker].dropna()
        
        if len(df) < 100:
            print(f"  Insufficient data ({len(df)} rows)")
            continue
            
        # Run backtest optimization
        bt = Backtest(df, PercentileStrategy, cash=math.inf)
        _, heatmap = bt.optimize(
            percentile_high=percentiles,
            percentile_low=percentiles,
            maximize='Win Rate [%]',
            return_heatmap=True
        )
        
        print(heatmap)
        plot_heatmap(heatmap, ticker)
    
    print(f"\nTotal time: {time.perf_counter() - start:.1f}s")