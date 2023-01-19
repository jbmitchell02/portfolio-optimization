import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize

def get_data(tickers):

    period = '1y' # Options are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, and max
    interval = '1d' # Options are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, and 3mo

    fields = []
    for i in range(len(tickers)):
        fields.append((tickers[i], 'Close'))
    return yf.download(tickers = tickers, period = period, interval = interval, group_by = 'ticker')[fields]


def get_cov(data):
    result = data.pct_change().dropna() * 100
    result = result.cov().to_numpy()
    return result

def portfolioVar(w, cov):
    w = np.matrix(w)
    cov = np.matrix(cov)
    result = w * cov * w.T
    return result

def minVarPortfolio(tickers, allow_short):
    data = get_data(tickers)
    cov = get_cov(data)
    w0 = [1 / len(tickers)] * len(tickers)
    constraint = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    bounds = [()]
    if allow_short:
        bounds = tuple((-np.inf, np.inf) for i in tickers)
    else:
        bounds = tuple((0, 1) for i in tickers)
    optimal = optimize.minimize(fun = portfolioVar, x0 = w0, args = cov, method = 'SLSQP', bounds = bounds, constraints = constraint)
    return dict(zip(tickers, optimal.x))
