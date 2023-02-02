# Author: Mitch Mitchell (jbm8efn@virginia.edu)
# Functions for calculating various optimal portfolios

import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize

def get_data(tickers, period, interval):
    fields = []
    for i in range(len(tickers)):
        fields.append((tickers[i], 'Close'))

    df = yf.download(
        tickers = tickers,
        period = period,
        interval = interval,
        group_by = 'ticker'
    )[fields]
    return df

def get_returns(df, type={'norm', 'log'}):
    if type == 'norm':
        returns = df.pct_change().dropna()
        returns = returns.mean() * 252
        return returns
    elif type == 'log':
        returns = np.log(1 + df.pct_change().dropna())
        returns = returns.mean() * 252
        return returns

def get_cov(df, type={'norm', 'log'}):
    if type == 'norm':
        cov = df.pct_change().dropna()
        cov = cov.cov() * 252
        return cov
    elif type == 'log':
        cov = np.log(1 + df.pct_change().dropna())
        cov = cov.cov() * 252
        return cov

def find_beta(cov, stock, market):
    a = cov[(stock, 'Close')][(market, 'Close')]
    b = cov[(market, 'Close')][(market, 'Close')]
    return (a / b)

def calc_beta(cov, tickers):
    betas = pd.DataFrame(columns = ['Beta'])
    for i in tickers:
        betas.loc[i] = find_beta(cov, i, tickers[len(tickers) - 1])
    return betas

def capm_returns(returns, r_f, betas):
    returns = pd.DataFrame()
    returns['Returns'] = returns - r_f
    for i in range(len(betas)):
        returns.iloc[i]['Returns'] = returns.iloc[i]['Returns'] * betas.iloc[i] + r_f
    return returns

def portfolio_var(w, cov):
    w = np.matrix(w)
    cov = np.matrix(cov)
    result = w * cov * w.T
    return result

def portfolio_returns(w, yearly_returns):
    w = np.matrix(w)
    yearly_returns = np.matrix(yearly_returns)
    return np.sum(w * yearly_returns.T)

def portfolio_sharpe(w, cov, r_f):
    a = portfolio_returns(w, cov) - r_f
    b = math.sqrt(portfolio_var(w, cov))
    return -a / b

def abs_min_var(tickers, period, interval, dist={'norm', 'log'}, shorting={False, True}):
    w0 = [1 / len(tickers)] * len(tickers)
    bounds = [()]
    if shorting:
        bounds = tuple((-np.inf, np.inf) for i in tickers)
    else:
        bounds = tuple((0, 1) for i in tickers)
    df = get_data(tickers, period, interval)
    cov = get_cov(df, dist)
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    optimal = optimize.minimize(
        fun = portfolio_var,
        x0 = w0,
        args = cov,
        method = 'SLSQP',
        bounds = bounds,
        constraints = cons
    )
    return dict(zip(tickers, optimal.x))

def min_var(tickers, period, interval, target, cons={'returns', 'beta'}, dist={'norm', 'log'}, shorting={False, True}):
    w0 = [1 / len(tickers)] * len(tickers)
    bounds = [()]
    if shorting:
        bounds = tuple((-np.inf, np.inf) for i in tickers)
    else:
        bounds = tuple((0, 1) for i in tickers)
    df = get_data(tickers, period, interval)
    cov = get_cov(df, dist)
    w_cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    return_cons = {'type': 'eq', 'fun': lambda x: portfolio_returns(x, get_returns(df, dist)) - target}
    beta_cons = {'type': 'eq', 'fun': lambda x: portfolio_returns(x, calc_beta(cov, tickers)['Beta']) - target}
    optimal = optimize.minimize(
        fun = portfolio_var,
        x0 = w0,
        args = cov,
        method = 'SLSQP',
        bounds = bounds,
        constraints = (w_cons, return_cons) if cons == 'returns' else (w_cons, beta_cons)
    )
    return dict(zip(tickers, optimal.x))

def max_sharpe(tickers, period, interval, r_f, dist={'norm', 'log'}, shorting={False, True}):
    w0 = [1 / len(tickers)] * len(tickers)
    bounds = [()]
    if shorting:
        bounds = tuple((-np.inf, np.inf) for i in tickers)
    else:
        bounds = tuple((0, 1) for i in tickers)
    cov = get_cov(get_data(tickers, period, interval), dist)
    cons = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
    optimal = optimize.minimize(
        fun = portfolio_sharpe,
        x0 = w0,
        args = (cov, r_f),
        method = 'SLSQP',
        bounds = bounds,
        constraints = cons
    )
    return dict(zip(tickers, optimal.x))

if __name__ == '__main__':
    market = '^GSPC'
    tickers = [
        'AAPL',
        'MSFT',
        'GOOG',
        'AMZN',
        'BRK-B',
        'V',
        'XOM',
        'UNH',
        'JNJ',
        'NVDA',
        market
    ]
    period = '5y'
    interval = '1d'
    df = get_data(tickers, period, interval)
    allow_short = False
    r_f = 0.01
    dist = 'log'
    cons = 'returns'
    target = 0.12
    optimal1 = abs_min_var(tickers, period, interval, dist, allow_short)
    optimal2 = min_var(tickers, period, interval, target, cons, dist, allow_short)
    optimal3 = max_sharpe(tickers, period, interval, r_f, dist, allow_short)