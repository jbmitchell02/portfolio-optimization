# Author: Mitch Mitchell (jbm8efn@virginia.edu)
# Functions for calculating various optimal portfolios

import math
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import optimize


# tickers: Ticker symbols for stocks to be optimized over (last entry should be market index if using CAPM functionality)
# period: Amount of historical data to get (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, or max)
# interval: Amount of time between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, or 3mo)
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


# df: Dataframe containing stock 'Close' data (preferably from get_data method)
# type: Type of distribution to be used (norm or log)
def get_returns(df, type={'norm', 'log'}):
    if type == 'norm':
        returns = df.pct_change().dropna()
        returns = returns.mean() * 252
        return returns
    elif type == 'log':
        returns = np.log(1 + df.pct_change().dropna())
        returns = returns.mean() * 252
        return returns


# df: Dataframe containing stock 'Close' data (preferably from get_data method)
# type: Type of distribution to be used (norm or log)
def get_cov(df, type={'norm', 'log'}):
    if type == 'norm':
        cov = df.pct_change().dropna()
        cov = cov.cov() * 252
        return cov
    elif type == 'log':
        cov = np.log(1 + df.pct_change().dropna())
        cov = cov.cov() * 252
        return cov


# cov: Covariance matrix of stocks (including market index)
# stock: Ticker symbol for stock to generate beta for
# market: Ticker symbol for market index
def find_beta(cov, stock, market):
    a = cov[(stock, 'Close')][(market, 'Close')]
    b = cov[(market, 'Close')][(market, 'Close')]
    return (a / b)


# cov: Covariance matrix of stocks (including market index)
# tickers: Ticker symbols for stocks to calculate beta for
def calc_beta(cov, tickers):
    betas = pd.DataFrame(columns = ['Beta'])
    for i in tickers:
        betas.loc[i] = find_beta(cov, i, tickers[len(tickers) - 1])
    return betas


# returns: Expected yearly returns for stocks
# r_f: Risk-free rate
# betas: Beta values for stocks (preferably from calc_beta method)
def capm_returns(returns, r_f, betas):
    returns = pd.DataFrame()
    returns['Returns'] = returns - r_f
    for i in range(len(betas)):
        returns.iloc[i]['Returns'] = returns.iloc[i]['Returns'] * betas.iloc[i] + r_f
    return returns


# w: Portfolio weights
# cov: Covariance matrix of stocks
def portfolio_var(w, cov):
    w = np.matrix(w)
    cov = np.matrix(cov)
    result = w * cov * w.T
    return result


# w: Portfolio weights
# returns: Expected yearly returns for stocks
def portfolio_returns(w, returns):
    w = np.matrix(w)
    returns = np.matrix(returns)
    return np.sum(w * returns.T)


# w: Portfolio weights
# cov: Covariance matrix of stocks
# r_f: Risk-free rate
def portfolio_sharpe(w, cov, r_f):
    a = portfolio_returns(w, cov) - r_f
    b = math.sqrt(portfolio_var(w, cov))
    return -a / b


# tickers: Ticker symbols for stocks to be optimized over (last entry should be market index if using CAPM functionality)
# period: Amount of historical data to get (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, or max)
# interval: Amount of time between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, or 3mo)
# dist: Type of distribution to be used (norm or log)
# shorting: Whether or not shorting is allowed (unexpected results may occur if allowed)
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


# tickers: Ticker symbols for stocks to be optimized over (last entry should be market index if using CAPM functionality)
# period: Amount of historical data to get (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, or max)
# interval: Amount of time between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, or 3mo)
# target: Target value for expected yearly returns or beta
# cons: Which additional constraint to use (returns or beta)
# dist: Type of distribution to be used (norm or log)
# shorting: Whether or not shorting is allowed (unexpected results may occur if allowed)
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


# tickers: Ticker symbols for stocks to be optimized over (last entry should be market index if using CAPM functionality)
# period: Amount of historical data to get (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, or max)
# interval: Amount of time between data points (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, or 3mo)
# r_f: Risk-free rate
# dist: Type of distribution to be used (norm or log)
# shorting: Whether or not shorting is allowed (unexpected results may occur if allowed)
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


# Main method
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
    target = 0.12
    cons = 'returns'
    r_f = 0.01
    dist = 'log'
    allow_short = False
    optimal1 = abs_min_var(tickers, period, interval, dist, allow_short)
    optimal2 = min_var(tickers, period, interval, target, cons, dist, allow_short)
    optimal3 = max_sharpe(tickers, period, interval, r_f, dist, allow_short)
