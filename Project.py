# -*- coding: utf-8 -*-
"""
Project 
08/08/2018

"""
import pandas as pd
import scipy.optimize as opt
import numpy as np
import os


os.getcwd()
os.chdir('D:\\AW Documents\\zwang15\\Jobs\\2018\\CoV\\')

def calRet(strategy, position):
    ret_st = 'ret_{}'.format(position)
    equity_st = 'equity_{}'.format(position)
    strategy.loc[:, ret_st] = strategy['price'].pct_change(1) * strategy[position]
    strategy.loc[:, equity_st] = (strategy[ret_st] + 1.0).cumprod()
    return strategy

def getTrades(trades, position):
    df = trades[[position]].dropna()
    df = df[df[position]<>0]
    df = df[position].apply(lambda x: 'Purchase' if x > 0 else 'Sell' )
    return df

def calAlpha(returns, position):
    ret_st = 'ret_{}'.format(position)
    beta_st = 'beta_{}'.format(position)
    alpha_st = 'alpha_{}'.format(position)
    cum_alpha_st = 'cum_alpha_{}'.format(position)
    
    returns.loc[:, beta_st] = returns['ret_sp'].rolling(window=200).corr(returns['ret_St1'])
    
    returns.loc[:, alpha_st] = returns[ret_st] - returns[beta_st] * returns['ret_sp']
    returns.loc[:, cum_alpha_st] = (returns[alpha_st] + 1.0).cumprod()
    return returns

#####
# functions for max sharpe
def maxSharpe(w0, means, cov):
    '''   max Sharpe '''

    assets = len(means)

    args = (means, cov)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple( (0,1) for asset in range(assets))

    results = opt.minimize(negSharpe, w0, args=args,
                        method='SLSQP', bounds=bounds, constraints=constraints)
    return results

def negSharpe(weights,means, cov):
    '''    min -sharpe = max sharpe    '''
    ret_p = np.sum( means*weights )
    std_p = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
    negSR = - ret_p / std_p
    return negSR


#########################   
# read data
prc = pd.read_csv('AAPL.csv').set_index('Date')[['Adj Close']]
prc.columns = ['price']

# Strategy 1
'''a moving average strategy that tests a 5-day and 20-day moving average crossover.
 The strategy should go long when the 5-day moving average crosses above the 20-day 
 moving average and short when the 5-day moving average crosses below the 20-day 
 moving average. '''
# Calculate moving average
strategy1 = prc.copy()
strategy1.loc[:, 'MA5'] = strategy1['price'].rolling(window=5).mean()
strategy1.loc[:, 'MA20'] = strategy1['price'].rolling(window=20).mean()
    
strategy1.dropna(how='any', inplace=True)
strategy1.loc[:, 'St1'] = strategy1[['MA5', 'MA20']].apply(lambda x: 1 if x['MA5'] > x['MA20'] else -1, axis=1).shift(1)
strategy1 = calRet(strategy1, position='St1')

# Strategy 2
'''
Build another strategy that buys AAPL when it closes down 3 days in a row and 
then sells the stock 1, 5, and 10 days later.  
'''
strategy2 = prc.copy()

strategy2.loc[:, 'down3day'] = (strategy2['price'] < strategy2['price'].shift(1)) & (strategy2['price'].shift(1) < strategy2['price'].shift(2))& (strategy2['price'].shift(2) < strategy2['price'].shift(3))
strategy2.loc[:, 'down3day'] = strategy2['down3day'].map({True: 1.0, False: 0.0}) 

    # initialize st2 and sell 1 day later
strategy2.loc[:, 'st2_1'] = strategy2['down3day'].shift(1)
strategy2 = calRet(strategy2, position='st2_1')


    # initialize st2 and sell 5 day later
    # during the 5 days, if new purchasing signal comes in, restart the 5 day
strategy2.loc[:, 'st2_5'] = strategy2['down3day'].replace(to_replace=0.0, limit=4, method='ffill').shift(1)
strategy2 = calRet(strategy2, position='st2_5')

    # initialize st2 and sell 10 day later
    # during the 10 days, if new purchasing signal comes in, restart the 10 day
strategy2.loc[:, 'st2_10'] = strategy2['down3day'].replace(to_replace=0.0, limit=9, method='ffill').shift(1)
strategy2 = calRet(strategy2, position='st2_10')

##############
# report results
strategy = strategy2.join(strategy1[['St1', 'ret_St1', 'equity_St1']], how='left')
    # graph
strategy[['equity_St1', 'equity_st2_1', 'equity_st2_5', 'equity_st2_10']].plot(figsize=(12,9))
strategy.to_csv('strategy.csv')

    # list trades
trades = strategy[['St1', 'st2_1', 'st2_5', 'st2_10']].diff(1)
trades_st1 = getTrades(trades, 'St1')
trades_st2_1 = getTrades(trades, 'st2_1')
trades_st2_5 = getTrades(trades, 'st2_5')
trades_st2_10 = getTrades(trades, 'st2_10')

trades_st2_10.to_csv('temp.csv')


    # Performance metrics
    # MDD is calculated in excel
    # alpha
sp = pd.read_csv('GSPC.csv').set_index('Date')[['Adj Close']]
sp.columns = ['price']
sp.loc[:, 'ret_sp'] = sp['price'].pct_change(1)   

returns = sp[['ret_sp']].join(strategy[['ret_St1', 'ret_st2_1', 'ret_st2_5', 'ret_st2_10']])
    
alpha = calAlpha(returns, 'St1')
alpha = calAlpha(alpha, 'st2_1')
alpha = calAlpha(alpha, 'st2_5')
alpha = calAlpha(alpha, 'st2_10')

alpha[['cum_alpha_St1', 'cum_alpha_st2_1', 'cum_alpha_st2_5', 'cum_alpha_st2_10']].plot(figsize=(15,9))
alpha.to_csv('temp.csv') 

    # Sharpe Ratio
sharpe = alpha[['ret_St1', 'ret_st2_1', 'ret_st2_5', 'ret_st2_10']].groupby(lambda _ : True).agg(['mean', 'std']).reset_index()
sharpe.to_csv('temp.csv')    
    

    # Transaction costs
print 'trades_st1', len(trades_st1)    
print 'trades_st2_1', len(trades_st2_1)    
print 'trades_st2_5', len(trades_st2_5)    
print 'trades_st2_10', len(trades_st2_10)  
  
    
# Capital Allocation
    # max sharpe; ignore risk free rate.
    # calculate inputs
means = np.array( alpha[['ret_St1', 'ret_st2_1', 'ret_st2_5', 'ret_st2_10']].dropna().mean())
cov = np.array(alpha[['ret_St1', 'ret_st2_1', 'ret_st2_5', 'ret_st2_10']].dropna().cov())

w0 = np.array([0.2,0.2, 0.1,0.5])
optimum = maxSharpe(w0, means, cov)
sharpe = - negSharpe(optimum['x'],means, cov) * np.sqrt(250)
opt_weights = optimum['x'].round(2)
print opt_weights

################################################
# future data exploration

# read all data
BTC_short_positions = pd.read_csv('BTC_short_positions.csv')
lastSwapsUSD = pd.read_csv('lastSwapsUSD.csv', nrows=200000)
lastSwapsUSD.columns = ['Timestamp', 'Date', 'Rate', 'Amount', 'Period', 'isFRR']
lastSwapsBTC = pd.read_csv('lastSwapsBTC.csv', nrows=200000)
lastSwapsBTC.columns = ['Timestamp', 'Date', 'Rate', 'Amount', 'Period', 'isFRR']


# explore an arbitrage opportunity
''' 1. buy BTC in bitfinex 
    2. sell future in bitmax to lock in profit, and take 70% usd 
    3. lend BTC and 70% usd in bitfinex
    4. close all positions
my work here is to back test the returns.
'''
    
lastSwapsUSD = lastSwapsUSD[lastSwapsUSD['Period']==30]
lastSwapsUSD.loc[:, 'trade_date'] = lastSwapsUSD['Date'].apply(lambda x: x.split(' ')[0])
usd = lastSwapsUSD.groupby('trade_date')[['Rate', 'Amount']].mean().reset_index()

lastSwapsBTC = lastSwapsBTC[lastSwapsBTC['Period']==30]
lastSwapsBTC.loc[:, 'trade_date'] = lastSwapsBTC['Date'].apply(lambda x: x.split(' ')[0])
btc = lastSwapsBTC.groupby('trade_date')[['Rate', 'Amount']].mean().reset_index()

# I have the agg daily return data
''' 1.This is a absolute return strategy;
    2. I can compare the return from this strategy with our targeted return or risk
    free rate to justify this strategy;
    3. future improvement
'''
    

