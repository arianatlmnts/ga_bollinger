import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def returns_fitness(returns):
    neg_returns = len(list(filter(lambda x: (x < 0), returns)))
    pos_returns = len(list(filter(lambda x: (x > 0), returns)))

    return pos_returns / (neg_returns+pos_returns)

def buy_sell(close,upper,lower,n):
    buy = []
    sell = []

    usd = [100]
    eur = [0]

    returns = []

    for i in range(n, len(close)):
        if (close[i-1] < lower[i-1]) and (close[i] > lower[i]) and (usd[-1]>0):
            buy.append(i)
            eur.append(usd[-1]/close[i])
            usd.append(0)

        if (close[i-1] > upper[i-1]) and (close[i] < upper[i]) and (eur[-1]>0) or i == len(close):
            sell.append(i)
            usd.append(eur[-1]*close[i])
            eur.append(0)


            if usd[-2] != 0:
                returns.append(usd[-1]-usd[-2])
            else:
                returns.append(usd[-1]-usd[-3])

    return returns, usd

def calculate_bollinger_bands(data,n=20,k1=2,k2=2):
    ma = data.rolling(window=n).mean()
    std = data.rolling(window=n).std()
    upper_band = ma + (k*std)
    lower_band = ma - (k*std)

    return ma, upper_band, lower_band


if __name__ == '__main__':
    df = pd.read_csv('DATA/EURUSD_Candlestick_15_m_BID_01.01.2007-31.12.2007.csv')

    #p = 200
    n = 20
    k = 2.5

    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']

    middle, upper, lower = calculate_bollinger_bands(Close,n=n,k=k)
    returns, usd = buy_sell(Close, upper, lower, n)

    returns_fit = returns_fitness(returns)
    print(returns_fit)
    print(usd)

    plt.figure(figsize=(10,6))
    plt.plot(Close)
    plt.plot(middle)
    plt.plot(upper)
    plt.plot(lower)
    plt.show()

