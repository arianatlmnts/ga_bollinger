import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

'''  ## Funcion duplicada
def returns_fitness(returns):
    neg_returns = len(list(filter(lambda x: (x < 0), returns)))
    pos_returns = len(list(filter(lambda x: (x > 0), returns)))

    return pos_returns / (neg_returns+pos_returns)
'''
def returns_fitness(pos_returns, neg_returns):
    return pos_returns / (neg_returns+pos_returns)

''' funciones para el calculo de medias'''

def SMA (close,window = 20 ):

    mean = []
    window_values = []

    for  i in range(len(close)):
        if i <= window-1:
            window_values.append(close[i])
            mean.append(np.mean(window_values))

        if i >= window:
            window_values.pop(0)
            window_values.append(close[i])
            mean.append(np.mean(window_values))

    return (mean)


def WMA (close,window = 20 ):

    mean = []
    window_values = []
    window_ind = np.arange(1,window+1)

    for  i in range(len(close)):
        if i <= window-1:
            window_values.append(close[i])
            i_sum = np.sum(list(range(1,i+2))) # si no lo vuelvo a calcular no hay problema ya alcanzo su valor opjetivo
            np_sum  = np.sum((np.arange(1,len(window_values)+1) * window_values)/i_sum)

            mean.append(np_sum)


        if i >= window:
            window_values.pop(0)
            window_values.append(close[i])
            sum_ventana = window_ind*window_values
            sum_ventana = sum_ventana/i_sum
            sum_ventana = np.sum(sum_ventana)

            mean.append(sum_ventana)


    return (mean)

def EMA (close,window = 20,alfa = 0.08):
    #alfa = 2/(window + 1)
    mean = []
    window_values = []
    aprendizaje = 1-alfa
    factor = []
    for i in range(len(close)):

        if i <= window:
            window_values.append(close[i])
            factor.append(aprendizaje)
            exponencial = np.flipud(np.arange(i+1))
            factor_exp = np.power(factor,exponencial)
            factor_ventana = window_values*factor_exp
            sum_exp = np.sum(factor_exp)
            factor_ventana =np.sum(factor_ventana/sum_exp)

            mean.append(factor_ventana)



        if i > window:
            window_values.pop(0)
            window_values.append(close[i])
            factor_ventana = window_values*factor_exp
            factor_ventana = np.sum(factor_ventana/sum_exp)

            mean.append(factor_ventana)

    return mean



def buy_sell(cierre,superior,inferior,window_size):

    compra = []
    venta = []

    dolares = 100
    eur = 0

    dolar = [100]
    euros = []
    regreso_po = 0
    regreso_neg = 0

    for i in range(window_size, superior.shape[0]):
        if( cierre[i-1] < inferior[i-1] and cierre[i] > inferior[i] and dolares > 0):
            eur = dolares/cierre[i]
            dolares = 0
            euros.append(eur)
            compra.append(i)

        if ( cierre[i-1] > superior[i-1] and cierre[i] < superior[i] and eur > 0):
            dolares = eur*cierre[i]
            if( (dolares - dolar[-1]) > 0 ):
                regreso_po += 1
            elif( (dolares - dolar[-1]) < 0):
                regreso_neg += 1
            dolar.append(dolares)
            eur = 0
            venta.append(i)

    if(dolares == 0):
        dolares = eur*cierre[-1]
        dolar.append(dolares)

    return regreso_po, regreso_neg, dolares



def calculate_bollinger_bands(data,n=20,k1=2,k2=2):
    ma = data.rolling(window=n).mean()
    std = data.rolling(window=n).std()
    upper_band = ma + (k1*std)
    lower_band = ma - (k2*std)

    return ma, upper_band, lower_band


if __name__ == '__main__':
    df = pd.read_csv('data/EURUSD_Candlestick_15_m_BID_01.01.2007-31.12.2007.csv')

    #p = 200
    n = 20
    k = 2.5

    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']

    middle, upper, lower = calculate_bollinger_bands(Close,n=n,k1=k,k2=k)

    Upper = np.array(upper)
    Lower = np.array(lower)
    Close = np.array(Close)


    regreso_po, regreso_neg, usd = buy_sell(cierre=Close, superior=Upper, inferior=Lower, window_size=n)

    returns_fit = returns_fitness(regreso_po, regreso_neg)
    print('return fitness: ',returns_fit)
    print('usd: ',usd)

    plt.figure(figsize=(10,6))
    plt.plot(Close)
    plt.plot(middle)
    plt.plot(upper)
    plt.plot(lower)
    plt.show()
