import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random


class Candidate(object):
    def __init__(self, genotype, fitness):
        self.genotype = genotype
        self.fitness = fitness

    def mutation(self):
        r = random.randint(0,100)
        p_mut = 20
        if r <= p_mut:
            x,y = random.sample(range(1, queens), 2)
            self.genotype[x],self.genotype[y] = self.genotype[y],self.genotype[x]
        self.fitness = fitness(self.genotype)



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

def fitness(g):

    df = pd.read_csv('DATA/EURUSD_Candlestick_15_m_BID_01.01.2007-31.12.2007.csv')

    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']


    k1 = g[0]
    k2 = g[1]
    ##m = g[2]
    n = int(g[3])


    middle, upper, lower = calculate_bollinger_bands(Close,n=n,k1=k1,k2=k2)

    Close = np.array(Close)
    Upper = np.array(upper)
    Lower = np.array(lower)


    #print(n)

    pos_returns, neg_returns, usd = buy_sell(cierre=Close, superior=Upper, inferior=Lower, window_size=n)

    return pos_returns / (neg_returns+pos_returns)


def calculate_bollinger_bands(data,n=20,k1=2,k2=2):
    ma = data.rolling(window=n).mean()
    std = data.rolling(window=n).std()
    upper_band = ma + (k1*std)
    lower_band = ma - (k2*std)

    return ma, upper_band, lower_band


if __name__ == '__main__':


    population_size = 100
    C = []
    p_mut = 0.1
    g = np.zeros(4)
    # initialize random population
    for i in range(1):
        x = random.uniform(1,3) #Valor aleatorio para el alpha de la banda superior
        y = random.uniform(1,3) #Valor aleatorio para la banda inferior
        g[0], g[1] = x, y
        g[2] = random.randint(0,2) #Selecciona el tipo de media a usar
        g[3] = random.randint(20,200) #


        C.append(Candidate(g.tolist(),fitness(g)))

    print(C[0].genotype, C[0].fitness)





    #plt.figure(figsize=(10,6))
    #plt.plot(Close)
    #plt.plot(middle)
    #plt.plot(upper)
    #plt.plot(lower)
    #plt.show()
