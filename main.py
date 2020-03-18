import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random 

'''  sección del geneotipo'''

class Candidate(object):
    def __init__(self, genotype, fitness):
        self.genotype = genotype
        self.fitness = fitness

    def mutation(self): #cambiar metodo esta mutacion no es aplicable a este metodo
        r = random.randint(0,100)
        p_mut = 10
        if r <= p_mut:
            x,y = random.sample(range(1, len(self.genotype)), 2)
            self.genotype[x],self.genotype[y] = self.genotype[y],self.genotype[x]
        self.fitness = fitness(self.genotype)


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

def EMA (close,window = 20):

    alfa = 2/(window + 1)
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



def calculate_bollinger_bands(data,select_mean =0, n=20,k1=2,k2=2):
    
    #mean = data.rolling(window=n).mean()
    #print('bollinger_ventana', n)

    if (select_mean == 0 ):
        mean = SMA(data, window = n)
    elif (select_mean == 1 ):
        mean = WMA(data, window = n)
    elif (select_mean == 2):
        mean = EMA(data, window = n)
    else: print('error mean')
    std = data.rolling(window=n).std()
    upper_band = mean + (k1*std)
    lower_band = mean - (k2*std)

    return mean, upper_band, lower_band


def fitness(gens):
    df = pd.read_csv('data/EURUSD_Candlestick_15_m_BID_01.01.2007-31.12.2007.csv')

    '''
    variable gens interpretacion
    gen 0 [k/upper]      valor k para la banda superior
    gen 1 [k/lower]      valor k para la banda inferior
    gen 2 [mean]         valor enteros de 0 a  2 para seleccionar una media
            0  media normal
            1  media ponderada
            2 media exponencial
    gen 3 [window Value] valor  de 20 a 200 para la ventana de la media
    '''

    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']

    middle, upper, lower = calculate_bollinger_bands(Close, select_mean= gens[2] ,n= gens[3], k1=gens[0],k2=gens[1])

    Upper = np.array(upper)
    Lower = np.array(lower)
    Close = np.array(Close)


    regreso_po, regreso_neg, usd = buy_sell(cierre=Close, superior=Upper, inferior=Lower, window_size= gens[3])

    returns_fit = returns_fitness(regreso_po, regreso_neg)

    return(returns_fit)




def main():
    population_size = int(input('introduzca población:  \n')) -1
    C = []
    #p_mut = 0.1 # se utiliza en otro punto afuera de este ciclo
    
    # initialize random population
    contador_w = 0
    while  (contador_w <= population_size): 
        g = list(np.zeros(4))
        x = random.uniform(1,3) #Valor aleatorio para el alpha de la banda superior
        y = random.uniform(1,3) #Valor aleatorio para la banda inferior
        g[0], g[1] = x, y
        g[2] = random.randint(0,2) #Selecciona el tipo de media a usar
        g[3] = random.randint(20,200) 
        C.append(Candidate(g,fitness(g)))

        contador_w += 1

    for i in C:
        print (i.fitness,i.genotype)



    # best fitness & survival
    ##C.sort(key=lambda x: x.fitness, reverse=True) # ordenar por fitness
    ##C = C[:population_size]



if __name__ == "__main__":
    main()
