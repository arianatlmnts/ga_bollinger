import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

'''  sección del geneotipo'''

class Candidate(object):
    def __init__(self, genotype, fitness):
        self.genotype = genotype
        self.fitness = fitness

    def mutation(self):
        p_mut = 10
        for i in range(0, 1): #mutacion para los primeros dos
            r = random.randint(0,100)
            if r <= p_mut: #condicion para que la mutacion no sea menor a 1 ni mayor a 3
                x = random.uniform(-1, 1)
                y = self.genotype[i] - x
                if( y < 1):
                    self.genotype[i] = 1
                elif( y > 3):
                    self.genotype[i] = 3
                else:
                    self.genotype[i] = y
        r = random.randint(0,100)
        if r <= p_mut: #mutacion para el tipo de media
            self.genotype[2] = random.randint(0, 2) #cambia la media por otra
        r = random.randint(0,100)
        if r <= p_mut: #mutacion para la ventana
            self.genotype[3] = random.randint(20, 200) #Cambia la ventana actual por una aleatoria

            #x,y = random.sample(range(1, len(self.genotype)), 2)
            #self.genotype[x],self.genotype[y] = self.genotype[y],self.genotype[x]
        self.fitness = fitness(self.genotype)


def crossover(g1,g2):
    cross_val = random.randint(0,len(g1))

    c1 = g1[:cross_val] + g2[cross_val:]
    c2 = g2[:cross_val] + g1[cross_val:]

    return c1,c2



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
    pos_returns, neg_returns , usd = buy_sell(cierre=Close, superior=Upper, inferior=Lower, window_size= gens[3])


    return (pos_returns / (neg_returns+pos_returns))


def calculate_bollinger_bands(data,select_mean =0, n=20,k1=2,k2=2):

    #mean = data.rolling(window=n).mean()
    #print('bollinger_ventana', n)

    if (select_mean == 0 ):
        mean = SMA(data, window = n)
    elif (select_mean == 1 ):
        mean = WMA(data, window = n)
    elif (select_mean == 2):
        mean = EMA(data, window = n)
    else: print('Error: ha eligido un valor no valido [error mean]')
    std = data.rolling(window=n).std()
    upper_band = mean + (k1*std)
    lower_band = mean - (k2*std)

    return mean, upper_band, lower_band


def main():
    population_size = int(input('tamaño de población: '))
    generations = int(input('número de generaciones: '))
    C = []

    best_fitness = [] # para graficar
    average_fitness = [] # para graficar

    # initialize random population
    for i in range(population_size):
        g = list(np.zeros(4))
        x = random.uniform(1,3) #Valor aleatorio para el alpha de la banda superior
        y = random.uniform(1,3) #Valor aleatorio para la banda inferior
        g[0], g[1] = x, y
        g[2] = random.randint(0,2) #Selecciona el tipo de media a usar
        g[3] = random.randint(20,200) #Selecciona la ventana a usar
        C.append(Candidate(g,fitness(g)))

    counter = 0
    while counter < generations:
        counter +=1

        C.sort(key=lambda x: x.fitness, reverse=True)
        C = C[:population_size]                         # mantener el tamaño de población
        best_fitness.append(C[0].fitness)               # mejor fitness por generación

        print('\n\ngeneración%2d:\n'%(counter))

        for i in C:
            print (i.fitness,i.genotype)

        print('mejor fitness: ', C[0].fitness)

        ## Cruzamiento
        p1 = C[0].genotype
        p2 = C[1].genotype

        print('padres: %s y %s'%(p1,p2))

        c1, c2 = crossover(p1,p2)

        print('hijos: %s y %s'%(c1,c2))

        offspring = crossover(p1,p2)
        for child in offspring:
            c = Candidate(child,fitness(child))
            c.mutation()
            C.append(c)


if __name__ == "__main__":
    main()
