import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import matplotlib.patches as mpatches
import time as time
import glob
'''  sección del geneotipo'''

class Candidate(object):
    def __init__(self, genotype, fitness):
        self.genotype = genotype
        self.usd = 0
        self.fitness, self.usd = fitness
    def mutation(self,df):
        p_mut = 30
        for i in range(0, 2): #mutacion para los primeros dos
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
        r = random.randint(0,100)
        if r <= p_mut:
            self.genotype[4] = random.randint(1,10)
        r = random.randint(0,100)
        if r <= p_mut:
            self.genotype[5] = random.uniform(0.001,0.010)
        self.fitness, self.usd = fitness(self.genotype, df = df)



def crossover(g1,g2):
    cross_val = random.randint(0,len(g1))

    c1 = g1[:cross_val] + g2[cross_val:]
    c2 = g2[:cross_val] + g1[cross_val:]

    return c1,c2



''' funciones para el calculo de medias'''
#Media simple
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

    return mean
#Media ponderada
def WMA (close, window = 20):

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


    return mean
#Media exponencial
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

def EMA2(close, window = 20):
    alpha = 2/(window+1)
    mean = []
    for i in range(len(close)):
        if i == 0:
            mean.append(close[0])
        if i > 0:
            componente_2 = mean[-1]*(1-alpha)
            componente_1 = close[i]*alpha
            suma = componente_1 + componente_2
            mean.append (suma)
        
    return  mean   



def stop_loss(close, eur, compra, i,epsilon = 0.010, transaccion = False):
    condition = False
    if(transaccion):
        usd_buy = close[compra[-1]]*eur
        stop_loss_value = (usd_buy*(1-epsilon)) #stop loss value es porcentaje maximo de perdida
        valor_actual = (eur*close[i])
        if ( valor_actual < stop_loss_value):
            condition = True
    return condition


def MDD(data):
    index_max = max(range(len(data)), key=data.__getitem__)
    index_min = min(range(index_max,len(data)), key = data.__getitem__)
    mdd = (data[index_max]-data[index_min])
    return mdd

def buy_sell(cierre,superior,inferior,window_size, cont, epsilon):
    costo_tran = 0
    compra = []
    venta = []

    dolares = 100
    eur = 0

    dolar = [100]
    euros = []
    regreso_po = 0
    regreso_neg = 0
    long_term = False #Para indicar que se tiene una operación en proceso(Para Stop loss)

    for i in range(int(window_size), superior.shape[0]):
        if( cierre[i-1] < inferior[i-1] and cierre[i] > inferior[i] and dolares > 0 and cont > 0):
            eur = dolares/cierre[i]
            #costo_tran = 0.01*cierre[i]*eur
            dolares = 0
            euros.append(eur)
            compra.append(i)
            long_term = True
            
        if ( (cierre[i-1] > superior[i-1] and cierre[i] < superior[i] and eur > 0 and cont > 0) or
        stop_loss(close= cierre, eur = eur, compra = compra, i = i, transaccion = long_term,  epsilon = epsilon) and cont > 0):

            dolares = eur*cierre[i] - costo_tran
            #cont -= 1
            if( (dolares - dolar[-1]) > 0 ):
                regreso_po += 1
            elif( (dolares - dolar[-1]) < 0):
                regreso_neg += 1
            dolar.append(dolares)
            eur = 0
            venta.append(i)
            long_term = False

    if(dolares == 0):
        dolares = eur*cierre[-1]
        dolar.append(dolares)

    return regreso_po, regreso_neg, dolares


def stop_loss_long(val_opt, valor,epsilon,close_val):
    condition = False
    stop_loss_value = val_opt*(1-epsilon)
    valor_actual    = valor*(close_val)
    if(valor_actual < stop_loss_value):
        condition = True
    return condition

def stop_loss_short(val_opt, valor, epsilon,close_val):
    condition = False
    stopp_loss_value = valor * (1 + epsilon)
    valor_actual = val_opt/close_val 
    if(valor_actual < stopp_loss_value):
        condition = True
    return condition


def longTerm(val_opt, price_open, close_val = 0,  act_open = False):
    return_val = 0
    if act_open:
        eur = val_opt/price_open
        return_val = eur
    elif not act_open:
        dollar = (close_val*price_open)-val_opt
        return_val = dollar 
    return return_val

def shortTerm(val_opt, price_open, close_val = 0, act_open = False):
    return_val = 0
    if act_open:
        dollar = val_opt*price_open #gregar comision despues
        return_val = dollar
    elif not act_open:
        dollar_return = close_val-(val_opt*price_open)
        return_val = dollar_return
    return return_val

        

        

def options(val_option,data_close,data_open,bol_up,bol_down,epsilon = 0.001):
    '''Solo se puede tener una posicion baierta a la vez, y tambien es cerrada o abierta a la vez
        estas no pueden coexistir al mismo tiempo'''

    open_position  = False
    operation      = ''
    shortTerm_buy  = []
    shortTerm_sell = []
    longTerm_buy   = []
    longTerm_sell  = []

    for i in range(10,len (data_close)):
        
        #si no hay trnasacciones abrir una (long Term o short Term)
        if not(open_position):
            if data_close[i]>bol_down[i] and data_close[i-1]<=bol_down[i-1]: # condicion_longterm   
                open_position = True
                operation = 'longTerm'
                longTerm_buy.append(longTerm(val_opt= val_option, price_open= data_close[i], #cambie por error de lectura en open  data_open[i+1]
                act_open = open_position ))

            elif data_close[i]<bol_up[i] and data_close[i-1]>=bol_up[i-1]:
                open_position = True
                operation = 'shortTerm'
                shortTerm_sell.append(shortTerm(val_opt = val_option,price_open = data_close[i],
                act_open = open_position))
        
        #si hay una seccion cerrar una la transaccion en progreso
        if (open_position):
            #print(longTerm_buy)
            if(longTerm_buy != []):

                if ((data_close[i] > bol_up[i] and data_close[i-1]<= bol_up[i-1]) or stop_loss_long(val_opt = val_option,
                valor = longTerm_buy[-1], epsilon = epsilon, close_val = data_close[i]) ) and operation == 'longTerm':

                    open_position = False
                    longTerm_sell.append(longTerm(val_opt= val_option, price_open = data_close[i], #cambie por error de lectura en open  data_open[i+1]
                    close_val = longTerm_buy[-1] ,act_open= open_position))

            if shortTerm_sell != []:

                if ((data_close[i] < bol_down[i] and data_close[i-1] >= bol_down[i-1]) or stop_loss_short(val_opt = val_option,
                valor = shortTerm_sell[-1],epsilon = epsilon , close_val = data_close[i])) and operation =='shortTerm':
                    open_position = False
                    shortTerm_buy.append(shortTerm(val_opt = val_option, price_open = data_close[i],    #cambie por error de lectura en open  data_open[i+1]
                    close_val = shortTerm_sell[-1], act_open = open_position))

    
    positive_returns = sum( i for i in longTerm_sell if i > 0)
    negative_returns = sum( i for i in longTerm_sell if i < 0)
    positive_returns += sum( i for i in shortTerm_buy if i > 0)
    negative_returns += sum( i for i in shortTerm_buy if i < 0)

    profit = positive_returns + negative_returns

    return  positive_returns, negative_returns, profit


def fitness(gens,df):
    '''
    variable gens interpretacion
    gen 0 [k/upper]      valor k para la banda superior
    gen 1 [k/lower]      valor k para la banda inferior
    gen 2 [mean]         valor enteros de 0 a  2 para seleccionar una media
            0  media normal
            1  media ponderada
            2 media exponencial
    gen 3 [window Value] valor  de 20 a 200 para la ventana de la media
    gen 5 [valor Stop/loss] Valor entre 0.001 y 0.01 para determinar cuanto se puede perser

    '''
    Close = df['Close']
    Open  = df['Open']
    middle, upper, lower = calculate_bollinger_bands(data = Close,
                                                     select_mean = int(gens[2]),
                                                     n = int(gens[3]),
                                                     k1 = gens[0],
                                                     k2 = gens[1])

    Upper = np.array(upper)
    Lower = np.array(lower)
    Close = np.array(Close)
    '''
    pos_returns, neg_returns , usd = buy_sell(cierre=Close, 
                                              superior=Upper, 
                                              inferior=Lower, 
                                              window_size= gens[3], 
                                              cont = gens[4],
                                              epsilon = gens[5])
    '''
    pos_returns, neg_returns, usd = options(val_option = 100, data_close= Close, data_open= Open,
                                             bol_up = Upper, bol_down = Lower,epsilon= gens[5]) 
    #print(neg_returns,pos_returns, usd )

    try:
      #return (pos_returns / (neg_returns+pos_returns), usd)
      
      mdd = MDD(Close)
      return ((pos_returns+neg_returns)/mdd, usd)
    except ZeroDivisionError:
      return -1


def calculate_bollinger_bands(data, select_mean, n, k1, k2):

    if (select_mean == 0 ):
      mean = SMA(close = data, window = n)
    elif (select_mean == 1 ):
      mean = WMA(close = data, window = n)
    elif (select_mean == 2):
      mean = EMA2(close = data, window = n)

    std = data.rolling(window=n).std()
    upper_band = mean + (k1*std)
    lower_band = mean - (k2*std)
    return mean, upper_band, lower_band



def graficar(select_mean, n, k1, k2, best, average,path):
    df = pd.read_csv(path)
    data = df['Close']
    if (select_mean == 0 ):
        mean = SMA(data, window = n)
    elif (select_mean == 1 ):
        mean = WMA(data, window = n)
    elif (select_mean == 2):
        mean = EMA(data, window = n)

    std = data.rolling(window=n).std()
    upper_band = mean + (k1*std)
    lower_band = mean - (k2*std)



    fig, (ax1, ax2) = plt.subplots(2)

    ax1.set_title('Bandas')
    ax1.plot(data, label='Datos')
    ax1.plot(mean, label= 'Media')
    ax1.plot(upper_band, label = 'Banda superios')
    ax1.plot(lower_band, label = 'Banda inferior')
    plt.ylabel('Dolar')
    plt.xlabel('Periodo')
    
    ax2.set_title('Aptitud por generación')
    ax2.plot(best, label='Aptitud Mayor')
    ax2.plot(average, label='Aptitud Promedio')
    plt.ylabel('Fitness')
    plt.xlabel('Generacion')
    plt.legend()

    plt.show()



def bandasBG(path_file):
    #population_size = int(input('tamaño de población: '))
    #generations = int(input('número de generaciones: '))
    
    print('\nRuta del archivo: \n', path_file,'\n')
    df = pd.read_csv(path_file ,sep=r'\s*,\s*',header=0, encoding='ascii', engine='python')
    
    start_time = time.time()

    population_size = 10
    generations = 5
    C = []

    best_fitness = [] # para graficar
    average_fitness = [] # para graficar

    # initialize random population
    for i in range(population_size):
        g = list(np.zeros(6))
        x = random.uniform(1,3) #Valor aleatorio para el alpha de la banda superior
        y = random.uniform(1,3) #Valor aleatorio para la banda inferior
        g[0], g[1] = x, y
        g[2] = random.randint(0,2) #Selecciona el tipo de media a usar
        #g[2] = 2 #Selecciona el tipo de media a usar
        g[3] = random.randint(20,200) #Selecciona la ventana a usar
        g[4] = random.randint(1,100) #numero de transacciones
        g[5] = random.uniform(0.001,0.010)
        C.append(Candidate(g,fitness(g, df=df)))

    counter = 0
    gene = []
    while counter < generations:
        #incremento de generacion
        counter +=1
        gene.append(counter)

        C.sort(key=lambda x: x.fitness, reverse=True)
        C = C[:population_size]                         # mantener el tamaño de población
        best_fitness.append(C[0].fitness)               # mejor fitness por generación

        total_sum = 0
        for c in C:
            total_sum += c.fitness
        average_fitness.append(total_sum/population_size)


        ## Cruzamiento
        p1 = C[0].genotype
        p2 = C[1].genotype
        offspring = crossover(p1,p2)

        #Reemplazo de individuos
        for child in offspring:
            c = Candidate(child,fitness(child,df = df))
            c.mutation(df = df)
            C.append(c)
        print('generación: ', counter)

    # Visualizacion

    C.sort(key = lambda x: x.fitness, reverse = True)

    print('\nMejor individuo creado:\n')
    print('Profit: ', C[0].usd)
    print('Datos del genotipo: ', C[0].genotype)

    print('\nTiempo de ejecucion: ', time.time()-start_time)

    '''
    graficar(k1 = mejor_ind[0],
             k2 = mejor_ind[1],
             select_mean = mejor_ind[2],
             n = mejor_ind[3],
             best = best_fitness,
             average = average_fitness, path = path_file)
    '''
def main ():

    'Solo poner el nombre de la carpeta en file y se ejecura en que se encuentren en el'
    data = glob.glob('series/1_min/training/*.csv')

    print (data)
    for i in data:
        bandasBG(i)


if __name__ == "__main__":
    main()

