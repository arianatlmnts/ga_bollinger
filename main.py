import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def returns_fitness(returns):
    neg_returns = len(list(filter(lambda x: (x < 0), returns)))
    pos_returns = len(list(filter(lambda x: (x > 0), returns)))

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
    window_sum =[]
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

def EMA (close,window = 20,alfa = 0.06):
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




def operaciones(close,open, middle, upper,lower,n):


    ''' Variable para operaciones'''
    eur = 0
    usd_buy = 100
    usd_sell = 0
    eur_buy = 0
    eur_sell = 0
    Realizar_LongTerm = True
    Realizar_ShortTerm = True

    '''Operaciones y returns'''
    returns = []
    contador = 0

    ''' Perdidas'''
    perdidas = 0
    perdidas_longTerm = 0
    perdidas_shortTerm = 0
    no_trasac_perdida = 0

    ''' ganancias '''
    ganancias = 0
    ganancias_shortTerm = 0
    ganancias_longTerm = 0

    for i in range(n, len(close)):


        if i < len(close):
                '''********************************Sección de Long Term****************************'''

                if (close[i-1] <= lower[i-1]) and (close[i] > lower[i]) and Realizar_LongTerm:
                    Realizar_LongTerm = False
                    eur = usd_buy/open[i+1]  # cambiar el valor de close por el de open # Historico
                    contador += 1


                ''' Señal de cierre de la operación long term'''

                if (close[i] >= middle[i] and not(Realizar_LongTerm)): # cmabie el parametro de upper por el parametro de media movil
                    Realizar_LongTerm = True
                    valor_actual = (eur*open[i+1])
                    usd_sell = valor_actual

                    returns.append(usd_sell - usd_buy)


                    if returns[-1] <= 0:
                        print('[Short_term] Operacion perdida: ', usd_sell, '-' ,usd_buy, 'resultado', usd_sell-usd_buy)
                        usd_buy = usd_sell
                        no_trasac_perdida += 1
                        perdidas += returns[-1]
                        perdidas_longTerm += returns[-1]

                    elif returns[-1] > 0:
                        usd_buy = usd_sell
                        ganancias += returns[-1]
                        ganancias_longTerm += returns[-1]
                    else:
                        print('error')

                ''' Funcion stop loss'''

                stop_loss_value = (usd_buy*(1-0.010)) #stop loss value es porcentaje maximo de perdida
                valor_actual = (eur*close[i])
                if (not(Realizar_LongTerm) and (valor_actual < stop_loss_value)):
                    Realizar_LongTerm = True
                    valor_actual = (eur*open[i+1])
                    returns.append(valor_actual- usd_buy)
                    print('Stopp loss longTerm', returns[-1])
                    usd_buy = valor_actual
                    no_trasac_perdida += 1
                    perdidas += returns[-1]
                    perdidas_longTerm += returns[-1]


                '''******************** sección de la estrategia Short term**********************'''
                # lo que tenga en dolares en ese momento lo cambio?? para conseguir euros???

                if (close[i-1] >= upper[i-1]) and (close[i] < upper[i]) and Realizar_ShortTerm:
                    Realizar_ShortTerm = False
                    valor_actual = open[i+1] #valor de apertura al siguiente día
                    eur_to_return = 100
                    eur_sell  = eur_to_return  * valor_actual # se comienza a trabajar con euros prestados del broker
                    contador += 1

                ''' Señal de cierre de la operación Short term'''

                if (close[i] < middle[i] and not(Realizar_ShortTerm)): # posible cabio de paramatros
                    Realizar_ShortTerm = True
                    valor_actual = (open[i+1])

                    eur_buy = valor_actual * eur_to_return


                    returns.append(eur_sell - eur_buy)


                    if returns[-1] <= 0:
                        print('operacion perdida: ', usd_sell, '-' ,usd_buy, 'resultado', usd_sell-usd_buy)
                        no_trasac_perdida += 1
                        perdidas += returns[-1]
                        perdidas_shortTerm  += returns[-1]

                    elif returns[-1] > 0:
                        ganancias += returns[-1]
                        ganancias_shortTerm += returns[-1]
                    else:
                        print('error')

                    '''[short term] Loss funtion'''

                    stop_loss_value = (eur_sell * (1 + 0.001)) #stop loss value es porcentaje maximo de perdida
                    valor_actual = (eur_to_return*close[i])

                    if (not(Realizar_ShortTerm) and (valor_actual > stop_loss_value)):
                        Realizar_ShortTerm = True
                        valor_actual = (open[i+1])
                        eur_buy = valor_actual * eur_to_return
                        print('entro_short')


                        returns.append(eur_sell - eur_buy)

                        no_trasac_perdida += 1
                        perdidas += returns[-1]











    print('\nvalor final de USD_BUY: ',usd_buy )
    print('N° transacciones con perdias', no_trasac_perdida)

    print('\nPerdidas acumulado [General]', perdidas)
    print('Perdidas acumulado [Short Term]', perdidas_shortTerm)
    print('Perdidas acumulado [long Term]', perdidas_longTerm)

    print('\nGanancias acumulado [General]', ganancias)
    print('Ganancias acumulado [Short_term]', ganancias_shortTerm)
    print('Ganancias acumulado [Long_term]', ganancias_longTerm)

    print('\nResultado [ganancias_general - perdias_general]', ganancias + perdidas)


    print('\nN° transacciones realizadas', contador, '\n')
    return returns

def calculate_bollinger_bands(data,n=20,k1=2,k2=2):
    #ma = data.rolling(window=n).mean() # Genotipo los valores de mean
    #ma = SMA(data, window = n )
    #ma = WMA(data, window = n )
    ma = EMA(data, window = n)


    std = data.rolling(window=n).std()  # Genotipo
    upper_band = ma + (k1*std)  # Valores vaiables de las partes
    lower_band = ma - (k2*std) # Valores iniciales de una variable

    return ma, upper_band, lower_band


if __name__ == '__main__':
    df = pd.read_csv('data/EURUSD_Candlestick_1_m_BID_01.11.2006-30.11.2006.csv')

    #p = 200
    n = 50
    k1 = 2
    k2 = 2


    ''' En longterm vamos a operar sobre los dolares
    Short se pediran euros al broker '''


    Open = df['Open']
    High = df['High']
    Low = df['Low']
    Close = df['Close']

    middle, upper, lower = calculate_bollinger_bands(Close,n=n,k1=k1,k2=k2)
    #print(middle)

    ''' las señales de compra se determinan por el precio de cierre de una acción
    pero los precios donde se aplican los calculos de inversion se realizan con
    los valores de open ya que si, se genera la señal de cierre nuestra accion se
    ejecutara en un "open" de la siguiente vez que habra el mercado'''

    ''' El cierre de una long-term o short-term se realiza por con la MA con respecto al
    precio de cierre de una acción y no con la otra banda [segun otros papers...]'''


    returns_Longterm = operaciones(Close, Open,middle, upper, lower, n)

    #print('retornos del long term', returns_Longterm)

    returns_fit = returns_fitness(returns_Longterm)
    print('operación fitness', returns_fit)

    plt.figure(figsize=(10,6))
    plt.plot(Close)
    plt.plot(middle)
    plt.plot(upper)
    plt.plot(lower)
    plt.fill_between(Close.index, lower, upper, color='#ADCCFF', alpha='0.4')
    plt.show()
