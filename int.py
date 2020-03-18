#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
print('pandas imported')


# In[2]:


def calculate_bollinger_bands(data,n,k):

    ma = data.rolling(window=n).mean()
    std = data.rolling(window=n).std()

    upper_band = ma + (k*std)
    lower_band = ma - (k*std)

    plt.figure(figsize=(20,6))
    plt.plot(data)
    plt.plot(ma)
    plt.plot(upper_band)
    plt.plot(lower_band)

    plt.show()
    return upper_band, lower_band


# In[10]:


if __name__ == '__main__':
    df = pd.read_csv('EURUSD_Candlestick_5_m_BID_01.05.2006-31.08.2006.csv')
    dias = 200
    inicio = dias
    fin = dias*2
    closing_price = df['Close']
    window_size = 20
    k = 1.5

    superior, inferior = calculate_bollinger_bands(closing_price,window_size,k)


# In[11]:


superior = np.array(superior)
inferior = np.array(inferior)
cierre = np.array(closing_price)

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
        
if( dolares == 0):
    dolares = eur*cierre[-1]
    dolar.append(dolares)


print(dolar)
#print(euros)
print(regreso_po)
print(regreso_neg)


# In[ ]:




