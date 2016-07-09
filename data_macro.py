
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime


# In[2]:

start = datetime.datetime(1960, 1, 1)
end = datetime.datetime.today()#datetime(2013, 1, 27)


# In[3]:

fredlist = ["GDPC1",  # GDP
            "UNRATE",  # Unemployment rate
            "CPIAUCSL", "PPIACO", "USSTHPI",  # CPI, PPI, HPI
            "DEXJPUS", "DEXUSEU", "DEXCHUS",  # Foreign exchange
            "DTB3", "GS10", "DGS30",  # Treasury yield
            "CSCICP03USM665S",  # Consumer survey
            "DAUTOSA", "RRSFS",  # Retail auto sales, Retail and Food Services
            "HOUST",  # New housing
            "ISRATIO",  # Inventory/sales manufacturer
           ]

fred = web.DataReader(fredlist, "fred", start, end)
print(fred.shape)


# In[4]:

yahoolist = ["^GSPC", "^FTSE", "^GDAXI", "^N225", "^HSI",  # Index
             "GLD", "OIL", "SLV",  # Commodities
            ]

yahoo = pd.DataFrame()
for macro in yahoolist:
    df = web.DataReader(macro, 'yahoo', start, end)[['Volume','Adj Close']]
    df.columns = [macro + " " + col for col in df.columns]
    yahoo = pd.concat([yahoo, df], axis=1)
print(yahoo.shape)

yahoo['1M Return(%)'] = yahoo['^GSPC Adj Close'].pct_change(-20)*-100.0


# In[5]:

df = pd.concat([fred, yahoo], axis=1)
df_fill = df.resample('1B').last().ffill()
df_fill.dropna(axis=0, inplace=True)

drop_list = ['^FTSE Volume', '^GDAXI Volume', '^N225 Volume', '^HSI Volume']
df_fill.drop(drop_list, 1, inplace=True)

print(pd.isnull(df_fill).sum(axis=1).sum())
print(df_fill.shape)


# In[6]:

ret_range = np.arange(-30, 31, 10)
ret_label = np.arange(ret_range.shape[0]-1)
df_fill['1M Return(%)'] = pd.cut(df_fill['1M Return(%)'], ret_range, labels=ret_label)
df_fill.to_csv('temp.csv')


# In[7]:

# df_fill['test'] = x
# yahoo['^GDAXI Volume'].dropna().head()
# (df_fill == 0).sum(axis=0)
# df_fill.drop(['^FTSE Volume', '^GDAXI Volume', '^N225 Volume', '^HSI Volume'], 1, inplace=True).shape


# In[8]:

X = df_fill.drop('1M Return(%)', axis=1)
Y = df_fill['1M Return(%)']
print(X.shape, Y.shape)


# In[9]:

N = 28

x = [X.values[i:i+N,:] for i in np.arange(X.shape[0]-N+1)]
y = Y.values[N-1:]

print(len(x), len(y))


# In[10]:

ratio = np.array([5,1,1])
ratio = np.cumsum(ratio)
bound = [0] + [int(len(x) * ratio[i] / ratio[-1]) for i in range(len(ratio))]
bound


# In[11]:

training_data = zip(x[bound[0]:bound[1]], y[bound[0]:bound[1]])
validation_data = zip(x[bound[1]:bound[2]], y[bound[1]:bound[2]])
test_data = zip(x[bound[2]:bound[3]], y[bound[2]:bound[3]])

