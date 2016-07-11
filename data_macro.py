import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from sklearn.cross_validation import train_test_split


def load_from_fred(start, end):
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
    return fred


def load_from_yahoo(start, end):
    yahoolist = ["^GSPC", "^FTSE", "^GDAXI", "^N225", "^HSI",  # Index
                 "GLD", "OIL", "SLV",  # Commodities
                ]
    yahoo = pd.DataFrame()
    for macro in yahoolist:
        df = web.DataReader(macro, 'yahoo', start, end)[['Volume','Adj Close']]
        df.columns = [macro + " " + col for col in df.columns]
        yahoo = pd.concat([yahoo, df], axis=1)
    yahoo['1M Return(%)'] = yahoo['^GSPC Adj Close'].pct_change(-20)*-100.0
    return yahoo


def concat_data(fred, yahoo):
    df = pd.concat([fred, yahoo], axis=1)
    df_fill = df.resample('1B').ffill()
    df_fill.dropna(axis=0, inplace=True)
    return df_fill


def remove_some_data(df):
    drop_list = ['^FTSE Volume', '^GDAXI Volume', '^N225 Volume', '^HSI Volume']
    df.drop(drop_list, 1, inplace=True)
    return df


def categorize_y(df):
    ret_range = np.array([-40, -5, -1, 1, 5, 20])
    ret_label = np.arange(ret_range.shape[0]-1)
    df['1M Return(%)'] = pd.cut(df['1M Return(%)'], ret_range, labels=ret_label)
    # df['1M Return(%)'], bins = pd.qcut(df['1M Return(%)'], 10, labels=range(10), retbins=True)
    # print("Bins: "); print(bins)
    assert not df.isnull().any().any(), "change ret_range"
    df.to_csv('temp.csv')
    return df


def serialize(df, n_window):
    from sklearn.preprocessing import normalize
    X = df.drop('1M Return(%)', axis=1)
    X = normalize(X, axis=0)
    Y = df['1M Return(%)']

    x = [X[i:i + n_window, :] for i in np.arange(X.shape[0] - n_window + 1)]
    y = Y.values[n_window-1:]
    return x, y


def divide(x, y):
    ratio = np.array([5, 1, 1])
    y = list(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = ratio[2]/ratio.sum())
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = ratio[1]/ratio[:2].sum())

    training_data = (X_train, y_train)
    validation_data = (X_valid, y_valid)
    test_data = (X_test, y_test)

    # ratio = np.array([5,1,1])
    # ratio = np.cumsum(ratio)
    # bound = [0] + [int(len(x) * ratio[i] / ratio[-1]) for i in range(len(ratio))]

    # training_data = (x[bound[0]:bound[1]], y[bound[0]:bound[1]])
    # validation_data = (x[bound[1]:bound[2]], y[bound[1]:bound[2]])
    # test_data = (x[bound[2]:bound[3]], y[bound[2]:bound[3]])

    return training_data, validation_data, test_data


def load_data(n_window, start = datetime.datetime(1960, 1, 1), end = datetime.datetime.today()):
    fred = load_from_fred(start, end)
    yahoo = load_from_yahoo(start, end)

    df = concat_data(fred, yahoo)
    df = remove_some_data(df)
    df = categorize_y(df)
    x, y = serialize(df, n_window)
    training_data, validation_data, test_data = divide(x, y)

    return training_data, validation_data, test_data


def load_data_wrapper():
    n_window = 28  # rolling window
    n_macro = 28  # number of macro
    n = n_window*n_macro

    tr_d, va_d, te_d = load_data(n_window)

    training_inputs = [np.reshape(x, (n, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [np.reshape(x, (n, 1)) for x in va_d[0]]
    validation_results = [vectorized_result(y) for y in va_d[1]]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [np.reshape(x, (n, 1)) for x in te_d[0]]
    test_results = [vectorized_result(y) for y in te_d[1]]
    test_data = list(zip(test_inputs, test_results))

    print("Training size = %d, Validation = %d, Test = %d" %(len(training_data), len(validation_data), len(test_data)))

    return (training_data, validation_data, test_data)


def vectorized_result(j):
    e = np.zeros((5, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
