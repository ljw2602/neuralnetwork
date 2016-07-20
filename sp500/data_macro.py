import pandas as pd
import numpy as np
import pandas_datareader.data as web
import datetime
from sklearn.cross_validation import train_test_split


def load_from_fred(start, end):
    """
    Load data from FRED (St. Louis Fed) using Pandas DataReader
    :param start: datetime
    :param end: datetime
    :return: pd.DataFrame
    """
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
    """
    Load data from Yahoo Finance using Pandas DataReader
    :param start: datetime
    :param end: datetime
    :return: pd.DataFrame
    """
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
    """
    Concatenate two dataframes, resample every business day, forward-fill empty cells
    :param fred: pd.DataFrame
    :param yahoo: pd.DataFrame
    :return: pd.DataFrame
    """
    df = pd.concat([fred, yahoo], axis=1)
    df_fill = df.resample('1B').ffill()
    df_fill.dropna(axis=0, inplace=True)
    return df_fill


def remove_some_data(df):
    """
    Remove unused data columns
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    drop_list = ['^FTSE Volume', '^GDAXI Volume', '^N225 Volume', '^HSI Volume']
    df.drop(drop_list, 1, inplace=True)
    return df


def categorize_y(df):
    """
    1 month return (%) is translated into categorical variable
    Boundaries are set at [-40, -5, -1, 1, 5, 20](%)
    :param df: pd.DataFrame
    :return: pd.DataFrame
    """
    ret_range = np.array([-40, -5, -1, 1, 5, 20])
    ret_label = np.arange(ret_range.shape[0]-1)
    df['1M Return(%)'] = pd.cut(df['1M Return(%)'], ret_range, labels=ret_label)

    assert not df.isnull().any().any(), "change ret_range"

    return df


def serialize(df, n_window):
    """
    Serialize dataframe with lookback period of n_window
    :param df: pd.DataFrame
    :param n_window: int
    :return: pd.DataFrame, pd.DataFrame
    """
    from sklearn.preprocessing import normalize
    X = df.drop('1M Return(%)', axis=1)
    X = normalize(X, axis=0)
    Y = df['1M Return(%)']

    x = [X[i:i + n_window, :] for i in np.arange(X.shape[0] - n_window + 1)]
    y = Y.values[n_window-1:]
    return x, y


def divide(x, y):
    """
    Split data into train/validation/test = 5/1/1
    :param x: list of pd.DataFrame
    :param y: list of int
    :return:
    """
    ratio = np.array([5, 1, 1])
    y = list(y)
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = ratio[2]/ratio.sum())
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size = ratio[1]/ratio[:2].sum())

    training_data = (X_train, y_train)
    validation_data = (X_valid, y_valid)
    test_data = (X_test, y_test)

    return training_data, validation_data, test_data


def load_data(n_window, start = datetime.datetime(1960, 1, 1), end = datetime.datetime.today()):
    """
    Prepare data
    :param n_window: int
    :param start: datatime
    :param end: datatime
    :return:
    """
    fred = load_from_fred(start, end)
    yahoo = load_from_yahoo(start, end)

    df = concat_data(fred, yahoo)
    df = remove_some_data(df)
    df = categorize_y(df)
    x, y = serialize(df, n_window)
    training_data, validation_data, test_data = divide(x, y)

    return training_data, validation_data, test_data


def load_data_wrapper():
    """
    Preprocess data
    :return:
    """
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
    """
    Transform categorical data into vector form
    ex> 5 -> [0,0,0,0,1,0,0]
    :param j: int
    :return: np.Array
    """
    e = np.zeros((5, 1))
    e[j] = 1.0
    return e


if __name__ == "__main__":
    training_data, validation_data, test_data = load_data_wrapper()
