import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split

from neuralnetwork.util import vectorized_result


def read_data():
    """
    Read file
    :return: pd.DataFrame
    """
    data = pd.read_csv('../data/auto.csv')
    return data


def caterogy_to_numeric(data):
    """
    String data to numeric
    :param data: pd.DataFrame
    :return:
    """
    le_region = preprocessing.LabelEncoder()
    data['REGION'] = le_region.fit_transform(data['REGION'])

    le_cat = preprocessing.LabelEncoder()
    data['CATEGORY'] = le_cat.fit_transform(data['CATEGORY']) + data['REGION'].max() + 1

    le_car = preprocessing.LabelEncoder()
    data['CAR'] = le_car.fit_transform(data['CAR']) + data['CATEGORY'].max() + 1

    le_name = preprocessing.LabelEncoder()
    data['EMPLOYEE'] = le_name.fit_transform(data['EMPLOYEE'])
    return


def vectorize(line, n):
    """
    Every element is zero except three points
    ex> [1,3,5] -> [1,0,1,0,1,0,0]
    :param line: pd.Series
    :param n: int
    :return: np.array
    """
    vec = np.zeros(n)
    vec[line['REGION']] = 1
    vec[line['CATEGORY']] = 1
    vec[line['CAR']] = 1
    return vec


def vectorize_X(data):
    """
    Vectorize X
    :param data: pd.DataFrame
    :return: pd.DataFrame
    """
    n = data['CAR'].max() + 1
    Xdata = pd.DataFrame(np.zeros((data.shape[0], n)))
    for i, line in data.iterrows():
        Xdata.loc[i] = vectorize(line, n)
    return Xdata


def load_data():
    """
    Read, split data
    :return: pd.DataFrame
    """
    data = read_data()
    caterogy_to_numeric(data)

    x = vectorize_X(data)
    y = data['EMPLOYEE']

    X_train, X_temp, y_train, y_temp = train_test_split(x, y,
                                                        test_size=0.4,
                                                        random_state=1000)

    X_valid, X_test, y_valid, y_test = train_test_split(X_temp, y_temp,
                                                        test_size=1.0 / 2.0,
                                                        random_state=1000)

    training_data = (X_train, y_train)
    validation_data = (X_valid, y_valid)
    test_data = (X_test, y_test)

    return training_data, validation_data, test_data, y.unique().size


def load_data_wrapper():
    """
    Preprocess data
    :return:
    """
    tr_d, va_d, te_d, n_bin = load_data()

    training_inputs = [x.as_matrix()[:, None] for i, x in tr_d[0].iterrows()]
    training_results = [vectorized_result(y, n_bin) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = [x.as_matrix()[:, None] for i, x in va_d[0].iterrows()]
    validation_results = [vectorized_result(y, n_bin) for y in va_d[1]]
    validation_data = list(zip(validation_inputs, validation_results))

    test_inputs = [x.as_matrix()[:, None] for i, x in te_d[0].iterrows()]
    test_results = [vectorized_result(y, n_bin) for y in te_d[1]]
    test_data = list(zip(test_inputs, test_results))

    # print("Training size = %d, Validation = %d" % (len(training_data), len(validation_data)))
    print("Training size = %d, Validation = %d, Test = %d" % (len(training_data), len(validation_data), len(test_data)))

    return training_data, validation_data, test_data, n_bin


if __name__ == "__main__":
    training_data, validation_data, test_data, n_bin = load_data_wrapper()
