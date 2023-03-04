import warnings

from pandas import read_csv
from keras.layers import Dense
from keras.models import Sequential
from matplotlib import pyplot

TRAINING_PERCENTAGE = 0.83
TESTING_PERCENTAGE = 1 - TRAINING_PERCENTAGE
NUMBER_OF_PREVIOUS_DATA_POINTS = 3
LENGTH_DATA_SET = 0
TRAINING_SET_LENGTH = 0
TESTING_SET_LENGTH = 0


def load_data():
    data_set_frame = read_csv('usdinr_d_2.csv', header=0, squeeze=True)
    column_headers = data_set_frame.columns.values.tolist()
    currency_index = column_headers.index('Close')

    data_file = read_csv("usdinr_d_2.csv", usecols=[currency_index], engine='python')
    load_data = []
    for data_point in data_file.values.tolist():
        load_data.append(data_point[0])
    global LENGTH_DATA_SET
    LENGTH_DATA_SET = len(load_data)
    return load_data


def split_train_test(load_data, training_percentage, testing_percentage):
    global TRAINING_SET_LENGTH, TESTING_SET_LENGTH
    TRAINING_SET_LENGTH = int(LENGTH_DATA_SET * training_percentage)
    TESTING_SET_LENGTH = LENGTH_DATA_SET - TRAINING_SET_LENGTH
    training_set, testing_set = load_data[0:TRAINING_SET_LENGTH], load_data[TRAINING_SET_LENGTH:LENGTH_DATA_SET]
    return training_set, testing_set


def modify_data_set_rnn(training_set, testing_set):
    train_actual = []
    train_predict = []
    for interval in range(len(training_set) - NUMBER_OF_PREVIOUS_DATA_POINTS - 1):
        train_actual.append(training_set[interval: interval + NUMBER_OF_PREVIOUS_DATA_POINTS])
        train_predict.append(training_set[interval + NUMBER_OF_PREVIOUS_DATA_POINTS])

    test_actual = []
    test_predict = []

    for interval in range(len(testing_set) - NUMBER_OF_PREVIOUS_DATA_POINTS - 1):
        test_actual.append(testing_set[interval: interval + NUMBER_OF_PREVIOUS_DATA_POINTS])
        test_predict.append(testing_set[interval + NUMBER_OF_PREVIOUS_DATA_POINTS])

    return train_actual, train_predict, test_actual, test_predict


def build_rnn(train_actual, train_predict):
    recurrent_neural_network = Sequential()

    recurrent_neural_network.add(Dense(12, input_dim=NUMBER_OF_PREVIOUS_DATA_POINTS, activation="relu"))
    recurrent_neural_network.add(Dense(8, activation="relu"))
    recurrent_neural_network.add(Dense(1))

    recurrent_neural_network.compile(loss='mean_squared_error', optimizer='adam')
    recurrent_neural_network.fit(train_actual, train_predict, epochs=50, batch_size=2, verbose=2)

    return recurrent_neural_network


def predict_rnn(recurrent_neural_network, train_actual, test_actual):
    training_predict, testing_predict = recurrent_neural_network.predict(
        train_actual), recurrent_neural_network.predict(test_actual)

    print('\t The prediction for the next day:', testing_predict[-1])
    return training_predict, testing_predict


def evaluate_performance_rnn(recurrent_neural_network, test_actual, test_predict):
    # mse_training = recurrent_neural_network.evaluate(train_actual, train_predict, verbose=0)
    mse_testing = recurrent_neural_network.evaluate(test_actual, test_predict, verbose=0)
    print('\t Performance Evaluation: Testing Mean Square Error:', mse_testing)


def rnn_graph_plot(load_data, training_predict, testing_predict, file_name):
    training_data_trend = [None] * LENGTH_DATA_SET
    testing_data_trend = [None] * LENGTH_DATA_SET

    training_data_trend[NUMBER_OF_PREVIOUS_DATA_POINTS:len(training_predict) + NUMBER_OF_PREVIOUS_DATA_POINTS] = list(training_predict[:, 0])
    testing_data_trend[NUMBER_OF_PREVIOUS_DATA_POINTS - 1:len(training_predict) + NUMBER_OF_PREVIOUS_DATA_POINTS] = list(testing_predict[:, 0])

    actual = pyplot.plot(load_data[int(TRAINING_PERCENTAGE * LENGTH_DATA_SET):], label="Actual data points", color="blue")
    testing = pyplot.plot(testing_data_trend, label="Testing prediction", color="red")

    pyplot.ylabel('Currency Values for 1 USD')
    pyplot.xlabel('Number of Days')
    pyplot.title("USD/INR' : Actual vs Predicted using RNN")

    pyplot.legend()

    pyplot.savefig(file_name)
    pyplot.clf()


def rnn_model():

    data = load_data()

    training_set, testing_set = split_train_test(data, TRAINING_PERCENTAGE, TESTING_PERCENTAGE)
    train_actual, train_predict, test_actual, test_predict = modify_data_set_rnn(training_set, testing_set)

    rnn = build_rnn(train_actual, train_predict)

    training_predict, testing_predict = predict_rnn(rnn, train_actual, test_actual)

    evaluate_performance_rnn(rnn, test_actual, test_predict)

    print('RNN Graph has been generated in PNG format.')
    rnn_graph_plot(data, training_predict, testing_predict, "rnn_tp.png")

    return training_predict, testing_predict


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    rnn_model()  # setting the entry point