import numpy as np
import matplotlib.pyplot as plt

def same_length(data):
    """ this function will make sure all matrix rows have the same length."""
    max_length = max(len(row) for row in data)
    for row in data:
        if len(row) < max_length:
            blank_space = max_length - len(row)
            row.extend([''] * blank_space)
    return data

def break_down_number(number):
    """ this is just so I can break down the length of the matrixes in matrix_manupilation function further down.
    So all the output arrays are a 1 by n matrix. I would like to change that to a 2D matrix and this is how
    I thought is the best way to do it. Specially since we have different datas and each array will have different
    length arrays, I needed a seperate function to help me create that 2D array."""
    if number < 2:
        return [number]  # return the number itself if it's less than 2
    else:
        factors = []
        divisor = 2
        while divisor <= number:
            if number % divisor == 0:
                factors.append(divisor)
                number = number // divisor
            else:
                divisor += 1

        half = len(factors) // 2
        first_half = factors[:half]
        second_half = factors[half:]
        result = [1, 1]  # Initialize result list with two values

        for factor in first_half:
            result[0] *= factor  # Multiply the factors in the first half

        for factor in second_half:
            result[1] *= factor  # Multiply the factors in the second half

    return result


def get_data(filename):
    """ from project1. I tested a Numpy way to read csv file but the output wasn't useful to me."""
    data = []
    with open(filename, 'r') as file:
        for line in file:
            row = line.strip().split(',')
            data.append(row)
        ndata = same_length(data)
        return ndata

def unique_id(data):
    """ return a sorted list of all the unique Arbitration IDs using where"""
    data_array = np.array(data)
    headers = data_array[0]
    index = np.where(headers == "Arbitration_ID")[0][0]
    idlist = data_array[1:][:, index] # [1:] simply means read from index 1 onward
    # [:, index] means we want the element in each row that has the same index as index variable
    return sorted(np.unique(idlist).tolist())

def ecu_data(arbitration_id, data):
    """ from project1, returns the entire row that has the given ID"""
    data_array = np.array(data)
    headers = data_array[0]
    index = np.where(headers == "Arbitration_ID")[0][0]
    rows = data_array[1:]
    results = rows[rows[:, index] == arbitration_id]
    return results.tolist()

def ecu_time_interval(arbitration_id, data):
    """modifying the ecu_time_interval function from project1"""
    data_array = np.array(data)
    time_index = np.where(data_array[0] == "Timestamp")[0][0]
    edata = np.array(ecu_data(arbitration_id, data_array))
    curr_rows = [float(row[time_index]) for row in edata[:-1]] # current starts from index 0 until second to last
    next_rows = [float(row[time_index]) for row in edata[1:]] # next starts from index 1 all the way to the last.
    results = np.subtract(next_rows, curr_rows)
    return results.tolist()

def statistical_functions(data, mean_value=False, median_value=False, std_value=False, variance_value=False):
    """ get mean, median, standard diviation and variance using numpy to get a matrix of each, you need to
    choose it first. by default, they are all False which means the function doesn't return anything.
    whichever you need to see, you fist have to change it to True when you call the function."""
    data_array = np.array(data)
    time_list = []
    ui = unique_id(data_array)
    for num in ui:
        time_list.append(ecu_time_interval(num, data_array))

    mean = [np.mean(arr) for arr in time_list]
    median = [np.median(arr) for arr in time_list]
    std = [np.std(arr) for arr in time_list]
    variance = [np.var(arr) for arr in time_list]

    mean_matrix = np.vstack(mean).T #vstack is used to create 1 matrix out of all the arrays in mean.
    median_matrix = np.vstack(median).T #.T will automatically reconfigure the arrays so they comply with each other for vstack
    std_matrix = np.vstack(std).T
    variance_matrix = np.vstack(variance).T

    if mean_value:
        return mean_matrix
    if median_value:
        return median_matrix
    if std_value:
        return std_matrix
    if variance_value:
        return variance_matrix

def matrix_manipulation(value1, value2):
    """ to demenstrate matrix manipulation"""
    functions = {
        'mean': lambda: statistical_functions(data, mean_value=True),
        'median': lambda: statistical_functions(data, median_value=True),
        'std': lambda: statistical_functions(data, std_value=True),
        'variance': lambda: statistical_functions(data, variance_value=True)
    } # using lambda to write short verson of functions instead of writing them as a full function

    matrix1: float = functions.get(value1, lambda: None)() #retreave the value was given in value1
    matrix2: float = functions.get(value2, lambda: None)() #None means return nothing if the value in not valid
    #float means it will save the outcome as a float in matrix1/2 variables

    matrix_size = break_down_number(len(matrix1[0])) # from earlier
    r = matrix_size[0]
    c = matrix_size[1]

    m1 = np.resize(matrix1, (r, c)) #resizing the 2 matrix so we can get their dot value
    m2 = np.resize(matrix2, (c,r))
    matrix_mult = np.dot(m1, m2)

    m2 = np.resize(matrix2, (r,c))
    elementwise_sum = m1 + m2

    matrix_transpose = np.transpose(m1)

    return matrix_mult, elementwise_sum, matrix_transpose

def bar_chart(data):
    """ Here I'm using a bar chart to illustrate the average time intervals per ID."""
    categories = unique_id(data)
    values = []
    for id in categories:
        if ecu_time_interval(id, data):
            values.append(sum(ecu_time_interval(id, data)) / len(ecu_time_interval(id, data)))
        else:
            values.append(0)

    plt.bar(categories, values, color='blue', alpha=1)
    plt.xlabel('Unique IDs')
    plt.ylabel('Average Time Intervals')
    plt.title('Bar Chart')
    plt.grid(True)
    plt.show()

def histogram_chart(data):
    """ Presenting the same data using histogram chart"""
    ui = unique_id(data)
    value = []
    for id in ui:
        if ecu_time_interval(id, data):
            value.append(sum(ecu_time_interval(id, data)) / len(ecu_time_interval(id, data)))
        else:
            value.append(0)

    plt.hist(value, bins=50, color='purple', alpha=0.7)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram Example')
    plt.grid(True)
    plt.show()

""" Test the Code: """
data = get_data('proj1_data1.csv')
# result = unique_id(data)
# result = ecu_data('340', data)
# result = ecu_time_interval(arbitration_id, data)
# result = statistical_functions(data, mean_value=False, median_value=False, std_value=False, variance_value=False)
# result = matrix_manipulation(value1, value2) # values 1 or 2 Can be mean, median, std or variance
# print(result)
# bar_chart(data)
# histogram_chart(data)


