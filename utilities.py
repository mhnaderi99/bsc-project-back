import matplotlib.pyplot as plt
import numpy
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import warnings

warnings.filterwarnings('ignore')


models = 6
file_number = 2


def fault_rate(row):
    if row['normal_time'] != 0:
        return row['num'] / row['normal_time']
    else:
        return 0


# basic execution time model
def func0(t, lambda0, v0):
    return lambda0 * np.exp((-lambda0/v0) * t)


# logarithmic poisson model
def func1(t, lambda0, theta):
    return lambda0 / (1 + lambda0 * theta * t)


# G-O model
def func2(t, a, b):
    return a * b * np.exp(-b*t)


# # G-O optimized model
# def func2(t, a, b, c):
#     return a * b * c * np.exp(-b*(np.power(t, c)))*np.power(t, c-1)


# delayed s-shaped model
def func3(t, a, b):
    return a*b**2*t*np.exp(-b*t)


# inflection s-shaped Model
def func4(t, a, b, beta):
    return (a*b*np.exp(-b*t)*(1+beta*np.exp(-b*t)) + a*b*beta*np.exp(-b*t)*(1-np.exp(-b*t)))/(1+beta*np.exp(-b*t))**2


# Yamada Exponential Model
def func5(t, a, r, alpha, beta):
    return a*r*alpha*beta*np.exp(-r*alpha*(1 - np.exp(-beta*t)) - beta*t)


# Yamada Imperfect Debugging Model
def func6(t, a, b, alpha):
    return a*b*(alpha*np.exp(alpha*t) + b*np.exp(-b*t))/(alpha + b)


funcs = {0: func0, 1: func1, 2: func2, 3: func3, 4: func4, 5: func5}


def intensity_rate_at_time0(t, params):
    lambda0 = params['lambda0']
    v0 = params['v0']
    return func0(t, lambda0, v0)


def intensity_rate_at_time1(t, params):
    lambda0 = params['lambda0']
    theta = params['theta']
    return func1(t, lambda0, theta)


def intensity_rate_at_time2(t, params):
    a = params['a']
    b = params['b']
    return func2(t, a, b)


def intensity_rate_at_time3(t, params):
    a = params['a']
    b = params['b']
    return func3(t, a, b)


def intensity_rate_at_time4(t, params):
    a = params['a']
    b = params['b']
    beta = params['beta']
    return func4(t, a, b, beta)


def intensity_rate_at_time5(t, params):
    a = params['a']
    r = params['r']
    alpha = params['alpha']
    beta = params['beta']
    return func5(t, a, r, alpha, beta)


intensity_rate_at_times = {0: intensity_rate_at_time0,
                           1: intensity_rate_at_time1,
                           2: intensity_rate_at_time2,
                           3: intensity_rate_at_time3,
                           4: intensity_rate_at_time4,
                           5: intensity_rate_at_time5}


def remaining_faults_until_target0(t, params, lambda_f):
    lambda0 = params['lambda0']
    v0 = params['v0']
    lambda_p = intensity_rate_at_time0(t, params)
    print(lambda_p - lambda_f)
    return (v0/lambda0)*(lambda_p - lambda_f)


def remaining_faults_until_target1(t, params, lambda_f):
    theta = params['theta']
    lambda_p = intensity_rate_at_time1(t, params)
    return -np.log(lambda_f/lambda_p)/theta


def remaining_faults_until_target2(t, params, lambda_f):
    b = params['b']
    lambda_p = intensity_rate_at_time2(t, params)
    return (lambda_p - lambda_f)/b


remaining_faults_until_targets = {0: remaining_faults_until_target0,
                                  1: remaining_faults_until_target1,
                                  2: remaining_faults_until_target2}


def remaining_time_until_target0(t, params, lambda_f):
    lambda0 = params['lambda0']
    v0 = params['v0']
    lambda_p = intensity_rate_at_time0(t, params)
    if lambda_p <= lambda_f:
        return 0
    return (v0/lambda0)*np.log(lambda_p/lambda_f)


def remaining_time_until_target1(t, params, lambda_f):
    lambda0 = params['lambda0']
    theta = params['theta']
    lambda_p = intensity_rate_at_time1(t, params)
    if lambda_p <= lambda_f:
        return 0
    return (lambda0 - lambda_f)/(lambda_f*lambda0*theta) - (lambda0 - lambda_p)/(lambda_p*lambda0*theta)


def remaining_time_until_target2(t, params, lambda_f):
    b = params['b']
    lambda_p = intensity_rate_at_time2(t, params)
    if lambda_p <= lambda_f:
        return 0
    return (np.log(lambda_p) - np.log(lambda_f))/b


def remaining_time_until_target3(t, params, lambda_f):
    lambda_p = intensity_rate_at_time3(t, params)
    if lambda_p <= lambda_f:
        return 0

    delta_t = 0
    while lambda_p > lambda_f:
        delta_t += 1
        lambda_p = intensity_rate_at_time3(t + delta_t, params)
    return delta_t


def remaining_time_until_target4(t, params, lambda_f):
    lambda_p = intensity_rate_at_time4(t, params)
    if lambda_p <= lambda_f:
        return 0

    delta_t = 0
    while lambda_p > lambda_f:
        delta_t += 1
        lambda_p = intensity_rate_at_time4(t + delta_t, params)
    return delta_t


def remaining_time_until_target5(t, params, lambda_f):
    lambda_p = intensity_rate_at_time5(t, params)
    if lambda_p <= lambda_f:
        return 0

    delta_t = 0
    while lambda_p > lambda_f:
        delta_t += 1
        lambda_p = intensity_rate_at_time5(t + delta_t, params)
    return delta_t


remaining_time_until_targets = {0: remaining_time_until_target0,
                                1: remaining_time_until_target1,
                                2: remaining_time_until_target2,
                                3: remaining_time_until_target3,
                                4: remaining_time_until_target4,
                                5: remaining_time_until_target5}


def intensity_rate_decrement_per_fault0(lambda0, v0):
    return -lambda0/v0



def m0(t, lambda0, v0):
    return v0 * (1 - np.exp(-(lambda0 / v0) * t))


def m1(t, lambda0, theta):
    return np.log(1 + lambda0 * theta * t) / theta


def m2(t, a, b):
    return a * (1 - np.exp(-b * t))


def m3(t, a, b):
    # bt = (t * b ** 2) / (1 + b * t)
    bt = b
    return a * (1 - (1 + bt * t) * np.exp(-bt * t))


# inflection s-shaped Model
def m4(t, a, b, beta):
    # bt = b/(1 + beta*np.exp(-b*t))
    bt = b
    return a*(1 - np.exp(-bt*t))/(1 + beta*np.exp(-bt*t))


# Yamada Exponential Model
def m5(t, a, r, alpha, beta):
    # at = a*np.exp(alpha*t)
    return a*(1 - np.exp(-r*alpha*(1 - np.exp(-beta*t))))


# Yamada Imperfect Debugging
def m6(t, a, b, alpha):
    return a*b*(np.exp(alpha*t) - np.exp(-b*t))/(alpha + b)


ms = {0: m0, 1: m1, 2: m2, 3: m3, 4: m4, 5: m5}


def miu0(t, params):
    lambda0 = params['lambda0']
    v0 = params['v0']
    return m0(t, lambda0, v0)


def miu1(t, params):
    lambda0 = params['lambda0']
    theta = params['theta']
    return m1(t, lambda0, theta)


def miu2(t, params):
    a = params['a']
    b = params['b']
    return m2(t, a, b)


def miu3(t, params):
    a = params['a']
    b = params['b']
    # bt = (t*b**2)/(1 + b*t)
    bt = b
    return m3(t, a, b)


# inflection s-shaped Model
def miu4(t, params):
    a = params['a']
    b = params['b']
    beta = params['beta']
    bt = b / (1 + beta * np.exp(-b * t))
    return m4(t, a, b, beta)


# Yamada Exponential Model
def miu5(t, params):
    a = params['a']
    r = params['r']
    alpha = params['alpha']
    beta = params['beta']
    return m5(t, a, r, alpha, beta)


# Yamada Imperfect Debugging Model
def miu6(t, params):
    a = params['a']
    b = params['b']
    alpha = params['alpha']
    return m6(t, a, b, alpha)


mius = {0: miu0, 1: miu1, 2: miu2, 3: miu3, 4: miu4, 5: miu5}


def faults_in_time_range(t1, t2, params, model):
    return mius[model](t2, params) - mius[model](t1, params)


def reliability(model_number, t, params, x):
    return numpy.exp(mius[model_number](t, params) - mius[model_number](t+x, params))


def safe_time_reliability(model_number, t, params, target):
    i = 0
    r = reliability(model_number, t, params, i)
    if r < target:
        return 0

    while r >= target:
        i += 1
        r = reliability(model_number, t, params, i)

    print(i)
    return i


def plot(xdata, ydata, popt):
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, func0(xdata, *popt), 'g--')
    plt.show()


def read_file(data_index):
    datasets = ['./downloads/atnt_data.xlsx', './downloads/musa_dataset.xlsx', './downloads/ntds_data.xlsx']
    df = pd.read_excel(datasets[data_index])
    df['fault_rate'] = df.apply(lambda row: fault_rate(row), axis=1)
    training_size = max(4*int(df['num'].size / 5), 20)
    start_index = 1
    x = df['normal_time'].to_numpy()[start_index:training_size]
    y = df['fault_rate'].to_numpy()[start_index:training_size]
    nums = df['num'].to_numpy()[start_index:training_size]
    eval_x = df['normal_time'].to_numpy()[training_size:]
    eval_nums = df['num'].to_numpy()[training_size:]
    return x, y, nums, eval_x, eval_nums


def handle_params(model_number, popt2):
    if model_number == 0:
        return {'v0': popt2[1], 'lambda0': popt2[0]}
    if model_number == 1:
        return {'theta': popt2[1], 'lambda0': popt2[0]}
    if model_number == 2:
        return {'b': popt2[1], 'a': popt2[0]}
    if model_number == 3:
        return {'b': popt2[1], 'a': popt2[0]}
    if model_number == 4:
        return {'beta': popt2[2], 'b': popt2[1], 'a': popt2[0]}
    if model_number == 5:
        return {'beta': popt2[3], 'alpha': popt2[2], 'r': popt2[1], 'a': popt2[0]}


class Model:
    def __init__(self, filename):
        self.filename = filename
        self.model = 0
        self.target = 0
        self.now = 0
        self.params = {}

    def calculate_error(self, model_number):
        mod = self.model
        para = self.params
        self.model = model_number
        x, y, nums, eval_x, eval_nums = read_file(file_number)
        popt2, pcov2 = curve_fit(ms[self.model], x, nums, maxfev=100000)

        self.params = handle_params(self.model, popt2)

        miiu_eval = mius[self.model](eval_x, self.params)
        x2 = np.linspace(np.min(x), np.max(x), num=1000)
        miiu = mius[self.model](x, self.params)
        error_fit = np.sqrt(np.sum(np.power(miiu - nums, 2)) / len(eval_nums))
        error_eval = np.sum(np.power(miiu_eval - eval_nums, 2)) / len(eval_nums)

        self.params = para
        self.model = mod
        return error_fit

    def calculate_errors(self):
        errors = []
        for i in range(models):
            error_eval = self.calculate_error(i)
            errors.append(error_eval)
        return errors

    def handle(self):
        x, y, nums, eval_x, eval_nums = read_file(file_number)
        self.now = x.max() + 1
        popt2, pcov2 = curve_fit(ms[self.model], x, nums, maxfev=100000, method='trf')

        self.params = handle_params(self.model, popt2)

        # plot(x, y, popt)

        x2 = np.linspace(np.min(x), np.max(x), num=1000)
        fitted = funcs[self.model](x, *popt2)
        fitted2 = funcs[self.model](x2, *popt2)

        miiu = mius[self.model](x, self.params)
        miiu2 = mius[self.model](x2, self.params)

        r = reliability(self.model, self.now, self.params, 100)
        print('Reliablity: ', r)
        # d = np.diff(miiu2, x2)

        # errors = self.calculate_errors()
        errors = []
        return popt2, x, y, fitted, errors, x2, fitted2, nums, miiu2


# xdata, ydata = read_file('atnt_data.xlsx')
#
# popt, pcov = curve_fit(func, xdata, ydata)
# popt2, pcov2 = curve_fit(func2, xdata, ydata)
# popt3, pcov3 = curve_fit(func3, xdata, ydata)

# lambda0 = popt[0]
# v0 = popt[1]
# plt.plot(xdata, ydata, 'b-', label='data')
# plt.plot(xdata, func(xdata, *popt), 'g--')
# plt.plot(xdata, func2(xdata, *popt2), 'r--')
# plt.plot(xdata, func3(xdata, *popt3), 'y--')
# plt.show()

# print(intensity_rate_at_time(100, lambda0, v0))
# print(remaining_faults_until_target(100, lambda0, v0, 0.05))
# print(remaining_time_until_target(100, lambda0, v0, 0.05))
# print(intensity_rate_decrement_per_fault(lambda0, v0))
# print(faults_in_time_range(100,200, lambda0, v0))