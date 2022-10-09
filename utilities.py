import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np


def fault_rate(row):
    if row['normal_time'] != 0:
        return row['num'] / row['normal_time']
    else:
        return 0


def func0(t, lambda0, v0):
    return lambda0 * np.exp((-lambda0/v0) * t)


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
    return func2(t, a, b)


funcs = {0: func0, 1: func1, 2: func2, 3: func3, 4: func4}


def intensity_rate_at_time0(t, params):
    lambda0 = params['lambda0']
    v0 = params['v0']
    # print(lambda0 * np.exp((-lambda0/v0) * t))
    return lambda0 * np.exp((-lambda0/v0) * t)


def intensity_rate_at_time1(t, params):
    lambda0 = params['lambda0']
    theta = params['theta']
    # print(lambda0 / (1 + lambda0 * theta * t))
    return lambda0 / (1 + lambda0 * theta * t)


def intensity_rate_at_time2(t, params):
    a = params['a']
    b = params['b']
    return b*(a - miu2(t, params))


intensity_rate_at_times = {0: intensity_rate_at_time0,
                           1: intensity_rate_at_time1,
                           2: intensity_rate_at_time2}


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
    return (v0/lambda0)*np.log(lambda_p/lambda_f)


def remaining_time_until_target1(t, params, lambda_f):
    lambda0 = params['lambda0']
    theta = params['theta']
    lambda_p = intensity_rate_at_time1(t, params)
    return (lambda0 - lambda_f)/(lambda_f*lambda0*theta) - (lambda0 - lambda_p)/(lambda_p*lambda0*theta)


def remaining_time_until_target2(t, params, lambda_f):
    b = params['b']
    lambda_p = intensity_rate_at_time2(t, params)
    return (np.log(lambda_p) - np.log(lambda_f))/b


remaining_time_until_targets = {0: remaining_time_until_target0,
                                1: remaining_time_until_target1,
                                2: remaining_time_until_target2}


def intensity_rate_decrement_per_fault0(lambda0, v0):
    return -lambda0/v0


def miu0(t, params):
    lambda0 = params['lambda0']
    v0 = params['v0']
    return v0*(1 - np.exp(-(lambda0/v0)*t))


def miu1(t, params):
    lambda0 = params['lambda0']
    theta = params['theta']
    return np.log(1 + lambda0*theta*t)/theta


def miu2(t, params):
    a = params['a']
    b = params['b']
    return a*(1 - np.exp(-b*t))


def miu3(t, params):
    a = params['a']
    b = params['b']
    bt = (t*b**2)/(1 + b*t)
    return a*(1 - (1 + bt*t)*np.exp(-bt*t))


# inflection s-shaped Model
def miu4(t, params):
    a = params['a']
    b = params['b']
    beta = params['beta']
    bt = b / (1 + beta * np.exp(-b * t))
    return a * (1 - np.exp(-bt * t)) / (1 + beta * np.exp(-bt * t))


mius = {0: miu0, 1: miu1, 2: miu2, 3: miu3, 4: miu4}


def m0(t, lambda0, v0):
    return v0 * (1 - np.exp(-(lambda0 / v0) * t))


def m1(t, lambda0, theta):
    return np.log(1 + lambda0 * theta * t) / theta


def m2(t, a, b):
    return a * (1 - np.exp(-b * t))


def m3(t, a, b):
    bt = (t * b ** 2) / (1 + b * t)
    return a * (1 - (1 + bt * t) * np.exp(-bt * t))


# inflection s-shaped Model
def m4(t, a, b, beta):
    bt = b/(1 + beta*np.exp(-b*t))
    return a*(1 - np.exp(-bt*t))/(1 + beta*np.exp(-bt*t))


ms = {0: m0, 1: m1, 2: m2, 3: m3, 4: m4}


def faults_in_time_range(t1, t2, params, model):
    return mius[model](t2, params) - mius[model](t1, params)


def plot(xdata, ydata, popt):
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, func0(xdata, *popt), 'g--')
    plt.show()


def read_file(filename):
    df = pd.read_excel('./downloads/musa_dataset.xlsx')
    df['fault_rate'] = df.apply(lambda row: fault_rate(row), axis=1)
    training_size = max(int(df['num'].size / 5), 15)
    x = df['normal_time'].to_numpy()[1:training_size]
    y = df['fault_rate'].to_numpy()[1:training_size]
    nums = df['num'].to_numpy()[1:training_size]
    return x, y, nums


class Model:
    def __init__(self, filename):
        self.filename = filename
        self.model = 0
        self.target = 0
        self.now = 0
        self.params = {}

    def handle(self):
        x, y, nums = read_file(self.filename)
        self.now = x.max() + 1
        popt, pcov = curve_fit(funcs[self.model], x, y, method='dogbox')
        popt2, pcov2 = curve_fit(ms[self.model], x, nums)
        print(popt, popt2)
        x2 = np.linspace(np.min(x), np.max(x), num=100)
        fitted = funcs[self.model](x, *popt2)
        fitted2 = funcs[self.model](x2, *popt2)
        error = np.sum(np.abs(y - fitted) ** 2) / len(y)
        if self.model == 0:
            self.params = {'v0': popt2[1], 'lambda0': popt2[0]}
        if self.model == 1:
            self.params = {'theta': popt2[1], 'lambda0': popt2[0]}
        if self.model == 2:
            self.params = {'b': popt2[1], 'a': popt2[0]}
        if self.model == 3:
            self.params = {'b': popt2[1], 'a': popt2[0]}
        if self.model == 4:
            self.params = {'beta': popt2[2], 'b': popt2[1], 'a': popt2[0]}
        # plot(x, y, popt)

        # miiu = mius[self.model](x, self.params)
        miiu2 = mius[self.model](x2, self.params)

        return popt2, x, y, fitted, error, x2, fitted2, nums, miiu2


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