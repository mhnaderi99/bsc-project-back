import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np


def fault_rate(row):
    if row['time'] != 0:
        return row['num'] / row['time']
    else:
        return 0


def func0(t, lambda0, v0):
    return lambda0 * np.exp((-lambda0/v0) * t)


def func1(t, lambda0, theta):
    return lambda0 / (1 + lambda0 * theta * t)


def func2(t, a, b):
    return a * b * np.exp(-b*t)


funcs = {0: func0, 1: func1, 2: func2}


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


mius = {0: miu0, 1: miu1, 2: miu2}


def faults_in_time_range(t1, t2, params, model):
    return mius[model](t2, params) - mius[model](t1, params)


def plot(xdata, ydata, popt):
    plt.plot(xdata, ydata, 'b-', label='data')
    plt.plot(xdata, func0(xdata, *popt), 'g--')
    plt.show()


def read_file(filename):
    df = pd.read_excel('./downloads/atnt_data-2.xlsx')
    df['fault_rate'] = df.apply(lambda row: fault_rate(row), axis=1)
    x = df['time'].to_numpy()
    y = df['fault_rate'].to_numpy()
    return x, y


class Model:
    def __init__(self, filename):
        self.filename = filename
        self.model = 0
        self.target = 0
        self.now = 0
        self.params = {}

    def handle(self):
        x, y = read_file(self.filename)
        self.now = x.max() + 1
        popt, pcov = curve_fit(funcs[self.model], x, y)
        if self.model == 0:
            self.params = {'v0': popt[1], 'lambda0': popt[0]}
        if self.model == 1:
            self.params = {'theta': popt[1], 'lambda0': popt[0]}
        if self.model == 2:
            self.params = {'b': popt[1], 'a': popt[1]}
        # plot(x, y, popt)
        return popt, x, y, funcs[self.model](x, *popt)


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