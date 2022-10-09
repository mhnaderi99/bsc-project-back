import json
from flask import Flask, request
from utilities import *

PORT = 8000
filename = ''
p = Model('')
app = Flask(__name__)


@app.route('/getHello')
def hello():
    return '<h1>Hello, World!</h1>'


@app.route('/about/')
def about():
    return '<h3>This is a Flask web application.</h3>'


@app.route('/uploadFile', methods=['POST'])
def upload():
    data = request.files['file']
    save_dst = 'downloads/' + data.filename
    data.save(dst=save_dst)
    p.filename = save_dst
    parameters, x, y, fit = p.handle()
    return {'status': 'OK', 'params': parameters.tolist(), 'x': x.tolist(), 'y': y.tolist(), 'fit': fit}


@app.route('/calculate', methods=['POST'])
def calculate():
    data = json.loads(request.form['data'])

    p.model = data['model']
    parameters, x, y, fit, error, x2, fit2, miiu, miiu2 = p.handle()
    # print(data)
    return {'status': 'OK', 'params': parameters.tolist(), 'model': p.model, 'xdata': x.tolist(), 'ydata': y.tolist(),
            'fit': fit.tolist(), 'error': error, 'x2': x2.tolist(), 'fit2': fit2.tolist(), 'miu': miiu.tolist(),
            'miu2': miiu2.tolist()}


@app.route('/intensityRateAtTime', methods=['GET'])
def get_intensity_rate_at_time():
    params = request.args
    # print(params['time'])
    return {'status': 'OK', 'rate': intensity_rate_at_times[p.model](float(params['time']), p.params)}


@app.route('/remainingFaultsUntilTarget', methods=['GET'])
def get_remaining_faults_until_target():
    params = request.args
    target = float(params['target'])
    # print(target)
    return {'status': 'OK', 'faults': remaining_faults_until_targets[p.model](p.now, p.params, target)}


@app.route('/remainingTimeUntilTarget', methods=['GET'])
def get_remaining_time_until_target():
    params = request.args
    target = float(params['target'])
    # print(target)
    return {'status': 'OK', 'time': remaining_time_until_targets[p.model](p.now, p.params, target)}


@app.route('/faultsInTimeRange', methods=['GET'])
def get_faults_in_time_range():
    params = request.args
    from_time = int(params['from'])
    to_time = int(params['to'])
    return {'status': 'OK', 'faults': faults_in_time_range(from_time, to_time, p.params, p.model)}
