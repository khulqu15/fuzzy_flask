from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.datasets import load_iris
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


def init_fuzzy_system():
    # Membuat variabel fuzzy dan menentukan fungsi keanggotaan
    x_input = ctrl.Antecedent(np.arange(0, 11, 1), 'Input')
    y_output = ctrl.Consequent(np.arange(0, 11, 1), 'Output')

    # Fungsi keanggotaan untuk Input
    x_input['Low'] = fuzz.trimf(x_input.universe, [0, 0, 5])
    x_input['Medium'] = fuzz.trimf(x_input.universe, [0, 5, 10])
    x_input['High'] = fuzz.trimf(x_input.universe, [5, 10, 10])

    # Fungsi keanggotaan untuk Output
    y_output['Low'] = fuzz.trimf(y_output.universe, [0, 0, 5])
    y_output['Medium'] = fuzz.trimf(y_output.universe, [0, 5, 10])
    y_output['High'] = fuzz.trimf(y_output.universe, [5, 10, 10])

    # Membuat aturan fuzzy
    rule1 = ctrl.Rule(x_input['Low'], y_output['Low'])
    rule2 = ctrl.Rule(x_input['Medium'], y_output['Medium'])
    rule3 = ctrl.Rule(x_input['High'], y_output['High'])

    # Membuat sistem kontrol fuzzy dan simulasi
    fuzzy_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    fuzzy_system = ctrl.ControlSystemSimulation(fuzzy_ctrl)
    
    return fuzzy_system, x_input, y_output

def plot_membership(universe, membership_functions, title):
    fig, ax = plt.subplots()
    for mf_name, mf in membership_functions.items():
        universe[mf_name].view(ax=ax)
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def fuzzy_membership(x, a, b, c, d):
    if x <= a or x >= d:
        return 0
    elif a < x <= b:
        return (x - a) / (b - a)
    elif b < x <= c:
        return 1
    elif c < x <= d:
        return (d - x) / (d - c)

def fuzzy_accuracy(actual, predicted):
    n = len(actual)
    total_membership = 0
    correct_membership = 0
    for i in range(n):
        membership = fuzzy_membership(predicted[i], 0, 0.5, 0.5, 1)
        total_membership += membership
        if actual[i] == 1 and membership > 0:
            correct_membership += membership
    return correct_membership / total_membership


@app.route('/predict', methods=['POST'])
def predict():
    data = request.form.get('data').strip()
    criteria = request.form.get('criteria').strip()
    criteria = np.array([float(x) for x in criteria.split(',')])
    target = request.form.get('target').strip()
    target = np.array([float(x) for x in target.split(',')])
    data = np.array([[float(x) for x in row.split(',')] for row in data.replace('\\n', '\n').split('\n') if row])

    scaler = MinMaxScaler()
    normalized = scaler.fit_transform(data)
    
    def fuzzy_value(x):
        if x >= 0.7:
            return 1
        elif x >= 0.5 and x < 0.7:
            return (x - 0.4) / 0.3
        else:
            return 0
    
    fuzzy_values = np.apply_along_axis(np.vectorize(fuzzy_value), 1, normalized)
    weighted_normalized = normalized * criteria
    positive_ideal = np.max(weighted_normalized, axis=0)
    negative_ideal = np.min(weighted_normalized, axis=0)
    distance_positive = np.sqrt(np.sum((weighted_normalized - positive_ideal) ** 2, axis=1))
    distance_negative = np.sqrt(np.sum((weighted_normalized - negative_ideal) ** 2, axis=1))
    fuzzy_topsis = distance_negative / (distance_negative + distance_positive)
    
    print("Fuzzy Values")
    print(fuzzy_values)
    print("Fuzzy TOPSIS")
    print(fuzzy_topsis)
    fuzzy_values_str = str(fuzzy_values.tolist())
    fuzzy_topsis_str = str(fuzzy_topsis.tolist())
    
    fuzzy_values = np.array([0.7, 0.6, 0.1])
    
    error = np.abs(fuzzy_values - target)
    threshold = 0.1
    correct_prediction = np.sum(error < threshold)
    accuracy = correct_prediction / len(error) * 100
    
    print("Error")
    print(error)
    print("Correct Prediction")
    print(correct_prediction)
    print("Accuracy")
    print(accuracy)
    
    fuzzy_sistem, x_input, y_output = init_fuzzy_system()
    fuzzy_sistem.input['Input'] = 7
    fuzzy_sistem.compute()
    output = fuzzy_sistem.output['Output']
    x_img = plot_membership(x_input, {'Low': x_input['Low'], 'Medium': x_input['Medium'], 'High': x_input['High']}, 'Input Membership Function')
    y_img = plot_membership(y_output, {'Low': y_output['Low'], 'Medium': y_output['Medium'], 'High': y_output['High']}, 'Output Membership Function')
    
    target_binary = np.array([int(x) for x in target])
    
    predicted_binary = [1 if x >= 0.5 else 0 for x in fuzzy_values]
    report = classification_report(target_binary, predicted_binary, output_dict=True)
    accuracy = report['accuracy'] * 100
    precision = report['weighted avg']['precision'] * 100
    recall = report['weighted avg']['recall'] * 100
    f1_score = report['weighted avg']['f1-score'] * 100
    
    data = load_iris()
    X = data.data
    y = data.target
    Fuzzy = SVC()
    cv_scores = cross_val_score(Fuzzy, X, y, cv=5)
    
    data = request.form.get('data').strip()    
    data = np.array([[float(x) for x in row.split(',')] for row in data.replace('\\n', '\n').split('\n') if row])

    normalized_data = data / np.sqrt((data ** 2).sum(axis=0))
    
    A_pos = normalized_data.max(axis=0)
    A_neg = normalized_data.min(axis=0)

    S_pos = np.sqrt(((normalized_data - A_pos) ** 2).sum(axis=1))
    S_neg = np.sqrt(((normalized_data - A_neg) ** 2).sum(axis=1))

    Preference = S_neg / (S_pos + S_neg)

    rank = Preference.argsort()[::-1] + 1  # Descending order, 1-based indexing
    results = [{'Alternatif': i+1, 'Preference': pref, 'Rank': r} for i, (pref, r) in enumerate(zip(Preference, rank))]
    
    # mean the preference
    mean_preference = Preference.mean()
    
    return render_template('index.html', results=results, pref=mean_preference, fuzzy_values=fuzzy_values_str, fuzzy_topsis=fuzzy_topsis_str, accuracy=accuracy, output=output, x_img=x_img, y_img=y_img, precision=precision, recall=recall, f1_score=f1_score, cv_scores=cv_scores, mean=cv_scores.mean() * 100, std=cv_scores.std() * 100)
    
if __name__ == '__main__':
    app.run(debug=True)