from flask import Flask, render_template, request
import pickle

filename = 'my_randomforest_model.pkl'
clf = pickle.load(open(filename, 'rb'))

cv = pickle.load(open('transform.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])


def predict():
    if request.method == 'POST':
        message = request.form['sentence']
        data = [message]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html', prediction = my_prediction)


