from flask import Flask, request, jsonify, session, url_for, redirect, render_template
import joblib
import pandas as pd
from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import cross_val_score

from flower_form import FlowerForm 

app = Flask(__name__)
app.config['SECRET_KEY'] = 'mysecretkey'

# Biến toàn cục iris
classifier_loaded = None
iris = datasets.load_iris()

# Hàm huấn luyện và lưu SVM
def train_and_save_model():
    global classifier_loaded

    columns = ["Petal length", "Petal Width", "Sepal Length", "Sepal Width"]
    df = pd.DataFrame(iris.data, columns=columns)
    y = iris.target

    print(df.describe())
    print("\nKiem tra xem du lieu co bi thieu (NULL) khong?")
    print(df.isnull().sum())

    model = svm.SVC(kernel='rbf')

    scores = cross_val_score(model, df, y, cv=5)
    print("\nĐố chính xác của mô hình với 5-fold cross-validation cho từng fold: ", scores)
    print("Độ chính xác trung bình của mô hình: %.3f" % scores.mean())

    model.fit(df, y)

    joblib.dump(model, 'svm_iris_model.pkl')
    print("Mô hình đã được lưu dưới tên 'svm_iris_model.pkl'.")

    classifier_loaded = model

# Prediction function (adapted for SVM - no encoder needed)
def make_prediction(model, sample_json):
    SepalLengthCm = sample_json['SepalLengthCm']
    SepalWidthCm = sample_json['SepalWidthCm']
    PetalLengthCm = sample_json['PetalLengthCm']
    PetalWidthCm = sample_json['PetalWidthCm']

    flower = [[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]]
    prediction = model.predict(flower)

    return prediction[0]  

# Read the SVM model
classifier_loaded = joblib.load("svm_iris_model.pkl") 

@app.route("/", methods=['GET', 'POST'])
def index():
    form = FlowerForm()

    if form.validate_on_submit():
        session['SepalLengthCm'] = form.SepalLengthCm.data
        session['SepalWidthCm'] = form.SepalWidthCm.data
        session['PetalLengthCm'] = form.PetalLengthCm.data
        session['PetalWidthCm'] = form.PetalWidthCm.data

        return redirect(url_for("prediction"))

    return render_template("home.html", form=form)

@app.route('/prediction')
def prediction():
    content = {'SepalLengthCm': float(session['SepalLengthCm']),
               'SepalWidthCm': float(session['SepalWidthCm']),
               'PetalLengthCm': float(session['PetalLengthCm']),
               'PetalWidthCm': float(session['PetalWidthCm'])}

    results = make_prediction(classifier_loaded, content)

    return render_template('prediction.html', results=iris.target_names[results])

if __name__ == '__main__':
    train_and_save_model()
    app.run(host='0.0.0.0', port=8080)