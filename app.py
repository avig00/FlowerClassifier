import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
app = Flask(__name__)

# Load model
model = pickle.load(open("model.pkl", "rb"))

# Home page
@app.route("/")
def Home():
    return render_template("index.html")

# Prediction page
@app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)



    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))




if __name__ == "__main__":
    app.run(debug=True)