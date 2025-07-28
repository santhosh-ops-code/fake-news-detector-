# app/app.py

from flask import Flask, render_template, request
import joblib
import os

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load(os.path.join("..", "model", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join("..", "model", "tfidf_vectorizer.pkl"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        news_text = request.form["news"]
        transformed_text = vectorizer.transform([news_text])
        prediction = model.predict(transformed_text)[0]
        result = "Real News ✅" if prediction == 1 else "Fake News ❌"
        return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
