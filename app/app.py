from flask import Flask, render_template, request
import joblib
import os
from pathlib import Path

app = Flask(__name__, template_folder="templates")  # ✅ Corrected

# Load model and vectorizer
BASE_DIR = Path(__file__).resolve().parent.parent
model = joblib.load(BASE_DIR / "model" / "fake_news_model.pkl")
vectorizer = joblib.load(BASE_DIR / "model" / "tfidf_vectorizer.pkl")

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
