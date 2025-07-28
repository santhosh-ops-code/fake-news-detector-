# 📰 Fake News Detector

A machine learning web application that classifies news content as Fake or Real using NLP techniques.

---

## 🚀 Features

- Predicts whether a news article is real or fake  
- Simple web interface built with Flask  
- Trained on real-world Fake and True news datasets  
- Uses TF-IDF vectorization and Passive Aggressive Classifier  
- Clean, responsive UI with prediction results

---

## 🧠 Technologies Used

- Python  
- Flask (Web Framework)  
- Scikit-learn (ML model)  
- NLTK (Natural Language Processing)  
- Pandas (Data processing)  
- HTML/CSS (Frontend)

---

## 📁 Project Structure

```
fake-news-detector/
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/
│       └── style.css
├── data/
│   ├── Fake.csv
│   └── True.csv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── train_model.py
├── requirements.txt
└── README.md
```

---

## 📦 Installation & Running Locally

Follow the steps below to run the project locally on your machine:

### 1. Clone the Repository

```bash
git clone https://github.com/santhosh-ops-code/fake-news-detector.git
cd fake-news-detector
```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the ML Model

```bash
python train_model.py
```

This will generate:

- model/model.pkl  
- model/vectorizer.pkl

### 5. Run the Flask App

```bash
cd app
python app.py
```

Then open your browser and go to:  
http://127.0.0.1:5000

---

## 📊 Dataset

We used the Fake and Real News Dataset from Kaggle:  
🔗 https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

- Fake.csv — Contains fake news articles  
- True.csv — Contains real news articles

---

## 🧪 Machine Learning Model

- Model: PassiveAggressiveClassifier  
- Preprocessing: Stopword removal, lowercasing, punctuation cleaning  
- Feature Extraction: TF-IDF Vectorizer  
- Evaluation Metric: Accuracy

---

## 🚀 Future Improvements

- Deploy on Render or Railway  
- Add file upload or article URL support  
- Enhance UI with Bootstrap or Tailwind  
- Real-time news scraping & prediction  
- Advanced models like BERT or RoBERTa

---

## 👥 Team Members

This project was developed as part of our major academic project:

| Name      | Role                          |
|-----------|-------------------------------|
| Santhosh  | Team Lead, Backend & ML       |
| Sushma    | Frontend Developer            |
| Harshitha | Data Preprocessing & Testing  |

---

## 📜 License

This project is licensed under the MIT License.  
Feel free to use and modify for educational purposes.

---

## 🙌 Acknowledgments

- Kaggle — for providing the dataset  
- Scikit-learn & Flask documentation  
- OpenAI ChatGPT — for assistance in structuring, debugging, and documenting the project