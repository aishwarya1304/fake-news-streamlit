📰 Fake News Detection System
📌 Overview
The Fake News Detection System is a Machine Learning-based project that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques.
The goal of this project is to combat misinformation by automatically analyzing textual content and predicting its authenticity.
🎯 Problem Statement
With the rapid growth of social media and online news platforms, fake news spreads quickly and influences public opinion.
This project builds a supervised ML model that can:
Analyze news article text
Extract meaningful linguistic patterns
Predict whether the news is real or fake
🛠️ Tech Stack
Python
Pandas – Data preprocessing
NumPy – Numerical computations
Scikit-learn – ML modeling
NLTK / spaCy – Text preprocessing
TF-IDF Vectorizer – Feature extraction
Logistic Regression / Naive Bayes / PassiveAggressiveClassifier – Classification models
Matplotlib / Seaborn – Data visualization
(Optional) Flask / Streamlit – Web deployment
📂 Project Structure
Fake-News-Detection/
│
├── data/
│   ├── Fake.csv
│   ├── True.csv
│
├── notebooks/
│   └── EDA_and_Model.ipynb
│
├── models/
│   └── final_model.pkl
│
├── app.py (if deployed)
├── requirements.txt
└── README.md
⚙️ How It Works
1️⃣ Data Collection
Dataset contains labeled news articles:
Fake news articles
True news articles
2️⃣ Data Preprocessing
Lowercasing
Removing punctuation
Removing stopwords
Tokenization
Lemmatization
Vectorization using TF-IDF
3️⃣ Model Training
Train-test split
Model training using classification algorithms
Accuracy, Precision, Recall, F1-score evaluation
4️⃣ Prediction
The trained model takes new news text as input and predicts:
Real
Fake
📊 Model Performance
Example metrics (replace with your actual results):
Accuracy: 94%
Precision: 93%
Recall: 95%
F1 Score: 94%
🚀 Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/Fake-News-Detection.git
cd Fake-News-Detection
2️⃣ Install Dependencies
pip install -r requirements.txt
3️⃣ Run the Model (Notebook)
Open:
notebooks/EDA_and_Model.ipynb
4️⃣ Run Web App (If Applicable)
python app.py
or
streamlit run app.py
📌 Features
✔ Cleaned and preprocessed dataset
✔ TF-IDF based feature extraction
✔ Multiple ML models comparison
✔ Model evaluation metrics
✔ Optional web interface for user input
🔍 Future Improvements
Use Deep Learning (LSTM, BERT)
Use transformer models for better accuracy
Add fact-checking API integration
Deploy on cloud (AWS / GCP / Heroku)
Real-time browser extension
📚 Dataset
Publicly available Fake and Real News dataset (e.g., Kaggle Fake News Dataset).
👩‍💻 Authors
Aishwarya Reddy Nagam
