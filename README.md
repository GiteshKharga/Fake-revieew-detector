# 🕵️‍♂️ Fake Review Detector

A **smart and lightweight web application** that detects fake reviews using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.  
Built with **Python** and **Streamlit**, this project analyzes text, predicts authenticity, and visualizes model performance — all in real-time.

---

## 🚀 Features

- **Text Cleaning & Feature Extraction**  
  Lowercasing, removing links, stopwords, special characters, and TF-IDF vectorization.

- **Sentiment Analysis**  
  Determines the positivity or negativity of reviews using TextBlob.

- **Machine Learning Models**  
  Trains and optimizes:
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Logistic Regression
  - Linear Support Vector Classifier (SVC)

- **Ensemble Learning**  
  Combines the best models for stronger and more robust predictions.

- **Interactive Web App**  
  Analyze reviews, view predictions, sentiment scores, word frequencies, and model performance metrics.

- **Memory-Efficient Processing**  
  Processes large datasets in chunks to prevent memory issues.

---

## 🛠️ Technologies Used

- Python
- Scikit-learn
- NLTK & TextBlob
- Streamlit
- Pandas & NumPy
- Matplotlib & Seaborn

---

## 🗂️ Project Structure

```
Fake-Review-Detector/
├── app.py               # Streamlit Web App
├── data_processor.py    # Data Cleaning & Feature Extraction
├── model_trainer.py      # Model Training & Evaluation
├── utils.py              # Utility Functions
├── dataset/              # Dataset Files
├── best_model.joblib     # Trained Model
├── model_metrics.csv     # Model Metrics
├── requirements.txt      # Required Python Packages
```

---

## ⚙️ Setup Instructions

1. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Mac/Linux
   venv\Scripts\activate     # For Windows
   ```

2. **Install the required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the models** (optional if you want to retrain)
   ```bash
   python model_trainer.py
   ```

4. **Run the web app**
   ```bash
   streamlit run app.py
   ```

---

## 📚 Dataset

- **Yelp Academic Dataset**  
  Used for training and evaluating the fake review detector.

---

## 📊 Key Functionalities

- **Review Analysis:**  
  Instantly predict authenticity, view sentiment scores, and explore word frequency graphs.
  
- **Model Performance:**  
  Compare accuracy, precision, recall, and F1 scores across different models.

---

## 📄 License

This project is **open-source** and available for educational and research purposes.

---

## ❤️ Made with Passion

Built with a love for learning, NLP, and intelligent system design!



