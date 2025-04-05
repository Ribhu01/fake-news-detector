# 📰 Fake News Detection App

This project is a Machine Learning-based web application that detects whether a given news article is **real** or **fake** using Natural Language Processing (NLP). The model is trained using a dataset from Kaggle and deployed as a Streamlit web app.

👉 **Live Demo**: [Fake News Detector](https://fake-news-detector-c9wijljfz3y8xgpvq2gxvy.streamlit.app)

---

## 🚀 Features

- 🧹 Text preprocessing (cleaning, stemming, stopword removal)
- 🧠 Multiple ML models for training and evaluation
- 📈 Model comparison and selection based on accuracy
- 🖥️ Simple and intuitive Streamlit web interface
- ✅ Clean modular pipeline and project structure

---

## 📁 Project Structure

```
fake-news-detector/
├── app.py                  # Streamlit frontend app
├── requirements.txt        # All project dependencies
├── models/
│   ├── fake_news_model.pkl # Trained ML model
│   └── vectorizer.pkl      # TF-IDF vectorizer
├── src/
│   ├── preprocess.py       # Text preprocessing (cleaning, stemming)
│   ├── main.py             # Runs the full pipeline
│   ├── model.py            # Defines & compares ML models
│   ├── pipeline.py         # Preprocessing, vectorization, splitting
│   └── evaluate.py         # Evaluation metrics, confusion matrix, etc.
└── README.md               # Project documentation
```

---

## ⚙️ How to Run Locally

1. **Clone the repo**
   ```bash
   git clone https://github.com/your-username/fake-news-detector.git
   cd fake-news-detector
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLTK resources**
   ```python
   import nltk
   nltk.download('stopwords')
   ```

4. **Train the model (Optional)**
   ```bash
   python src/main.py
   ```

5. **Run the Streamlit app**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset

The project uses the **Fake and Real News Dataset** from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset), which includes:
- `title`, `text`, and `label` columns
- Binary labels (0 = Fake, 1 = Real)

---

## 📦 Dependencies

Key libraries used:
- `scikit-learn`
- `nltk`
- `streamlit`
- `pandas`
- `pickle`

All dependencies are listed in [`requirements.txt`](./requirements.txt).

---

## 💡 Future Improvements

- Add deep learning models (e.g., LSTM or BERT)
- Enhance preprocessing with lemmatization
- Add confidence scores and better explanations in output
- Deploy on platforms like Hugging Face or AWS

---

## 📬 Contact
Feel free to reach out or raise an issue if you find a bug or want to contribute!
