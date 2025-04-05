import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import clean_text
from src.model import get_models
from src.evaluate import evaluate_model

def run_pipeline():
    df = pd.read_csv("data/train.csv")
    df['content'] = df['title'].fillna('') + ' ' + df['text'].fillna('')
    df['clean_content'] = df['content'].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df['clean_content']).toarray()
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = get_models()

    best_model = None
    best_accuracy = 0

    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {accuracy * 100:.2f}%")
        print(report)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model

    # Save best model and vectorizer
    with open("models/fake_news_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    print("\n Model training complete and saved!")

