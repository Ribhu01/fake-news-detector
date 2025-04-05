from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def get_models():
    return {
        'Logistic Regression': LogisticRegression(max_iter=1000),
        'Naive Bayes': MultinomialNB(),
        'SVM (Linear)': LinearSVC(),
        'Random Forest': RandomForestClassifier(n_estimators=100)
    }
    