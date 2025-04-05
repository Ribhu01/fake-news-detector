import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

nltk.download('stopwords')
stopword = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = re.sub('[^a-zA-Z]', ' ', str(text))
    text = text.lower().split()
    text = [stemmer.stem(word) for word in text if word not in stopword]
    return ' '.join(text)

