# src/preprocess.py
import re
import nltk
from nltk.corpus import stopwords

try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    # Fallback/Warning if download was missed
    print("WARNING: NLTK 'stopwords' not downloaded. Please run 'python' then 'import nltk' and 'nltk.download('stopwords')'.")
    stop_words = set() 

def clean_text(text):
    """
    Performs basic text cleaning: lowercasing, removing punctuation, 
    and removing stop words.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return " ".join(words)