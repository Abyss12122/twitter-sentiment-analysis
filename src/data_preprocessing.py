import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --- Download NLTK data (uncomment if running for the first time) ---
# nltk.download('stopwords')
# nltk.download('wordnet')
# ------------------------------------------------------------------

def load_data(file_path):
    print("Loading data...")
    COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
    df = pd.read_csv('D:/Work/Programs/ML Programs/sentiment-analysis-project/data/data.csv', encoding='latin1', names=COLUMNS)
    
    df = df[['target', 'text']]
    
    df['target'] = df['target'].replace(4, 1)
    
    df.dropna(inplace=True)
    
    print("Data loading complete.")
    return df

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text_tokens = text.split()
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in text_tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def preprocess_data(df):
    print("Starting text preprocessing...")
    df['cleaned_text'] = df['text'].apply(clean_text)
    print("Text preprocessing complete.")
    return df