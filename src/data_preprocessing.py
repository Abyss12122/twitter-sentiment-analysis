import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import contractions  
import emoji         
import logging       

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Download NLTK data (uncomment if running for the first time) ---
# nltk.download('stopwords')
# nltk.download('wordnet')
# ------------------------------------------------------------------

def load_data(file_path):
    logging.info(f"Loading data from {file_path}...")
    COLUMNS = ['target', 'ids', 'date', 'flag', 'user', 'text']
    ENCODING = "latin-1" 
    
    try:
        df = pd.read_csv(file_path, encoding=ENCODING, names=COLUMNS)
        
        df = df[['target', 'text']]
        
        df['target'] = df['target'].replace(4, 1)
        
        df.dropna(inplace=True)
        
        logging.info("Data loading complete.")
        return df
    except FileNotFoundError:
        logging.error(f"Error: Data file not found at {file_path}")
        return None

def clean_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    text = str(text).lower()
    
    text = contractions.fix(text)
    
    text = emoji.demojize(text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    text = re.sub(r'\@\w+', '', text)
    
    text = re.sub(r'#', '', text)
    
    text = re.sub(r'[^a-z\s]', '', text)
    
    text = re.sub(r'(.)\1+', r'\1', text)
    
    text_tokens = text.split()
    
    cleaned_tokens = [lemmatizer.lemmatize(word) for word in text_tokens if word not in stop_words]
    
    return " ".join(cleaned_tokens)

def preprocess_data(df):
    if 'text' not in df.columns:
        logging.error("DataFrame does not have a 'text' column.")
        return df
        
    logging.info("Starting advanced text preprocessing...")
    
    df['cleaned_text'] = df['text'].apply(clean_text)
    
    logging.info("Advanced text preprocessing complete.")
    return df