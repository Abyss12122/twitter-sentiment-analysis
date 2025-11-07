from src.data_preprocessing import load_data, preprocess_data
from src.feature_extraction import split_data, vectorize_text
from src.train_model import train_model, evaluate_model
import os

RAW_DATA_PATH = 'data/raw/training.1600000.processed.noemoticon.csv'

MODEL_DIR = 'models/'
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
MODEL_PATH = os.path.join(MODEL_DIR, 'logistic_regression_v1.joblib')

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

def main():
    df = load_data(RAW_DATA_PATH)
    df_clean = preprocess_data(df)
    
    df_clean.to_csv('data/processed/cleaned_tweets.csv', index=False)
    
    X_train, X_test, y_train, y_test = split_data(df_clean)
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test, VECTORIZER_PATH)
    
    model = train_model(X_train_vec, y_train, MODEL_PATH)
    
    evaluate_model(model, X_test_vec, y_test)

if __name__ == "__main__":
    main()