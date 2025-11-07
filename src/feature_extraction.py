from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib 

def split_data(df):
    print("Splitting data...")
    X = df['cleaned_text']
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split complete.")
    return X_train, X_test, y_train, y_test

def vectorize_text(X_train, X_test, model_path="models/tfidf_vectorizer.joblib"):
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    
    X_test_tfidf = vectorizer.transform(X_test)
    
    joblib.dump(vectorizer, model_path)
    
    print(f"Text vectorization complete. Vectorizer saved to {model_path}")
    return X_train_tfidf, X_test_tfidf, vectorizer