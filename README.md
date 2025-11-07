# Twitter Sentiment Analysis 

This project is a machine learning model trained to classify tweets as either **positive (1)** or **negative (0)**. It uses the classic [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) from Kaggle, which contains 1.6 million labeled tweets.

The project demonstrates a complete, simple NLP pipeline:
* Text cleaning and preprocessing with **NLTK**
* Feature extraction using **TF-IDF**
* Model training and evaluation with **Scikit-learn**

---

## Results

The final model, a **Logistic Regression** classifier, achieved the following performance on a 30% test split (320,000 tweets):

* **Accuracy:** ~78%

Classification Report:
| | precision | recall | f1-score | support |
| :--- | :--- | :--- | :--- | :--- |
| **Negative** | 0.79 | 0.75 | 0.77 | 239361 |
| **Positive** | 0.76 | 0.80 | 0.78 | 240639 |
| | | | | |
| **accuracy** | | | 0.77 | 480000 |
| **macro avg** | 0.77 | 0.77 | 0.77 | 480000 |
| **weighted avg**| 0.77 | 0.77 | 0.77 | 480000 |

---

##  Installation & Setup

Follow these steps to set up the project on your local machine.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Abyss12122/twitter-sentiment-analysis
    cd sentiment-analysis-project
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # Create the environment
    python -m venv venv
    
    # Activate on Windows
    .\venv\Scripts\activate
    
    # Activate on macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download NLTK data:**
    Run the following command to download the `stopwords` and `wordnet` packages used for text cleaning.
    ```bash
    python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
    ```

---

## How to Run

1.  **Download the Dataset:**
    
    Manually download the `.csv` file from the link above and place it in a `data/` folder.*

2.  **Run the Training Script:**
    (Assuming your main Python file is named `train.py`)
    ```bash
    python train.py
    ```
    The script will load the data, clean it, train the model, and print the final accuracy and classification report to the console.
