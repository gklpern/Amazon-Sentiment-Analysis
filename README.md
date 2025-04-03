# Amazon-Sentiment-Analysis
This project performs Sentiment Analysis on Amazon reviews using Natural Language Processing (NLP) and Deep Learning techniques. A LSTM-based model is trained to classify reviews as positive or negative.

📥 Downloading the Dataset from Kaggle

If you are unable to manually upload datasets, you can automatically download the latest version of the Amazon Reviews Dataset from Kaggle using the kagglehub library.
How to Download the Dataset

1️⃣ Install kagglehub if you haven't already:

pip install kagglehub

2️⃣ Run the following script to download the dataset:

import kagglehub

# Download the latest version of the dataset
path = kagglehub.dataset_download("bittlingmayer/amazonreviews")

print("Path to dataset files:", path)

3️⃣ Once the download is complete, the dataset files will be saved in the directory specified by path. You can now use them in your project.
Important Notes:

    Ensure you have access to the Kaggle API.

    You may need to authenticate using your Kaggle API token.

    If you encounter any issues, try running kagglehub.authenticate() before downloading.

This method ensures that you always get the latest dataset directly from Kaggle without needing manual uploads. 🚀


🛠 Model Performance

Precision, Recall, F1-Score, and Accuracy:
            precision    recall  f1-score   support

           0       0.95      0.93      0.94    205496
           1       0.93      0.95      0.94    194504

    accuracy                           0.94    400000
   macro avg       0.94      0.94      0.94    400000
weighted avg       0.94      0.94      0.94    400000



📌 Features

    Dataset: The project uses Amazon reviews from train.ft.txt.bz2 and test.ft.txt.bz2 files.

    Data Cleaning: Special characters, numbers, stopwords, and URLs are removed from text.

    Tokenization: Text is converted into numerical form using Keras Tokenizer.

    Model Training: An LSTM model is trained using Adam optimizer and binary_crossentropy loss function.

    Evaluation: The model's performance is analyzed using precision, recall, f1-score, and accuracy.

    User Testing: The model can quickly predict sentiment for given text inputs.



    📦 Amazon-Sentiment-Analysis
 ┣ 📂 data
 ┃ ┣ 📄 amazon_reviews_train.csv
 ┃ ┣ 📄 amazon_reviews_test.csv
 ┣ 📂 models
 ┃ ┣ 📄 my_model.h5  # Trained model
 ┣ 📜 sentiment_analysis.py  # Main model script
 ┣ 📜 requirements.txt  # Dependencies
 ┣ 📜 README.md  # Project documentation




pip install -r requirements.txt
python sentiment_analysis.py


🚀 Example Usage

The following Python script loads the model and predicts sentiment for a given text:

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re

model = load_model("models/my_model.h5")

def clean_text(text):
    text = text.lower().strip()
    text = re.sub(r'https?:\/\/\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text

def predict_sentiment(text, tokenizer, maxlen=100):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=maxlen, padding='post')
    prediction = model.predict(padded_sequence)
    return "Positive" if prediction > 0.5 else "Negative"

tokenizer = ...  # Load or retrain tokenizer

# Test prediction
print(predict_sentiment("This product is amazing!", tokenizer))




📌 Development Stages

✅ Data cleaning and preprocessing completed.
✅ LSTM model trained and evaluated.
✅ Real-time prediction on user input is functional.
🔜 Model optimization and additional data augmentation can be implemented.
