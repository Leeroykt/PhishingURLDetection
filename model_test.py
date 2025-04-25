from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import load_model
import joblib
import pandas as pd
import re
import numpy as np

# Load pre-trained model and scaler
model = load_model('phishing_model.h5')
scaler = joblib.load('scaler.pkl')

# Preprocess function
def preprocess(urls):
    urls = urls.str.lower()

    
    urls = urls.apply(lambda x: re.sub(r'http[s]?://|www\.', '', x))

    # Extract the correct 3 features
    url_length = urls.apply(len)  
    num_dots = urls.apply(lambda x: x.count('.'))  
    num_hyphens = urls.apply(lambda x: x.count('-'))  
   
    features = pd.DataFrame({
        'url_length': url_length,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens
    })

   
    return features.to_numpy()

# Load test data
test_data = pd.read_csv('phishing_site_urls.csv')


test_data['Label'] = test_data['Label'].map({'good': 0, 'bad': 1})

# Preprocess the test data
X_test = preprocess(test_data['URL'])
X_test_scaled = scaler.transform(X_test)

# Model predictions
y_pred = model.predict(X_test_scaled)
y_pred = (y_pred > 0.5).astype(int) 

# Evaluation metrics
print("Classification Report:")
print(classification_report(test_data['Label'], y_pred))
print("Confusion Matrix:")
print(confusion_matrix(test_data['Label'], y_pred))
