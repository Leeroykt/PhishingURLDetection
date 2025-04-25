import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import joblib

# Load dataset
data = pd.read_csv('phishing_site_urls.csv')

# Lowercase the URL column for consistency
data['URL'] = data['URL'].str.lower()

# --- Label preprocessing ---
print("Original labels:", data['Label'].unique())

# Clean and map labels
data['Label'] = data['Label'].str.strip().str.lower()
label_map = {'good': 0, 'bad': 1}  # Use your actual label values here
data['Label'] = data['Label'].map(label_map)

# Drop rows with invalid or missing labels
data.dropna(subset=['Label'], inplace=True)
data['Label'] = data['Label'].astype(int)

print("✅ Cleaned dataset size:", len(data))

# --- Feature extraction ---
def extract_features(url):
    url = re.sub(r'http[s]?://|www\.', '', url)
    features = {
        'url_length': len(url),
        'num_dots': url.count('.'),
        'num_special_chars': sum(not c.isalnum() for c in url),
        'has_ip': 1 if re.search(r'(\d{1,3}\.){3}\d{1,3}', url) else 0
    }
    return features

features_df = data['URL'].apply(extract_features).apply(pd.Series)
X = features_df
y = data['Label']

# --- Split dataset ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Feature scaling ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save the scaler
joblib.dump(scaler, 'scaler.pkl')

# --- Build and train the model ---
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

# Save the model
model.save('phishing_model.h5')
print("✅ Model trained and saved to phishing_model.h5")
