import pandas as pd
from preprocess import clean_text

# Load datasets
fake = pd.read_csv('data/fake.csv')
true = pd.read_csv('data/true.csv')

fake['label'] = 0
true['label'] = 1

# Combine datasets
df = pd.concat([fake, true], axis=0)
df = df.sample(frac=1)

# Create content column
df['content'] = df['title'] + " " + df['text']

# Clean text
df['content'] = df['content'].apply(clean_text)

# -------------------------------
# TF-IDF (Text → Numbers)
# -------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['content']).toarray()
y = df['label']

# -------------------------------
# Train-Test Split
# -------------------------------
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

# -------------------------------
# Train Model
# -------------------------------
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------------
# Evaluate Model
# -------------------------------
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# -------------------------------
# Save Model
# -------------------------------
import pickle

pickle.dump(model, open('model/model.pkl', 'wb'))
pickle.dump(vectorizer, open('model/vectorizer.pkl', 'wb'))