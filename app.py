import nltk
nltk.download('stopwords')
from flask import Flask, request, render_template
import pickle
from preprocess import clean_text

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model/model.pkl', 'rb'))
vectorizer = pickle.load(open('model/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    
    cleaned = clean_text(news)
    vector = vectorizer.transform([cleaned]).toarray()
    
    prediction = model.predict(vector)[0]
    
    if prediction == 1:
        result = "Real News 🟢"
    else:
        result = "Fake News 🔴"
    
    return render_template('index.html', prediction_text=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)