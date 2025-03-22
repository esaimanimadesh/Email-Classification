from flask import Flask, render_template
import joblib
import re
import nltk
from nltk.corpus import stopwords
from fetch_gmail import fetch_emails  # Import the function to get emails

# Download stopwords (only once)
nltk.download("stopwords")

app = Flask(__name__)

# Load trained model and vectorizers
vectorizer = joblib.load("vectorizer.pkl")
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("spam_classifier.pkl")

def preprocess_text(text):
    """Clean and preprocess email text."""
    text = text.lower()
    text = re.sub(r"\W", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\d+", "", text)
    stop_words = set(stopwords.words("english"))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

def classify_emails(emails):
    """Classify emails as spam or not spam."""
    processed_emails = [preprocess_text(email) for email in emails]
    email_vectors = vectorizer.transform(processed_emails)
    email_tfidf = tfidf.transform(email_vectors)
    predictions = model.predict(email_tfidf)
    return predictions

@app.route("/")
def index():
    """Fetch and classify emails, then display them in a webpage."""
    emails = fetch_emails()
    predictions = classify_emails(emails)
    
    email_data = [{"email": email, "label": "Spam 🚨" if pred == 1 else "Not Spam ✅"} 
                  for email, pred in zip(emails, predictions)]

    return render_template("index.html", email_data=email_data)

if __name__ == "__main__":
    app.run(debug=True)
