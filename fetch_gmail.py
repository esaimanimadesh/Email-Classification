import os
import pickle
import base64
import joblib
import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Download NLTK stopwords (only needed once)
nltk.download("stopwords")

# Set up Gmail API
SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]

def get_gmail_service():
    """Authenticate and return the Gmail API service."""
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
        creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return build("gmail", "v1", credentials=creds)

def fetch_emails():
    """Fetch the latest 5 emails from Gmail."""
    service = get_gmail_service()
    results = service.users().messages().list(userId="me", maxResults=5).execute()
    messages = results.get("messages", [])

    emails = []
    for msg in messages:
        msg_data = service.users().messages().get(userId="me", id=msg["id"]).execute()
        payload = msg_data["payload"]
        headers = payload["headers"]

        subject = next((h["value"] for h in headers if h["name"] == "Subject"), "No Subject")
        snippet = msg_data.get("snippet", "")

        emails.append(subject + " " + snippet)  # Combine subject and snippet

    return emails

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

# Load trained spam classifier model
vectorizer = joblib.load("vectorizer.pkl")
tfidf = joblib.load("tfidf.pkl")
model = joblib.load("spam_classifier.pkl")

def classify_emails(emails):
    """Classify emails as spam or not spam."""
    processed_emails = [preprocess_text(email) for email in emails]
    email_vectors = vectorizer.transform(processed_emails)
    email_tfidf = tfidf.transform(email_vectors)
    predictions = model.predict(email_tfidf)
    return predictions

# Fetch and classify emails
emails = fetch_emails()
predictions = classify_emails(emails)

# Print classification results
for i, email in enumerate(emails):
    label = "SPAM 🚨" if predictions[i] == 1 else "NOT SPAM ✅"
    print(f"\n📩 Email {i+1}: {label}\n{email}")
