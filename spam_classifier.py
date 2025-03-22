import os
import pandas as pd
import re
import nltk
import joblib
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report



# Ensure stopwords are available
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))



# Preprocessing function
def preprocess_text(text):
    """Clean and preprocess email text."""
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Load dataset
dataset_path = "D:/ML/dataset/enron1"
emails, labels = [], []

for folder in ["ham", "spam"]:
    folder_path = os.path.join(dataset_path, folder)
    if not os.path.exists(folder_path):
        print(f"⚠️ Warning: Folder '{folder_path}' not found. Skipping...")
        continue
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, "r", encoding="latin1") as file:
                emails.append(file.read())
                labels.append(1 if folder == "spam" else 0)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

# Convert to DataFrame
df = pd.DataFrame({"Email": emails, "Label": labels})
df["Cleaned_Email"] = df["Email"].apply(preprocess_text)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["Cleaned_Email"], df["Label"], test_size=0.2, random_state=42)

# Convert text to feature vectors using TF-IDF
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

# Train a Naïve Bayes classifier
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Transform the test data
X_test_counts = vectorizer.transform(X_test)
X_test_tfidf = tfidf_transformer.transform(X_test_counts)

# Predict and evaluate
y_pred = model.predict(X_test_tfidf)

joblib.dump(model, "spam_classifier.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")
joblib.dump(tfidf_transformer, "tfidf.pkl")
print("✅ Model saved!")

# Print model accuracy
print("\n✅ Model Accuracy:", accuracy_score(y_test, y_pred))
print("\n🔹 Classification Report:\n", classification_report(y_test, y_pred))

# Test with a sample email
sample_email = ["Congratulations! You've won a free lottery. Click the link to claim."]
sample_tfidf = tfidf_transformer.transform(vectorizer.transform(sample_email))
prediction = model.predict(sample_tfidf)
print("\n📝 Sample Email Classification:", "Spam" if prediction[0] == 1 else "Not Spam")

while True:
    user_input = input("\n📩 Enter an email text to classify (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    user_tfidf = tfidf_transformer.transform(vectorizer.transform([user_input]))
    prediction = model.predict(user_tfidf)
    print("🔹 Prediction:", "🚨 Spam" if prediction[0] == 1 else "✅ Not Spam")

