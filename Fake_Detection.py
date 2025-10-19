import numpy as np
import pandas as pd
import re
from collections import Counter

# -------------------------------
# Global variables (Initialized by train_model)
# -------------------------------
W, b, vocab, word_to_index = None, None, None, None


# -------------------------------
# Text cleaning
# -------------------------------
def clean_text(text):
    """Converts text to lowercase, removes punctuation/numbers, and strips whitespace."""
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -------------------------------
# Convert text to vector (Feature extraction)
# -------------------------------
def text_to_vector(text):
    """Converts cleaned text into a bag-of-words feature vector based on the trained vocabulary."""
    global vocab, word_to_index
    if vocab is None:
        return np.zeros(0)

    vec = np.zeros(len(vocab))
    for word in text.split():
        if word in word_to_index:
            vec[word_to_index[word]] += 1
    return vec


# -------------------------------
# Sigmoid activation function
# -------------------------------
def sigmoid(z):
    """Logistic sigmoid function, clipped to prevent numerical overflow."""
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))


# -------------------------------
# Train model (Logistic Regression via Gradient Descent)
# -------------------------------
def train_model():
    """Loads data, builds vocabulary, trains the Logistic Regression model, and updates global parameters."""
    global W, b, vocab, word_to_index

    print("📥 Loading and preparing dataset...")

    try:
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
    except FileNotFoundError:
        print("❌ ERROR: True.csv or Fake.csv not found. Cannot train model.")
        W, b, vocab, word_to_index = None, None, None, None
        return None, None, None, None

    true_df["label"] = 1
    fake_df["label"] = 0

    df = pd.concat([true_df, fake_df], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df["text"] = df["text"].apply(clean_text)

    # -------------------------------
    # Build vocabulary
    # -------------------------------
    all_words = []
    for t in df["text"]:
        all_words.extend(t.split())

    N = 5000
    vocab = [word for word, _ in Counter(all_words).most_common(N)]
    word_to_index = {word: i for i, word in enumerate(vocab)}

    # -------------------------------
    # Convert to vectors
    # -------------------------------
    print(f"🔤 Creating feature vectors using {len(vocab)} unique words...")
    X = np.array([text_to_vector(t) for t in df["text"]])
    y = np.array(df["label"]).reshape(-1, 1)

    # -------------------------------
    # Split data
    # -------------------------------
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -------------------------------
    # Initialize parameters
    # -------------------------------
    m, n = X_train.shape
    W = np.zeros((n, 1))
    b = 0
    lr = 0.005
    epochs = 300

    print("⚙️ Training Logistic Regression model...")
    for epoch in range(epochs):
        z = np.dot(X_train, W) + b
        A = sigmoid(z)

        cost = -(1 / m) * np.sum(y_train * np.log(A + 1e-8) + (1 - y_train) * np.log(1 - A + 1e-8))
        dW = (1 / m) * np.dot(X_train.T, (A - y_train))
        db = (1 / m) * np.sum(A - y_train)

        W -= lr * dW
        b -= lr * db

        if epoch % 50 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:03d} | Cost: {cost:.4f}")

    # -------------------------------
    # Evaluate model
    # -------------------------------
    z_test = np.dot(X_test, W) + b
    preds = sigmoid(z_test) > 0.5
    acc = np.mean(preds == y_test)
    print(f"\n✅ Training complete! Test Accuracy: {acc * 100:.2f}%")

    # ✅ Return trained model components
    return W, b, vocab, word_to_index


# -------------------------------
# Predict input text
# -------------------------------
def predict_input(text):
    """Predicts the label and confidence score for a single piece of text."""
    global W, b, vocab

    if W is None or vocab is None:
        print("⚠️ Model not trained or missing parameters.")
        return "⚠️ PENDING", 0.5

    cleaned = clean_text(text)
    vec = text_to_vector(cleaned).reshape(1, -1)

    if vec.shape[1] != len(vocab):
        return "⚠️ PENDING", 0.5

    prob = sigmoid(np.dot(vec, W) + b)[0][0]

    HIGH_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD = 0.25

    if prob >= HIGH_CONFIDENCE_THRESHOLD:
        label = "TRUE NEWS (High Confidence)"
    elif prob >= 0.5:
        label = "TRUE NEWS (Moderate Confidence)"
    elif prob < LOW_CONFIDENCE_THRESHOLD:
        label = "FAKE NEWS (High Confidence)"
    else:
        label = "FAKE NEWS (Moderate Confidence)"

    return label, float(prob)


# -------------------------------
# Run standalone
# -------------------------------
if __name__ == "__main__":
    W, b, vocab, word_to_index = train_model()
    print("Model trained and ready for prediction!")
