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
    # NOTE: vocab is read globally here
    if vocab is None:
        return np.zeros(0) # Handle case where vocab isn't initialized
        
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
    # prevent overflow
    z = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z))

# -------------------------------
# Train model (Logistic Regression via Gradient Descent)
# -------------------------------
def train_model():
    """Loads data, builds vocabulary, trains the Logistic Regression model, and updates global parameters."""
    # FIX: Declare globals so we update the module-level variables
    global W, b, vocab, word_to_index 

    print("📥 Loading and preparing dataset...")
    
    # NOTE: Assuming 'True.csv' and 'Fake.csv' are available in the execution environment
    try:
        # Load datasets
        true_df = pd.read_csv("True.csv")
        fake_df = pd.read_csv("Fake.csv")
    except FileNotFoundError:
        print("ERROR: True.csv or Fake.csv not found. Cannot train model.")
        W, b, vocab, word_to_index = None, None, None, None # Ensure they remain None on failure
        return 
        
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

    # Use top N frequent words to reduce noise
    N = 5000 
    vocab = [word for word, _ in Counter(all_words).most_common(N)]
    word_to_index = {word: i for i, word in enumerate(vocab)}
    
    # -------------------------------
    # Convert to vectors
    # -------------------------------
    print(f"Creating feature vectors using {len(vocab)} unique words...")
    X = np.array([text_to_vector(t) for t in df["text"]])
    y = np.array(df["label"]).reshape(-1, 1)

    # -------------------------------
    # Split data (train/test)
    # -------------------------------
    split = int(0.85 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # -------------------------------
    # Initialize parameters
    # -------------------------------
    m, n = X_train.shape
    W = np.zeros((n, 1)) # Now this correctly assigns to the global W
    b = 0                # Now this correctly assigns to the global b
    lr = 0.005 # TUNING: Increased learning rate for faster convergence
    epochs = 300 

    print("Training started (Logistic Regression via Gradient Descent)...")
    for epoch in range(epochs):
        z = np.dot(X_train, W) + b
        A = sigmoid(z)

        # Cost function (Binary Cross-Entropy)
        cost = -(1/m) * np.sum(y_train*np.log(A+1e-8) + (1-y_train)*np.log(1-A+1e-8))

        # Gradients
        dW = (1/m) * np.dot(X_train.T, (A - y_train))
        db = (1/m) * np.sum(A - y_train)

        # Weight update
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
    print(f"\n Training complete! Test Accuracy: {acc*100:.2f}%")

# -------------------------------
# Predict input text
# -------------------------------
def predict_input(text):
    """Predicts the label and confidence score for a single piece of text."""
    # NOTE: Read globals W, b, vocab
    if W is None or vocab is None:
        print("Model not trained or missing parameters.")
        return "⚠️ PENDING", 0.5 

    cleaned = clean_text(text)
    vec = text_to_vector(cleaned).reshape(1, -1)
    
    if vec.shape[1] != len(vocab):
        # This handles the case where text_to_vector failed to return the correct size
        return "⚠️ PENDING", 0.5

    # prob is the probability of being REAL (label=1)
    prob = sigmoid(np.dot(vec, W) + b)[0][0]
    
    # TUNING: Adjusted confidence thresholds for clearer, symmetric boundaries.
    HIGH_CONFIDENCE_THRESHOLD = 0.75
    LOW_CONFIDENCE_THRESHOLD = 0.25
    
    # Determine the label
    if prob >= HIGH_CONFIDENCE_THRESHOLD:
        label = "TRUE NEWS (High Confidence)"
    elif prob >= 0.5:
        label = "TRUE NEWS (Moderate Confidence)"
    elif prob < LOW_CONFIDENCE_THRESHOLD:
        label = "FAKE NEWS (High Confidence)"
    else: # 0.25 <= prob < 0.5
        label = "FAKE NEWS (Moderate Confidence)"
        
    return label, prob

# -------------------------------
# Run standalone (for debugging/testing)
# -------------------------------
if __name__ == "__main__":
    train_model()
