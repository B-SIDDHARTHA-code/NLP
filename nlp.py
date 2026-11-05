import pandas as pd
import gradio as gr
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Locate and load the dataset
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "text_classification.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {CSV_PATH}\n"
        "Place text_classification.csv in the same folder as nlp.py."
    )

df = pd.read_csv(CSV_PATH)

# 2. Clean column names
df.columns = df.columns.str.strip().str.lower()

print("Dataset columns:", df.columns.tolist())

# 3. Validate required columns
if not {"text", "label"}.issubset(df.columns):
    raise ValueError(
        "Your CSV must contain columns named 'text' and 'label'.\n"
        f"Columns found: {df.columns.tolist()}\n"
        "Rename your CSV columns to text and label."
    )

# 4. Remove missing values and prepare data
df = df.dropna(subset=["text", "label"]).copy()

df["text"] = df["text"].astype(str).str.strip()
df["label"] = df["label"].astype(str).str.strip()

# Remove empty text and labels
df = df[
    (df["text"] != "") &
    (df["label"] != "")
].copy()

# Remove duplicate rows
df = df.drop_duplicates(subset=["text", "label"])

# Check dataset
print("\nDataset preview:")
print(df.head())

print("\nCategory distribution:")
print(df["label"].value_counts())

# 5. Validate the dataset
if len(df) < 10:
    raise ValueError(
        "The dataset needs more examples for a meaningful "
        "train/test split. Add more labelled text rows."
    )

if df["label"].nunique() < 2:
    raise ValueError(
        "The dataset must contain at least two categories."
    )

# Stratified splitting requires at least two samples per category.
if df["label"].value_counts().min() < 2:
    raise ValueError(
        "Every category needs at least two examples. "
        "Add more examples to the underrepresented categories."
    )

# 6. Prepare features and target
X = df["text"]
y = df["label"]

# 7. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 8. Build the NLP classification model
model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )
    ),
    (
        "classifier",
        LogisticRegression(max_iter=1000)
    )
])

# 9. Train the model
print("\nTraining the NLP model...")
model.fit(X_train, y_train)

print("Training completed successfully!")

# 10. Evaluate the model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# 11. Prediction function
def predict_text(text):
    if not text or not text.strip():
        return "Please enter some text."

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]
    confidence = probabilities.max() * 100

    return (
        f"Predicted Category: {prediction}\n"
        f"Model Confidence: {confidence:.2f}%"
    )

# 12. Create the Gradio interface
demo = gr.Interface(
    fn=predict_text,
    inputs=gr.Textbox(
        lines=5,
        label="Enter Text",
        placeholder="Enter a sentence or message..."
    ),
    outputs=gr.Textbox(
        label="Prediction Result"
    ),
    title="NLP Text Classification",
    description=(
        "Classify text using Natural Language Processing, "
        "TF-IDF feature extraction, and Logistic Regression."
    )
)

# 13. Launch the application
if __name__ == "__main__":
    demo.launch()
