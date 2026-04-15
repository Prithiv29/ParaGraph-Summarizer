# pip install sentence-transformers
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

df = pd.read_csv("para sumarizer (1) (1).csv")
X_text = df['Sentence'].tolist()
y = df['Label'].values

# Replace TF-IDF with semantic embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')  # free, runs locally, no API
X = embedder.encode(X_text, show_progress_bar=True)
# X is now (1036, 384) — 384 meaning-aware features per sentence

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

