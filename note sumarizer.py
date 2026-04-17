import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("para sumarizer (1) (1).csv")
x1 = df['Sentence']
y = df['Label'].values

vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2), stop_words='english')
x = vectorizer.fit_transform(x1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

rmodel = RandomForestClassifier(n_estimators=100, random_state=42)
rmodel.fit(x_train, y_train)

y_pred = rmodel.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(df['Label'].value_counts())

joblib.dump(rmodel, "model.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


def summarize(paragraph, model, vectorizer, top_n=3):
    sentences = [s.strip() for s in paragraph.split(".") if s.strip()]

    x_input = vectorizer.transform(sentences)

    proba = model.predict_proba(x_input)[:, 1]
    ranked = sorted(zip(proba, sentences), reverse=True)
    top = [s for score, s in ranked if score > 0.5][:top_n]

    return ". ".join(top) + "."


paragraph = """Ghosts are often described as the spirits of people who have died.
Many cultures believe ghosts remain on Earth due to unfinished business.
They are commonly associated with haunted places like old houses and cemeteries.
Some people claim ghosts can be seen as shadows or transparent figures.
Others believe ghosts can only be felt through sudden chills or strange sounds.
Stories about ghosts have been passed down through generations.
Ghosts are a popular subject in horror movies and books.
Some people try to communicate with ghosts using tools like Ouija boards.
There are many different types of ghosts in folklore.
Some ghosts are believed to be friendly, while others are considered dangerous.
Paranormal investigators study ghost sightings and unexplained events.
Science has not found solid evidence proving ghosts exist.
Many ghost sightings can be explained by psychological or environmental factors.
Fear of ghosts is known as phasmophobia.
Ghost stories are often used to scare or entertain people.
Some believe ghosts appear more often at night."""

summary = summarize(paragraph, rmodel, vectorizer, top_n=4)
print("\nSummary:")
print(summary)
