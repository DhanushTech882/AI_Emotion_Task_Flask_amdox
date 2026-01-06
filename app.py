from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

app = Flask(__name__)

# Training data
texts = [
    "I am very happy today",
    "I feel excited and energetic",
    "This is a wonderful day",
    "I am sad and lonely",
    "I feel depressed",
    "I am stressed with work",
    "I feel anxious and tired",
    "I am angry and frustrated",
    "This makes me mad"
]

emotions = [
    "Happy",
    "Happy",
    "Happy",
    "Sad",
    "Sad",
    "Stress",
    "Stress",
    "Angry",
    "Angry"
]

# Train ML model
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

model = MultinomialNB()
model.fit(X, emotions)

# Task mapping
task_map = {
    "Happy": "Work on creative or challenging tasks",
    "Sad": "Take a break and talk to someone",
    "Stress": "Relax, meditate, or do light work",
    "Angry": "Take deep breaths and calm down"
}

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    task = None

    if request.method == "POST":
        user_text = request.form["text"]

        if user_text.strip() != "":
            user_vector = vectorizer.transform([user_text])
            emotion = model.predict(user_vector)[0]
            task = task_map[emotion]

    return render_template("index.html", emotion=emotion, task=task)

if __name__ == "__main__":
    app.run(debug=True)

