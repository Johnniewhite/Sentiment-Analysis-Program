import transformers
from flask import Flask, request, jsonify
from transformers import RobertaTokenizerFast, TFRobertaForSequenceClassification, pipeline

app = Flask(__name__)

# Load model and tokenizer once at app startup
tokenizer = RobertaTokenizerFast.from_pretrained("arpanghoshal/EmoRoBERTa")
model = TFRobertaForSequenceClassification.from_pretrained("arpanghoshal/EmoROBERTa")
emotion = pipeline("sentiment-analysis", model="arpanghoshal/EmoROBERTa")

@app.route("/analyze_sentiment", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    phrase = data.get("phrase")

    if phrase:
        emotion_labels = emotion(phrase)
        return jsonify({"emotion_labels": emotion_labels})
    else:
        return jsonify({"error": "Missing phrase"}), 400

if __name__ == "__main__":
    app.run(debug=True)  # Set debug=False in production
