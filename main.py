from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# 💡 Load Hugging Face MarianMT model and tokenizer
model_name = 'Helsinki-NLP/opus-mt-en-hi'
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# 🏠 Serve HTML frontend
@app.route('/')
def home():
    return render_template("index.html")

# 📬 Translation API
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)

    return jsonify({"translated_text": output})

# ✅ Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
    from gtts import gTTS
import os

@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    output = tokenizer.decode(translated[0], skip_special_tokens=True)

    # Create speech
    tts = gTTS(text=output, lang='hi')
    tts.save("static/output.mp3")

    return jsonify({
        "translated_text": output,
        "audio_url": "/static/output.mp3"
    })

