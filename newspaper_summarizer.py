from flask import Flask, request, jsonify, render_template
import re
from collections import Counter
import numpy as np
from Fake_Detection import train_model, predict_input  # ✅ lowercase file name

# ✅ Train fake news model once on startup and store weights/vocab
W, b, vocab, word_to_index = train_model()

app = Flask(__name__)

# --------------------------
# Text Summarizer
# --------------------------
class TextSummarizer:
    def __init__(self):
        self.stop_words = {
            'a','an','the','and','or','but','in','on','at','to','for','of','with','by',
            'is','are','was','were','be','been','being','have','has','had','do','does',
            'did','will','would','could','should','may','might','must','can','it','its',
            'they','them','their','this','that','these','those','i','you','he','she','we'
        }

    def preprocess_text(self, text):
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        tokenized_sentences = []
        for s in sentences:
            cleaned = re.sub(r'[^\w\s]', '', s.lower())
            words = [w for w in cleaned.split() if w not in self.stop_words and len(w) > 2]
            tokenized_sentences.append(words)
        return tokenized_sentences, sentences

    def score_sentences(self, tokenized_sentences):
        word_freq = Counter(w for s in tokenized_sentences for w in s)
        if not word_freq:
            return [0 for _ in tokenized_sentences]
        max_freq = max(word_freq.values())
        for w in word_freq:
            word_freq[w] /= max_freq
        sentence_scores = []
        for sentence in tokenized_sentences:
            if not sentence:
                sentence_scores.append(0)
                continue
            score = sum(word_freq[w] for w in sentence) / len(sentence)
            sentence_scores.append(score)
        return sentence_scores

    def summarize(self, text, summary_length=3):
        tokenized, sentences = self.preprocess_text(text)
        if not sentences:
            return "No meaningful content to summarize."
        summary_length = min(summary_length, len(sentences))
        scores = self.score_sentences(tokenized)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        selected = []
        selected_words = set()
        for idx, _ in ranked:
            words = set(tokenized[idx])
            overlap = len(words & selected_words) / max(len(words), 1)
            if overlap < 0.7:
                selected.append(idx)
                selected_words.update(words)
            if len(selected) >= summary_length:
                break
        selected.sort()
        return ' '.join([sentences[i] for i in selected])


# --------------------------
# Text Analyzer
# --------------------------
class TextAnalyzer:
    @staticmethod
    def calculate_readability(text):
        if not text or not text.strip():
            return 0
        sentences = re.split(r'(?<!\w\.\w.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text.lower())
        word_count = len(words)
        if len(sentences) == 0 or word_count == 0:
            return 0
        avg_sentence_length = word_count / len(sentences)
        avg_word_length = sum(len(word) for word in words) / word_count
        readability = 206.835 - (1.015 * avg_sentence_length) - (84.6 * (avg_word_length / 5))
        return max(0, min(100, readability))

    @staticmethod
    def get_text_stats(text):
        if not text or not text.strip():
            return {'sentences': 0, 'words': 0, 'characters': 0, 'paragraphs': 0}
        sentences = re.split(r'(?<!\w\.\w.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        characters = len(re.sub(r'\s', '', text))
        return {
            'sentences': len(sentences),
            'words': len(words),
            'characters': characters,
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()])
        }

summarizer = TextSummarizer()
analyzer = TextAnalyzer()

# --------------------------
# Flask Routes
# --------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text', '')
    length = data.get('length', 3)
    summary = summarizer.summarize(text, length)
    return jsonify({'summary': summary})

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text', '')
    stats = analyzer.get_text_stats(text)
    readability = analyzer.calculate_readability(text)

    # ✅ Use global W, vocab to check if model is ready
    if W is not None and vocab is not None and text.strip():
        prediction_label, confidence_score = predict_input(text)
    else:
        prediction_label = "⚠️ PENDING"
        confidence_score = 0.5

    return jsonify({
        'words': stats['words'],
        'sentences': stats['sentences'],
        'paragraphs': stats['paragraphs'],
        'characters': stats['characters'],
        'readability': readability,
        'predictionLabel': prediction_label,
        'confidenceScore': confidence_score
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
