from flask import Flask, request, jsonify
import re
import math
from collections import Counter, defaultdict

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

    # ✅ FIXED: First split sentences, THEN clean
    def preprocess_text(self, text):
        # Split text into sentences using punctuation
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]

        tokenized_sentences = []
        for s in sentences:
            # Clean and tokenize
            cleaned = re.sub(r'[^\w\s]', '', s.lower())
            words = [w for w in cleaned.split() if w not in self.stop_words and len(w) > 2]
            tokenized_sentences.append(words)

        return tokenized_sentences, sentences

    # ✅ IMPROVED: Normalized word frequencies and sentence scoring
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
            # Average normalized word frequency per sentence
            score = sum(word_freq[w] for w in sentence) / len(sentence)
            sentence_scores.append(score)
        return sentence_scores

    # ✅ Better ranking logic with redundancy control
    def summarize(self, text, summary_length=3):
        tokenized, sentences = self.preprocess_text(text)
        if not sentences:
            return "No meaningful content to summarize."
        summary_length = min(summary_length, len(sentences))
        scores = self.score_sentences(tokenized)

        # Rank by score (descending)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        selected = []
        selected_words = set()
        for idx, _ in ranked:
            words = set(tokenized[idx])
            # Allow some overlap, but not too much
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
            return {'sentences':0,'words':0,'characters':0,'paragraphs':0}
        sentences = re.split(r'(?<!\w\.\w.)(?<=\.|\?|\!)\s', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = re.findall(r'\b\w+\b', text)
        characters = len(text.replace(' ', '').replace('\n',''))
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
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Newspaper Summarizer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box;}
body{font-family:'Segoe UI',sans-serif;background:linear-gradient(135deg,#667eea 0%,#764ba2 100%);min-height:100vh;padding:20px;}
.container{max-width:1200px;margin:0 auto;}
.header{text-align:center;color:white;margin-bottom:30px;}
.header h1{font-size:2.5rem;margin-bottom:10px;text-shadow:2px 2px 4px rgba(0,0,0,0.3);}
.header p{font-size:1.1rem;opacity:0.9;}
.main-content{display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:30px;}
@media(max-width:768px){.main-content{grid-template-columns:1fr;}}
.panel{background:white;border-radius:15px;padding:25px;box-shadow:0 10px 30px rgba(0,0,0,0.2);}
.panel h2{color:#2c3e50;margin-bottom:20px;font-size:1.5rem;border-bottom:2px solid #3498db;padding-bottom:10px;}
textarea{width:100%;height:300px;border:2px solid #bdc3c7;border-radius:10px;padding:15px;font-size:14px;resize:vertical;transition:border-color 0.3s;}
textarea:focus{outline:none;border-color:#3498db;}
.controls{background:white;border-radius:15px;padding:25px;box-shadow:0 10px 30px rgba(0,0,0,0.2);margin-bottom:30px;}
.control-group{display:flex;gap:20px;align-items:center;flex-wrap:wrap;}
.btn{background:#27ae60;color:white;border:none;padding:12px 25px;border-radius:8px;cursor:pointer;font-size:16px;font-weight:bold;transition:all 0.3s;}
.btn:hover{background:#219a52;transform:translateY(-2px);}
.btn-secondary{background:#e74c3c;}
.btn-secondary:hover{background:#c0392b;}
.length-control{display:flex;align-items:center;gap:10px;}
.length-control input{width:80px;padding:8px;border:2px solid #bdc3c7;border-radius:5px;text-align:center;}
.stats{background:#34495e;color:white;border-radius:15px;padding:20px;box-shadow:0 10px 30px rgba(0,0,0,0.2);}
.stats-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:15px;text-align:center;}
.stat-item{padding:15px;background:rgba(255,255,255,0.1);border-radius:10px;}
.stat-value{font-size:1.8rem;font-weight:bold;color:#1abc9c;margin-bottom:5px;}
.stat-label{font-size:0.9rem;opacity:0.8;}
.summary-output{background:#fff9e6;border:2px solid #f39c12;border-radius:10px;padding:20px;min-height:200px;font-size:14px;line-height:1.6;}
.file-upload{margin-bottom:15px;}
.file-upload input{display:none;}
.file-upload label{background:#3498db;color:white;padding:10px 20px;border-radius:5px;cursor:pointer;display:inline-block;transition:background 0.3s;}
.file-upload label:hover{background:#2980b9;}
</style>
</head>
<body>
<div class="container">
<div class="header"><h1>📰 Newspaper Summarizer</h1><p>AI-powered text summarization without external libraries</p></div>
<div class="main-content">
<div class="panel">
<h2>Input Text</h2>
<div class="file-upload"><input type="file" id="fileInput" accept=".txt"><label for="fileInput">📁 Upload Text File</label></div>
<textarea id="inputText" placeholder="Paste your newspaper article here or upload a text file..."></textarea>
</div>
<div class="panel">
<h2>Summary & Analysis</h2>
<div class="summary-output" id="summaryOutput">Your summary will appear here...</div>
</div>
</div>
<div class="controls">
<div class="control-group">
<div class="length-control"><label>Summary Length:</label><input type="number" id="summaryLength" min="1" max="10" value="3"></div>
<button class="btn" onclick="generateSummary()">🔍 Generate Summary</button>
<button class="btn btn-secondary" onclick="clearText()">🗑️ Clear Text</button>
</div>
</div>
<div class="stats">
<div class="stats-grid" id="statsGrid">
<div class="stat-item"><div class="stat-value" id="wordCount">0</div><div class="stat-label">Words</div></div>
<div class="stat-item"><div class="stat-value" id="sentenceCount">0</div><div class="stat-label">Sentences</div></div>
<div class="stat-item"><div class="stat-value" id="paragraphCount">0</div><div class="stat-label">Paragraphs</div></div>
<div class="stat-item"><div class="stat-value" id="charCount">0</div><div class="stat-label">Characters</div></div>
<div class="stat-item"><div class="stat-value" id="readabilityScore">0</div><div class="stat-label">Readability</div></div>
</div>
</div>
</div>

<script>
function updateStats(){
const text=document.getElementById('inputText').value;
if(!text.trim()){resetStats();return;}
fetch('/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text})})
.then(response=>response.json())
.then(data=>{
document.getElementById('wordCount').textContent=data.words;
document.getElementById('sentenceCount').textContent=data.sentences;
document.getElementById('paragraphCount').textContent=data.paragraphs;
document.getElementById('charCount').textContent=data.characters;
document.getElementById('readabilityScore').textContent=data.readability.toFixed(1);
});
}
function resetStats(){
document.getElementById('wordCount').textContent='0';
document.getElementById('sentenceCount').textContent='0';
document.getElementById('paragraphCount').textContent='0';
document.getElementById('charCount').textContent='0';
document.getElementById('readabilityScore').textContent='0';
}
function generateSummary(){
const text=document.getElementById('inputText').value;
const length=parseInt(document.getElementById('summaryLength').value);
if(!text.trim()){alert('Please enter some text or upload a file first.');return;}
fetch('/summarize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:text,length:length})})
.then(response=>response.json())
.then(data=>{
document.getElementById('summaryOutput').textContent=data.summary;
updateStats();
})
.catch(error=>{console.error('Error:',error);alert('Error generating summary. Please try again.');});
}
function clearText(){
document.getElementById('inputText').value='';
document.getElementById('summaryOutput').textContent='Your summary will appear here...';
resetStats();
}
document.getElementById('fileInput').addEventListener('change',function(event){
const file=event.target.files[0];
if(file){
if(file.size>1024*1024){alert('File size too large. Please select a file smaller than 1MB.');return;}
const reader=new FileReader();
reader.onload=function(e){document.getElementById('inputText').value=e.target.result;updateStats();}
reader.readAsText(file);
}});
document.getElementById('inputText').addEventListener('input',updateStats);
</script>
</body>
</html>
'''

@app.route('/summarize', methods=['POST'])
def summarize_text():
    data = request.json
    text = data.get('text','')
    length = data.get('length',3)
    summary = summarizer.summarize(text, length)
    return jsonify({'summary':summary})

@app.route('/analyze', methods=['POST'])
def analyze_text():
    data = request.json
    text = data.get('text','')
    stats = analyzer.get_text_stats(text)
    readability = analyzer.calculate_readability(text)
    return jsonify({
        'words': stats['words'],
        'sentences': stats['sentences'],
        'paragraphs': stats['paragraphs'],
        'characters': stats['characters'],
        'readability': readability
    })

if __name__=='__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
