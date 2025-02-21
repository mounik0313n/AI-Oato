import os
import whisper  # OpenAI Whisper library for transcription
import google.generativeai as genai
import nltk
import time
import warnings
from flask import Flask, request, render_template, send_file
from werkzeug.utils import secure_filename

GOOGLE_GEMINI_API_KEY = "AIzaSyBmL_zLi-7T6Ait-lpxJudUmNAZjkvk7TA"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Suppress the warning from pydub about missing ffmpeg
warnings.filterwarnings("ignore", message="Couldn't find ffmpeg or avconv - defaulting to ffmpeg")

# Function to process .wav files directly
def transcribe_audio(wav_file, model_size="base"):
    # Use OpenAI Whisper model to transcribe the audio file
    model = whisper.load_model(model_size)  # Correctly using whisper's load_model method
    result = model.transcribe(wav_file, fp16=False)
    return result['text']

def summarize_with_gemini(text):
    prompt = f"Summarize the following conversation in a structured and meaningful way:\n\n{text}\n\nSummary:"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "No summary available."

def extract_key_topics(text):
    prompt = f"Extract the key topics from this conversation:\n\n{text}\n\nKey Topics:"
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text.strip() if response.text else "No key topics extracted."

def save_to_file(text, filename):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(text)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded."
        
        file = request.files['file']
        if file.filename == '':
            return "No selected file."
        
        start_time = time.time()
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Directly process the .wav file without conversion
        wav_file = file_path
        transcript = transcribe_audio(wav_file)
        summary = summarize_with_gemini(transcript)
        key_topics = extract_key_topics(transcript)

        summary_file = os.path.join(app.config['PROCESSED_FOLDER'], "summary.txt")
        key_topics_file = os.path.join(app.config['PROCESSED_FOLDER'], "key_topics.txt")

        save_to_file(summary, summary_file)
        save_to_file(key_topics, key_topics_file)

        processing_time = round(time.time() - start_time, 2)
        return render_template('result.html', summary=summary, key_topics=key_topics, processing_time=processing_time)
    
    return render_template('index.html')

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename), as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
