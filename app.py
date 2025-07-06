from key import api_key

from flask import Flask, render_template, request, jsonify
import pyaudio
import wave
import speech_recognition as sr
import os
from datetime import datetime
import threading
import queue
import numpy as np
import openai
import tiktoken
from pathlib import Path
import json
from tqdm import tqdm
from docx import Document
from gtts import gTTS
import io
from typing import *
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import faiss
import torch
from sentence_transformers import SentenceTransformer

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Global variables
recording = False
audio_queue = queue.Queue()

app = Flask(__name__)

class OptimizedRAG:
    def __init__(self, api_key: str):
        openai.api_key = api_key
        self.documents = []
        self.sources = []
        self.enc = tiktoken.encoding_for_model("gpt-4")
        # Initialize FAISS index for CPU
        self.dimension = 768  # SentenceTransformer embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        # Load local embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.model.eval()
        # Use CUDA if available for the transformer model
        if torch.cuda.is_available():
            self.model.to('cuda')
        
        self.max_tokens = 1000
        self.chunk_overlap = 50  # Token overlap between chunks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.cache = {}

    @lru_cache(maxsize=1000)
    def count_tokens(self, text: str) -> int:
        """Count tokens with caching."""
        return len(self.enc.encode(text))

    def save_index(self, filename: str = 'rag_index.faiss'):
        """Save FAISS index and related data."""
        faiss.write_index(self.index, filename)
        cache_data = {
            'documents': self.documents,
            'sources': self.sources
        }
        with open(filename + '.json', 'w') as f:
            json.dump(cache_data, f)

    def load_index(self, filename: str = 'rag_index.faiss'):
        """Load FAISS index and related data."""
        if os.path.exists(filename) and os.path.exists(filename + '.json'):
            self.index = faiss.read_index(filename)
            with open(filename + '.json', 'r') as f:
                cache_data = json.load(f)
                self.documents = cache_data['documents']
                self.sources = cache_data['sources']
            return True
        return False

    def get_embeddings_batch(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for a batch of texts using SentenceTransformer."""
        with torch.no_grad():
            embeddings = self.model.encode(
                texts, 
                batch_size=32, 
                show_progress_bar=False,
                convert_to_numpy=True,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
        return embeddings

    def process_document(self, text: str, source: str, chunk_size: int = 500) -> None:
        """Process document with improved chunking and batched embedding."""
        chunks = self._chunk_text(text, chunk_size)
        
        # Process in smaller batches to manage memory
        batch_size = 32
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = self.get_embeddings_batch(batch_chunks)
            self.index.add(embeddings)
            
            # Store document chunks and sources
            self.documents.extend(batch_chunks)
            self.sources.extend([source] * len(batch_chunks))

    def _chunk_text(self, text: str, chunk_size: int) -> List[str]:
        """Improved text chunking with overlap and semantic boundaries."""
        chunks = []
        sentences = text.replace('\n', ' ').split('. ')
        current_chunk = []
        current_size = 0

        for i, sentence in enumerate(sentences):
            sentence_tokens = self.count_tokens(sentence)
            
            if current_size + sentence_tokens > chunk_size:
                if current_chunk:
                    chunk_text = '. '.join(current_chunk) + '.'
                    chunks.append(chunk_text)
                    
                    # Keep last sentence for overlap
                    overlap_sentences = current_chunk[-self.chunk_overlap:]
                    current_chunk = overlap_sentences + [sentence]
                    current_size = sum(self.count_tokens(s) 
                                     for s in current_chunk)
                else:
                    chunks.append(sentence + '.')
                    current_chunk = []
                    current_size = 0
            else:
                current_chunk.append(sentence)
                current_size += sentence_tokens

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    def get_relevant_context(self, query: str, max_chunks: int = 5) -> tuple[str, List[str]]:
        """Get relevant context using FAISS for faster similarity search."""
        if not self.documents:
            return "", []

        # Get query embedding
        query_embedding = self.get_embeddings_batch([query])

        # Search FAISS index
        D, I = self.index.search(query_embedding, max_chunks)
        
        selected_chunks = []
        selected_sources = []
        total_tokens = 0

        for idx in I[0]:
            chunk = self.documents[idx]
            source = self.sources[idx]
            chunk_tokens = self.count_tokens(chunk)
            
            if total_tokens + chunk_tokens <= self.max_tokens:
                selected_chunks.append(chunk)
                selected_sources.append(source)
                total_tokens += chunk_tokens
            else:
                break

        context = "\n\n".join(selected_chunks)
        return context, selected_sources

    @lru_cache(maxsize=100)
    def ask(self, question: str) -> str:
        """Cached question answering with improved prompt."""
        context, sources = self.get_relevant_context(question)
        
        if not context:
            return self._get_fallback_response(question)

        prompt = self._create_enhanced_prompt(question, context, sources)

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are an AI teacher specializing in clear, accurate explanations of deep learning concepts. Always cite relevant sources when using provided context."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more focused responses
                max_tokens=500
            )
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error getting response: {str(e)}"

    def _create_enhanced_prompt(self, question: str, context: str, sources: List[str]) -> str:
        """Create an enhanced prompt for better response quality."""
        return f"""You are an AI teacher specializing in clear, accurate explanations of deep learning concepts. Always cite relevant sources when using provided context.

Question: {question}

Relevant Context:
{context}

Sources: {', '.join(sources)}

Instructions:
1. Answer the question directly and concisely
2. Use specific information from the context
3. Cite sources when using specific information
4. If the context doesn't fully address the question, acknowledge this
5. Focus on accuracy and clarity

Answer:"""

    def _get_fallback_response(self, question: str) -> str:
        """Enhanced fallback response when no context is available."""
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a friendly, patient, and knowledgeable teacher who explains concepts to young students in a simple, engaging, and interactive way. Your primary goal is to make learning enjoyable and easy for students to understand. When a student asks a question, respond with detailed and age-appropriate explanations. Use examples, stories, or analogies to make the topic relatable and interesting. Encourage students to ask follow-up questions or share their thoughts. Keep your tone warm and approachable, ensuring they feel comfortable and motivated to learn. Break down complex topics into small, manageable steps and use clear, concise language suitable for young learners. Always provide positive reinforcement to boost their confidence, and focus on building curiosity and a love for learning"},
                {"role": "user", "content": question}
            ],
            temperature=0.3,
            max_tokens=500
        )
        return response['choices'][0]['message']['content']

def record_audio():
    """Record audio from microphone."""
    global recording
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                   channels=CHANNELS,
                   rate=RATE,
                   input=True,
                   frames_per_buffer=CHUNK)
    
    frames = []
    
    while recording:
        data = stream.read(CHUNK)
        frames.append(data)
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    audio_queue.put(frames)

def transcribe_audio(audio_file):
    """Transcribe audio file to text."""
    recognizer = sr.Recognizer()
    
    with sr.AudioFile(audio_file) as source:
        audio = recognizer.record(source)
        
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return "Could not understand audio"
    except sr.RequestError:
        return "Could not request results from speech recognition service"

# Initialize RAG system
def init_rag(api_key: str):
    """Initialize the RAG system with the given API key."""
    rag_system = OptimizedRAG(api_key)
    
    # Try to load existing index
    if not rag_system.load_index():
        directory = r"C:\Users\vedth\Desktop\classerly\classerly-1\Classerly-milestone-1\Science\Strand E"
        process_directory(directory, rag_system)
        rag_system.save_index()
    
    return rag_system

def process_directory(directory_path: str, rag: OptimizedRAG) -> None:
    """Process all documents in the given directory."""
    path = Path(directory_path)
    files = list(path.rglob("*.txt")) + list(path.rglob("*.docx"))
    
    for file_path in tqdm(files, desc="Processing files"):
        try:
            if file_path.suffix == '.txt':
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            elif file_path.suffix == '.docx':
                doc = Document(file_path)
                content = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
            
            rag.process_document(content, str(file_path))
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Flask routes
@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/start_recording', methods=['POST'])
def start_recording():
    """Start audio recording."""
    global recording
    recording = True
    
    threading.Thread(target=record_audio).start()
    return jsonify({"status": "started"})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """Stop audio recording and transcribe."""
    global recording
    recording = False
    
    frames = audio_queue.get()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recording_{timestamp}.wav"
    
    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    text = transcribe_audio(filename)
    os.remove(filename)
    
    return jsonify({"text": text})

@app.route('/ask', methods=['POST'])
def ask():
    """Handle questions and return answers."""
    data = request.get_json()
    question = data.get('question')
    response_format = data.get('format', 'text')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        answer = rag_system.ask(question)
        
        if response_format == 'audio':
            tts = gTTS(text=answer, lang='en')
            audio_io = io.BytesIO()
            tts.write_to_fp(audio_io)
            audio_io.seek(0)
            
            return jsonify({
                'answer': answer,
                'audio': audio_io.getvalue().decode('latin1')
            })
        else:
            return jsonify({'answer': answer})
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize the RAG system with your OpenAI API key
    
    rag_system = init_rag(api_key)
    app.run(debug=True)
    
