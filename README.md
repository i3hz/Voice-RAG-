# 🎙️  Voice RAG

**Voice RAG** is a Flask-based voice-enabled chatbot system powered by Retrieval-Augmented Generation (RAG) and OpenAI's GPT-4. It allows users to ask spoken questions, converts them to text, retrieves relevant document chunks from a local knowledge base, and generates accurate, context-aware answers. Optionally, it also responds with voice.

---

## 🔧 Features

- 🎤 **Voice Interaction**: Record questions via your microphone
- 📄 **Local Document Processing**: Ingest `.txt` and `.docx` files using SentenceTransformer + FAISS
- ⚡ **Fast Semantic Search**: Uses FAISS for efficient retrieval of relevant document chunks
- 🤖 **Context-Aware Answers**: GPT-4 generates answers grounded in the retrieved context
- 🔁 **Fallback AI Mode**: If no relevant context is found, it still provides a general answer
- 🔊 **Text-to-Speech**: Converts AI-generated text responses into spoken answers

---

## 🛠️ Tech Stack

| Component               | Technology                    |
|------------------------|-------------------------------|
| Web Framework          | Flask                         |
| Vector Store           | FAISS                         |
| Embedding Model        | SentenceTransformer (`MiniLM`)|
| LLM Backend            | OpenAI GPT-4                  |
| Speech Recognition     | `speech_recognition` (Google) |
| Audio I/O              | PyAudio, Wave                 |
| Text-to-Speech         | gTTS                          |

---
## 🧠 How It Works

1. **Startup**:
    - Loads existing FAISS index if available.
    - If not, scans and processes documents into vector embeddings.
2. **Voice Interaction**:
    - User records a question.
    - Speech is transcribed using Google's Speech API.
3. **Context Retrieval**:
    - SentenceTransformer encodes the query.
    - FAISS retrieves the most relevant chunks.
4. **Answer Generation**:
    - GPT-4 uses the query + retrieved context to answer.
    - Optional TTS returns the spoken answer.

---

## ▶️ Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt
```
Make sure to also install PyAudio correctly depending on your OS.
2. Add OpenAI Key

Create a file key.py:

api_key = "your-openai-api-key"

3. Run the App

python app.py

Then open your browser at http://127.0.0.1:5000
🗣️ Example Flow

    Press "Record"

    Ask: "What are Newton’s three laws of motion?"

    System:

        Transcribes speech

        Retrieves matching notes

        Queries GPT-4 with the context

        Speaks the answer back to you
🚀 Future Improvements

    Live waveform display while recording

    Multi-user session history

    Audio input/output support on mobile
