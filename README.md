# CISNR RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about CISNR using Groq's Llama 3.3 model and ChromaDB vector store.

## Features

- 📄 Loads data from Word documents
- 🤖 Uses Groq's Llama 3.3 70B model
- 🔍 Vector search with ChromaDB
- 💬 Interactive command-line interface

## Prerequisites

- Python 3.8+
- Groq API Key ([Get one here](https://console.groq.com/))

## Installation

1. **Clone the repository**
```bash
   git clone https://github.com/YOUR_USERNAME/cisnr-rag-chatbot.git
   cd cisnr-rag-chatbot
```

2. **Create a virtual environment**
```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
   pip install -r requirements.txt
```

4. **Set up environment variables**
   - Copy `.env.example` to `.env`
   - Add your Groq API key to `.env`:
```bash
   GROQ_API_KEY=your_actual_api_key_here
```

5. **Add your data**
   - Place your `cisnr_data.docx` file in the project root directory

## Usage

Run the chatbot:
```bash
python rag_cisnr.py
```

Example interaction:
```
You: What is CISNR?
Bot: CISNR (Centre of Intelligent System & Network Research) provides smart and intelligent solutions...

You: Tell me about AquaCure
Bot: AquaCure is a water supply management system...
```

Type `exit` to quit.

## Project Structure
```
cisnr-rag-chatbot/
├── rag_cisnr.py           # Main chatbot script
├── cisnr_data.docx        # Data source (Word document)
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variables template
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## Technologies Used

- **LangChain** - RAG framework
- **Groq** - LLM API (Llama 3.3 70B)
- **ChromaDB** - Vector database
- **HuggingFace** - Sentence embeddings
- **python-docx** - Word document processing

## License

MIT License
