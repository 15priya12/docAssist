# Document Q&A with Gemini LLM

A Streamlit application that lets you upload PDF documents and ask questions about their content using Google's Gemini LLM with Retrieval Augmented Generation (RAG).

## Features

- Upload PDF documents
- Ask natural language questions about document content
- View document chunks used to generate answers
- Simple, intuitive interface

## Quick Start

1. Clone the repository and install dependencies:
```bash
git clone https://github.com/your-username/document-qa-app.git
cd document-qa-app
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

3. Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/)

4. Enter your API key in the app sidebar, upload a PDF, and start asking questions!

## How It Works

The app processes your document into chunks, creates embeddings using Google's model, and stores them in a vector database. When you ask a question, it retrieves the most relevant chunks and passes them to Gemini to generate an answer based only on your document's content.

## Deployment

Deploy to Streamlit Cloud by connecting your GitHub repository and setting your Gemini API key as a secret.

## Requirements

- Python 3.8+
- Google Gemini API key
- See requirements.txt for dependencies
