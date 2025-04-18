import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Function to get Gemini API key from environment or user input
def get_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.session_state.get("api_key", "")
    return api_key

# Function to set up embeddings
def get_embedding(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    return embeddings

# Process document function
def process_document(uploaded_file, api_key):
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name
    
    # Load and process the PDF
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(pages)
    
    # Create embedding function
    embedding_function = get_embedding(api_key)
    
    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_function
    )
    
    # Clean up temp file
    os.unlink(tmp_path)
    
    return vectorstore

# Format documents for the prompt
def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

# Main app function
def main():
    st.title("Document Q&A with Gemini")
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", 
                               value=st.session_state.get("api_key", ""), 
                               type="password")
        if api_key:
            st.session_state.api_key = api_key
    
    # Check if API key is available
    if not get_api_key():
        st.warning("Please enter your Gemini API key in the sidebar.")
        st.stop()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            # Process the document
            vectorstore = process_document(uploaded_file, get_api_key())
            st.session_state.vectorstore = vectorstore
            st.success(f"Document '{uploaded_file.name}' processed successfully!")
    
    # Only show question input if document is processed
    if 'vectorstore' in st.session_state:
        st.subheader("Ask a question about your document")
        question = st.text_input("Your question")
        
        if question:
            with st.spinner("Finding answer..."):
                # Set up the RAG pipeline
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity")
                
                # Define the prompt template
                PROMPT_TEMPLATE = """
                You are a research assistant for question-answering tasks.
                Use the following pieces of retrieved context to answer
                the question. If you don't know the answer, say that you don't know.
                DON'T MAKE UP ANYTHING.
                
                {context}
                ---
                Answer the question based on the above context: {question}
                """
                
                prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
                
                # Set up LLM
                llm = GoogleGenerativeAI(
                    model="gemini-2.0-flash", 
                    google_api_key=get_api_key()
                )
                
                # Create the RAG chain
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                )
                
                # Get response
                response = rag_chain.invoke(question)
                
                st.subheader("Answer")
                st.write(response)
                
                # Show retrieved chunks (optional, for debugging)
                with st.expander("View retrieved document chunks"):
                    retrieved_docs = retriever.invoke(question)
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(doc.page_content)
                        st.markdown("---")

if __name__ == "__main__":
    main()