import streamlit as st
import os
import tempfile
import pandas as pd

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

def get_api_key():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.session_state.get("api_key", "")
    return api_key

def get_embedding(api_key):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001", 
        google_api_key=api_key
    )
    return embeddings

def process_document(uploaded_file, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    loader = PyPDFLoader(tmp_path)
    pages = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", " "]
    )
    chunks = text_splitter.split_documents(pages)
    
    embedding_function = get_embedding(api_key)
  
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function
    )
    
    os.unlink(tmp_path)
    
    return vectorstore

def format_docs(docs):
    return "\n\n---\n\n".join([doc.page_content for doc in docs])

def main():
    st.title("Document Q&A with Gemini")
    
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter Gemini API Key", 
                               value=st.session_state.get("api_key", ""), 
                               type="password")
        if api_key:
            st.session_state.api_key = api_key

    if not get_api_key():
        st.warning("Please enter your Gemini API key in the sidebar.")
        st.stop()
    
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])
    
    if uploaded_file:
        with st.spinner("Processing document..."):
            vectorstore = process_document(uploaded_file, get_api_key())
            st.session_state.vectorstore = vectorstore
            st.success(f"Document '{uploaded_file.name}' processed successfully!")

    if 'vectorstore' in st.session_state:
        st.subheader("Ask a question about your document")
        question = st.text_input("Your question")
        
        if question:
            with st.spinner("Finding answer..."):
                retriever = st.session_state.vectorstore.as_retriever(search_type="similarity")
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
                
                llm = GoogleGenerativeAI(
                    model="gemini-2.0-flash", 
                    google_api_key=get_api_key()
                )
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                )

                response = rag_chain.invoke(question)
                
                st.subheader("Answer")
                st.write(response)

                with st.expander("View retrieved document chunks"):
                    retrieved_docs = retriever.invoke(question)
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}**")
                        st.text(doc.page_content)
                        st.markdown("---")

if __name__ == "__main__":
    main()
