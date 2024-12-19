import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
import spacy
import os

# Load Models
nlp = spacy.load('en_core_web_sm')
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Functions
def parse_and_extract(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    text = " ".join([doc.page_content for doc in documents])
    doc = nlp(text)
    extracted_info = {
        "parties": [ent.text for ent in doc.ents if ent.label_ == "ORG"],
        "dates": [ent.text for ent in doc.ents if ent.label_ == "DATE"],
        "clauses": [
            sentence for sentence in text.split(".")
            if "indemnity" in sentence.lower() or "confidentiality" in sentence.lower()
        ]
    }
    return extracted_info, documents

def process_and_store_embeddings(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)
    texts = [chunk.page_content for chunk in chunks]
    vector_store = FAISS.from_texts(texts, embedding_model)
    return vector_store

def create_qa_chain(vector_store):
    llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.0)
    retriever = vector_store.as_retriever()
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever)
    return qa_chain

def answer_question(qa_chain, question, chat_history):
    result = qa_chain({"question": question, "chat_history": chat_history})
    return result["answer"]

# Streamlit App

# API Key handling
api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key


st.title("PDF Document Analysis and Q&A")
st.write("Upload a PDF and ask questions based on its content.")

# File Upload
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save uploaded file
    file_path = f"temp_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Parse and Extract
    with st.spinner("Extracting information..."):
        extracted_info, documents = parse_and_extract(file_path)
        vector_store = process_and_store_embeddings(documents)
        qa_chain = create_qa_chain(vector_store)
    
    # Display Extracted Information
    st.subheader("Extracted Information")
    st.write("**Parties:**", extracted_info["parties"])
    st.write("**Dates:**", extracted_info["dates"])
    st.write("**Clauses:**", extracted_info["clauses"])

    # Query Section
    st.subheader("Ask Questions")
    chat_history = []
    query = st.text_input("Enter your query:")
    if query:
        with st.spinner("Fetching answer..."):
            answer = answer_question(qa_chain, query, chat_history)
            st.write("**Answer:**", answer)
            chat_history.append((query, answer))
