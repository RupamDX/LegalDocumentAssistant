Legal Document Analysis and Q&A Application

Welcome to the Legal Document Analysis and Q&A Application repository! This project leverages advanced AI technologies to analyze legal documents, extract crucial information, and provide intelligent Q&A capabilities. Built as an MVP in just 10 hours, this project showcases the potential of combining open-source tools, GPT-3.5, and vector databases to streamline legal workflows.

Features

Information Extraction: Identify key entities like parties, dates, and important clauses (e.g., confidentiality, indemnity) from uploaded legal documents.

Q&A Functionality: Ask specific questions about the document and receive accurate, context-aware answers.

High Performance: Powered by advanced embeddings and vector databases for quick and efficient results.

Technologies Used

OpenAI GPT-3.5: For generating intelligent and context-aware responses.

LangChain: For creating a conversational retrieval chain.

FAISS: A fast vector database for embedding storage and retrieval.

SentenceTransformer: Open-source model for generating embeddings.

SpaCy: For natural language processing and entity recognition.

Streamlit: For building an interactive web-based interface.

Installation

Follow these steps to set up the application:

Clone the Repository:

git clone https://github.com/your-username/legal-docs-analytics.git
cd legal-docs-analytics

Set Up a Virtual Environment:

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install Dependencies:

pip install -r requirements.txt

Download SpaCy Model:

python -m spacy download en_core_web_sm

Run the Application:

streamlit run app.py

Access the App:
Open your browser and navigate to http://localhost:8501.

Usage

Upload a PDF File:

Click the "Upload a PDF file" button and select a legal document.

View Extracted Information:

Automatically extracted entities (parties, dates, clauses) will be displayed.

Ask Questions:

Enter a question about the document in the input field and get context-specific answers.

Demo

Check out the https://www.linkedin.com/posts/rupam-patra_legaltech-ai-innovation-activity-7276355674190626817-gZT9?utm_source=share&utm_medium=member_desktop video to see the application in action!

Future Enhancements

Enhanced Entity Recognition: Support for more entity types and domain-specific clauses.

Document Summarization: Generate concise summaries of legal documents.

Multi-Language Support: Extend capabilities to analyze documents in multiple languages.

Improved UI/UX: Make the interface more intuitive and user-friendly.

Acknowledgments

OpenAI for the GPT-3.5 API.

SentenceTransformer and FAISS for embedding-based retrieval.

LangChain for building robust conversational AI chains.

Streamlit for enabling quick MVP development.

Contributing

Contributions are welcome! Please fork this repository and submit a pull request with your improvements.


Contact

If you have any questions, feel free to reach out:

Email: patra.ru@northeastern.edu

LinkedIn:https://www.linkedin.com/in/rupam-patra

