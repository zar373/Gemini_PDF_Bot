import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

import os


load_dotenv()

google.generativeai.configure(api_key= os.getenv("GOOGLE_API_KEY"))

def get_pdf_text(pdf_docs):
    """
    We read the pdf and go through each and every pages and extract the text.
    """
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= 10000, chunk_overlap=1000
    
    )
    chunks_text= text_splitter.split_text(text)
    return chunks_text

def get_vector_store(text_chunks):
    """
    Creates a Folder, where we can see our vectores are stored in an unreadable format. Whenever a question is asked, we get the info from there.
    """
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store= FAISS.from_texts(text_chunks, embedding= embeddings)
    vectore_store=FAISS.from_texts(text_chunks, embedding= embeddings)
    vector_store.save_local("faiss_index")

def get_convserational_chain():
    """
    Create the chain in a defined input format. This input wil create the chain and then we will create the chain.
    """
    prompt_template="""
    Answer the question as detailed as possible and in simple easy understanding language using the provided context from the document. if the question is not in the context, say "I couldn't find the answer in the document provided".
    Context:\n {context} \n
    Question: \n {question} \n
    Answer: 
    """

    model= ChatGoogleGenerativeAI(model= "gemini-pro", temperature=0.3)

    prompt= PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain= load_qa_chain(model, chain_type= "stuff", prompt=prompt )
    return chain 

def user_input(user_question):
    """
    When the user inputs the question, we get the answer. After loading embeddings, we use FAISS, already the pdf is converted in to the vectors is sotred in the FAISS index. Therefore we are loading the FAISS index. Then similarity search is done by the get_conversational_chain, and then we will get the response.
    """
    embeddings= GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db= FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs= new_db.similarity_search(user_question)
    
    chain= get_convserational_chain()

    response= chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs= True
    )

    print(response)
    st.write('Reply: ', response["output_text"])


def main():
    """
    Upload the PDF and then 3 functions are executed in Sidebar Code to create the FAISS Index
    """
    st.set_page_config(page_title="Gemini PDF Bot", layout="wide")

    # Styling the app
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #CEC2EB;
        }
        .stButton button {
            background-color: black !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
        }
        [data-testid="stFileUpload"] label {
            background-color: black !important;
            color: white !important;
            border: none !important;
            font-weight: bold !important;
        }
        [data-testid="stSidebar"] {
            background-color: #EBE8FC;
        }
        .menu-heading {
            font-weight: bold;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.header("🤖 Chat with PDF using Gemini")
    st.write("Step 1: Upload your PDF.")
    st.write("Step 2: Click the 'Submit & Execute' button.")
    st.write("Step 3: Chat with PDF!")
    user_question= st.text_input("Ask a question about your PDF")


    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.markdown('<h2 class="menu-heading">Menu</h2>', unsafe_allow_html=True)
        pdf_docs = st.file_uploader("Upload your PDF and click on Execute", accept_multiple_files=True)
        if st.button("Submit & Execute"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file before proceeding!")
            else:
                with st.spinner("Executing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done! You PDF is ready to interact with.")

if __name__ == "__main__":
    main()



