import streamlit as st
#from PyPDF2 import PdfReader 
from pypdf import PdfReader                                         #pip install pypdf
import os
import pickle
import google.generativeai as genai                                 #pip install google-generativeai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings     #pip install -U langchain-google-genai
from langchain_community.vectorstores import FAISS                  #pip install -U langchain-community #pip install faiss-cpu
from dotenv import load_dotenv                                      #pip install python-dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Set page configuration to wide layout
st.set_page_config(layout="wide", page_title="PDF Reader", page_icon=":books:")

# Create a local directory for the PDF files
pdf_dir = 'knowledge'
os.makedirs(pdf_dir, exist_ok=True)

# Get a list of PDF files from the local directory
pdf_files = [file for file in os.listdir(pdf_dir) if file.lower().endswith(".pdf")]

# Function to read PDF content same name: get_pdf_text
def read_pdf(file_path):
    pdf_reader = PdfReader(file_path)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to chunk text from PDF file
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

#Functionn to get vector store from chunk text
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context,
    elaborate your answer and provide example if possible,
    make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7)
    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)
    print(response)
    st.write("Reply: ")
    st.write(response["output_text"])

def four_questions(possible_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(possible_question)
    chain = get_conversational_chain()
    response = chain(
        {"input_documents":docs, "question": possible_question}
        , return_only_outputs=True)
    print(response)
    #st.write("<div style='text-align: right'>Possible Questions: </div>", unsafe_allow_html=True)
    st.write("Possible Questions: ")
    #output_text = response["output_text"]
    #st.write(f'<div style="text-align: right">{output_text}</div>', unsafe_allow_html=True)
    st.write(response["output_text"])



def main():
    with st.sidebar:
        upload_folder = "knowledge"
        uploaded_file = st.file_uploader("Upload PDF file", type="pdf")
        if uploaded_file is not None:
            # Display file details
            file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
            #st.write("File Details:", file_details)

            # Save the uploaded PDF file to the specified folder
            save_path = os.path.join(upload_folder, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"PDF file saved successfully at {save_path}")

        # Refresh button
        if st.button("Refresh"):
            # Clear the cache and reload the app
            #st.caching.clear_cache()
            st.experimental_rerun()

        # Display a list of PDF files as radio buttons in the sidebar
        selected_file = st.radio("Select a PDF file:", pdf_files, index=0)
        st.success(selected_file)

        # Read text from PDF Stage
        if selected_file:
            file_path = os.path.join(pdf_dir, selected_file)
            try:
                with st.spinner("Reading PDF..."):
                    #text = read_pdf(file_path)
                    raw_text = read_pdf(file_path)
                    with st.spinner("Split text and generate vector..."):
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                st.success("Done")
                st.info("The PDF content is hidden; input your query to proceed.")
            except FileNotFoundError:
                st.error(f"File not found: {file_path}")
                return
            except Exception as e:
                st.error(f"Error occurred while reading the PDF: {e}")
                return
        
        
    #st.title("Query your PDF")
    #st.header("Strategic Decision Making - Beta")
    #st.write(selected_file)
    possible_question = "List 4 questions I can ask from this document, must be short and concise."
    four_questions(possible_question)
    st.markdown("<hr>", unsafe_allow_html=True)
    # User input area
    #st.write("Copy & Paste the question, or write your own question related to the selected document:")     
    user_question = st.text_area("Copy & Paste the question, or write your own question related to the selected document:")
    

    # Disclaimer
    #st.write("Disclaimer: While Cognimus AI aims for accuracy, it may occasionally provide incomplete or inaccurate information, including details about individuals. Therefore, it's recommended to verify its responses, especially for sensitive matters. Additionally, Cognimus AI cannot ensure a 100% success rate in strategy formulation, nor can it be held responsible for execution outcomes")
 
    if st.button("Find related answer"):
            with st.spinner("Processing..."):
                 # Add a horizontal line
                st.markdown("<hr>", unsafe_allow_html=True)
                if user_question:
                    
                    user_input(user_question)
                    st.success("Done")
    #st.write(text_chunks)
if __name__ == "__main__":
    main()