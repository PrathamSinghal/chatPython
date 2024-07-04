from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import requests
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

os.environ["GOOGLE_API_KEY"] = 'AIzaSyBahSHeO97ZBXD0zibPODrDerSDTG7p8Qs'

app = Flask(__name__)


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(pdf_id, text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local(f"faiss_index_{pdf_id}")


def get_conversational_chain():
    prompt_template = """Please provide a detailed and accurate answer to the question based on the provided context 
    from the uploaded PDF document. Use clear and descriptive language to ensure the user understands the response.\n\n 
    Context:\n{context}\n 
    Question:\n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro",
                                   temperature=0.7, google_api_key=os.getenv("GOOGLE_API_KEY"))

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


def user_input(pdf_id, user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    new_db = FAISS.load_local(f"faiss_index_{pdf_id}", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}
        , return_only_outputs=True)

    return response["output_text"]


@app.route('/process_pdf', methods=['POST'])
def process_pdf():
    # data = request.json
    pdf_id = request.json.get('pdfId')
    pdf_file_url = request.json.get('url')

    # Download the PDF file
    response = requests.get(pdf_file_url)
    pdf_file_path = f"{pdf_id}.pdf"
    with open(pdf_file_path, 'wb') as pdf:
        pdf.write(response.content)

    pdf_docs = [pdf_file_path]
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    get_vector_store(pdf_id, text_chunks)

    return jsonify({"message": "PDF files processed successfully!", "pdfId": pdf_id})


@app.route('/ask_question', methods=['POST'])
def ask_question():
    # data = request.json
    pdf_id = request.json.get('pdfId')
    user_question = request.json.get('question')
    response = user_input(pdf_id, user_question)
    return jsonify({"reply": response})


if __name__ == "__main__":
    app.run(debug=True, port=4005)
