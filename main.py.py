import os
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

TOKEN_LIMIT = 8000  #Token limit for GEMINI 1.5 Pro

def get_csv_text(csv_path):
    data = pd.read_csv(csv_path)
    text = ""
    for _, row in data.iterrows():
        disease_info = f"Disease: {row['Disease']}\nSymptoms: {row['Symptoms']}\nPossible Treatments: {row['Possible Treatment']}\n"
        text += disease_info + "\n"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    
    Instructions: Answer the following question based solely on the information provided in the {context}, which includes a list of diseases, their symptoms, and possible treatments. 
    Only refer to the symptoms and treatments in this {context}; if the requested information is not available, respond with “Information Not Available.”
    
    Response Format if only Symptoms are to be found:
    Disease: Identify the disease based on symptoms provided in the question.
    Symptoms: List the relevant symptoms found in the {context} that match the question.
    Response Limitation: Use only the information in the {context}; if the exact answer is not present, do not speculate or add external information.
    IMP:"Also Warn the user to consider taking Clinical Expertise from Practitioner and does not rely on the treaments
    given by the AI Bot."
    
    Response Format if only Treatment or Treatments are to be found:
    Disease: Identify the disease based on symptoms provided in the question.
    Treatment: Provide the treatment options specified in the {context} for the disease or symptoms mentioned.
    Response Limitation: Use only the information in the {context}; if the exact answer is not present, do not speculate or add external information.
    IMP:"Also Warn the user to consider taking Clinical Expertise from Practitioner and does not rely on the treaments
    given by the AI Bot."

    Response Format if both symptoms and treatments are to be found:
    Disease: Identify the disease based on symptoms provided in the question.
    Symptoms: List the relevant symptoms found in the {context} that match the question.
    Treatment: Provide the treatment options specified in the {context} for the disease or symptoms mentioned.
    Response Limitation: Use only the information in the {context}; if the exact answer is not present, do not speculate or add external information.
    IMP:"Also Warn the user to consider taking Clinical Expertise from Practitioner and does not rely on the treaments
    given by the AI Bot. Use same disclaimer for all answers"

    Response Format if symptoms or treatments of certain diseases are Colliding or matching then:

    Disease: Identify the disease based on symptoms provided in the question and List 'ALL POSSIBLE DISEASES (sorted by BEST SIMILIARTIY FIRST) in the format specified below'.
    Response Limitation: Use only the information in the {context}; if the exact answer is not present, do not speculate or add external information.
    If symptoms or treatments are to be listed with Disease then give them as in the following format:
    1->"Disease" : Asthama || "Symptoms": Wheezing, shortness of breath, chest tightness || "Possible Treatments": Inhalers, bronchodilators, corticosteroids
    2->"Disease" : Pneumonia || "Symptoms": Fever, cough, difficulty breathing, chest pain || "Possible Treatments": Inhalers, Antibiotics (bacterial), supportive care
    Above two are just for reference for format do not consider them in answers.
    IMP:"Also Warn the user to consider taking Clinical Expertise from Practitioner and does not rely on the treaments
    given by the AI Bot. Use same disclaimer for all answers"

    Response Format if Data is not Found:
    Response as NO DATA FOUND
    IMP:"Also Warn the user to consider taking Clinical Expertise from Practitioner and does not rely on the treaments
    given by the AI Bot. Use same disclaimer for all answers"
    
    Context:\n {context}\n
    Question:\n{question}\n
    Answer:
"""
    model = ChatGoogleGenerativeAI(model='gemini-1.5-pro', temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    chain = load_qa_chain(model, chain_type='stuff', prompt=prompt)
    return chain

def process_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model='models/embedding-001')
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()

    response = chain(
        {
            "input_documents": docs,
            "question": user_question
        },
        return_only_outputs=True
    )
    return response['output_text']

def save_to_csv(data, filename="output.csv"):
    df = pd.DataFrame(data, columns=["Question", "Answer"])
    df.to_csv(filename, index=False)
    print(f"Results saved to {filename}")

def main(csv_path, questions):
    # Load and process the CSV
    csv_text = get_csv_text(csv_path)
    text_chunks = get_text_chunks(csv_text)
    get_vector_store(text_chunks)

    results = []
    for question in questions:
        # Ensure token limit is respected
        if len(question.split()) > TOKEN_LIMIT:
            print(f"Question exceeds token limit of {TOKEN_LIMIT} tokens.")
            continue
        answer = process_question(question)
        results.append({"Question": question, "Answer": answer})

    save_to_csv(results)

# Example Usage
csv_path = "disease.csv"  # Specify your CSV file path
questions = [
    "What are the symptoms of Diabetes?",
    "How can Hypertension be treated?",
    "A patient has symptoms which includes Wheezing and shortness of breath, what is/are possible disease for such symptoms. Also give possible treatments."
    "A patient has Fever. What disease it may have?"

]
main(csv_path, questions)
