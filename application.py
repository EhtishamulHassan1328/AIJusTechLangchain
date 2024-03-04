from flask import Flask, jsonify, request
from flask_cors import CORS
import config
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from openai import Client
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import os


openai_api_key = config.OPENAI_API_KEY

# Now you can use openai_api_key in your code
# print("OpenAI API Key:", openai_api_key)

# Initialize the Flask application
application = Flask(__name__)


# Enabling CORS for all routes
CORS(application)

# Random API key
API_KEY = 'ehtisham'


# Home Page Route
@application.route('/', methods=['GET'])
def home():
    return "Home route"


# Create Embeddings Route
# Define your OpenAI model
OPENAI_MODEL = 'text-embedding-3-small'


client = Client(api_key=openai_api_key)


def get_embedding(text, model="text-embedding-3-adalaah"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def fetch_text_from_txt(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text


# Generate Response Route
@application.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    try:
        if 'api_key' not in request.json:
            return jsonify({'error': 'API key not passed by the user.'}), 401

        if request.json['api_key'] != API_KEY:
            return jsonify({'error': 'API key not matched.'}), 401

        # Get the OpenAI API key from the environment variable or config module
        openai_api_key = os.environ.get('OPENAI_API_KEY') or config.OPENAI_API_KEY

        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not found.'}), 500

        # Initialize an empty list to store embeddings
        document_embeddings = []

        # Load documents from the 'data' directory
        # loader = DirectoryLoader('data', glob='**/*.txt')
        # documents = loader.load()

        # Fetch text from a specific file
        text = fetch_text_from_txt('data/cleaned_constitution_data.txt')
        # print("Text:", text)

        # Generate embedding for the text
        embedding = get_embedding(text, model=OPENAI_MODEL)
        document_embeddings.append(embedding)

        print(document_embeddings)

        # Save embeddings to a pickle file
        with open('embeddings.pkl', 'wb') as f:
            pickle.dump(document_embeddings, f)

        return jsonify({'message': 'Embeddings created and saved successfully.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    



@application.route('/create_embeddings2', methods=['POST'])
def create_embeddings2():
    try:
        if 'api_key' not in request.json:
            return jsonify({'error': 'API key not passed by the user.'}), 401

        if request.json['api_key'] != API_KEY:
            return jsonify({'error': 'API key not matched.'}), 401

        # Get the OpenAI API key from the environment variable or config module
        openai_api_key = os.environ.get('OPENAI_API_KEY') or config.OPENAI_API_KEY
        print(openai_api_key)

        if not openai_api_key:
            return jsonify({'error': 'OpenAI API key not found.'}), 500

        loader = TextLoader("data/CasesTextual.txt", encoding='UTF-8')
        # print(len(loader))
        documents = loader.load()
        print("Total documents:", len(documents))

        # Split documents into chunks of 7,000 lines each
        chunk_size = 3500
        for i in range(0, len(documents), chunk_size):
            print(chunk_size)
            chunk_documents = documents[i:i+chunk_size]
            print(len(documents))
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3500, chunk_overlap=0, separators=[" ", ",", "\n"])
            docs = text_splitter.split_documents(chunk_documents)
            print("Processing chunk", i, "with", len(docs), "documents")
            
            embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
            print(embeddings)
            
            db = FAISS.from_documents(docs, embeddings)
            print(db)

            # Serialize the new embeddings
            new_embeddings = db.serialize_to_bytes()
            print("New embeddings size:", len(new_embeddings))
            # print("New Embeddings::",new_embeddings)

            # Load existing embeddings or create an empty list if the file doesn't exist
            if os.path.exists('embeddings12.pkl'):
                with open('embeddings12.pkl', 'wb') as f:
                    existing_embeddings = pickle.load(f)
            else:
                existing_embeddings = []

            # Append new embeddings to existing embeddings
            combined_embeddings = new_embeddings
            print("Combined embeddings size:", len(combined_embeddings))
            # print("Combined Embeddings::",combined_embeddings)

            # Save the combined embeddings to the pickle file
            with open('embeddings12.pkl', 'wb') as f:
                pickle.dump(combined_embeddings, f)
            print("Embeddings saved successfully")

        return jsonify({'message': 'Embeddings created and saved successfully.'}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Generate Answer on the basis of query
@application.route('/generate_answer', methods=['POST'])
def generate_answer():
    try:
        if 'api_key' not in request.json:
            return jsonify({'error': 'API key not passed by the user.'}), 401

        if request.json['api_key'] != API_KEY:
            return jsonify({'error': 'API key not matched.'}), 401

        # Load serialized FAISS index from the pickle file
        with open('embeddings2.pkl', 'rb') as f:
            serialized_faiss_index = pickle.load(f)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Deserialize FAISS index from bytes
        db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_faiss_index)

        query = request.json.get('query')

        print(query)

        # Check if the query is None or empty
        if query is None or query.strip() == '':
            return jsonify({'error': 'Query is empty or not provided.'}), 400

        try:
            # Perform similarity search
            retriever = db.as_retriever(
                search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
            )

            docs = retriever.invoke(query)

            # Set up the template and model
            template = """Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"""
            prompt = ChatPromptTemplate.from_template(template)
            model = ChatOpenAI(openai_api_key= openai_api_key)

            # Define a function to format documents
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            # Define the processing chain
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            # Invoke the processing chain with the query
            result = chain.invoke(query)

            # Extract the answer from the result
            # answer = result['answer']

            return jsonify({'answer': result}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
            return jsonify({'error': str(e)}), 500



# Generate Answer with RetrievalQA
@application.route('/generate_answer2', methods=['POST'])
def generate_answer2():
    try:
        if 'api_key' not in request.json:
            return jsonify({'error': 'API key not passed by the user.'}), 401

        if request.json['api_key'] != API_KEY:
            return jsonify({'error': 'API key not matched.'}), 401

        # Load serialized FAISS index from the pickle file
        with open('embeddings2.pkl', 'rb') as f:
            serialized_faiss_index = pickle.load(f)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Deserialize FAISS index from bytes
        db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_faiss_index)

        query = request.json.get('query')

        print(query)

        # Check if the query is None or empty
        if query is None or query.strip() == '':
            return jsonify({'error': 'Query is empty or not provided.'}), 400

        try:
            # Perform similarity search
            retriever = db.as_retriever(
                search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5}
            )

            docs = retriever.invoke(query)

            # Set up the template and model
            template = """Answer the question based only on the following context:\n\n{context}\n\nQuestion: {question}"""
            prompt = ChatPromptTemplate.from_template(template)
            model = ChatOpenAI(openai_api_key= openai_api_key)

            # Define a function to format documents
            def format_docs(docs):
                return "\n\n".join([d.page_content for d in docs])

            # Define the processing chain
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | model
                | StrOutputParser()
            )

            # Invoke the processing chain with the query
            result = chain.invoke(query)

            # Extract the answer from the result
            # answer = result['answer']

            return jsonify({'answer': result}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
            return jsonify({'error': str(e)}), 500


# Generate Answer with RetrievalQA
@application.route('/generate_answer_retrieval', methods=['POST'])
def retrieval_answer():
    try:
        if 'api_key' not in request.json:
            return jsonify({'error': 'API key not passed by the user.'}), 401

        if request.json['api_key'] != API_KEY:
            return jsonify({'error': 'API key not matched.'}), 401

        # Load serialized FAISS index from the pickle file
        with open('embeddings2.pkl', 'rb') as f:
            serialized_faiss_index = pickle.load(f)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        print("I am here.")
        # Deserialize FAISS index from bytes
        db = FAISS.deserialize_from_bytes(embeddings=embeddings, serialized=serialized_faiss_index)

        query = request.json.get('query')

        print(query)

        llm = ChatOpenAI(openai_api_key= openai_api_key)

        # Check if the query is None or empty
        if query is None or query.strip() == '':
            return jsonify({'error': 'Query is empty or not provided.'}), 400

        try:
            # Create a RetrievalQA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.5})
            )

            # Pass question to the qa_chain
            result = qa_chain({"query": query})
            answer = result["result"]

            return jsonify({'answer': answer}), 200

        except Exception as e:
            return jsonify({'error': str(e)}), 500
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500



# Run the Flask application
if __name__ == '__main__':
    application.run(debug=True)
