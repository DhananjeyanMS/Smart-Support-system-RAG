from flask import Flask, request, render_template, jsonify, Response, stream_with_context
from dotenv import load_dotenv
import os
import openai
import logging
from langchain.document_loaders import PyPDFLoader, UnstructuredURLLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

app = Flask(__name__)

# Logging setup for error handling
logging.basicConfig(level=logging.INFO)

# Access the OpenAI API key
openai_api_key = os.getenv('OPENAI_API_KEY')


### MODULES ###
class DocumentReader:
    @staticmethod
    def read_pdf(file_path):
        try:
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            logging.error(f"Error reading PDF file {file_path}: {str(e)}")
            return []

    @staticmethod
    def read_docx(file_path):
        try:
            loader = UnstructuredWordDocumentLoader(file_path)
            documents = loader.load()
            return documents
        except Exception as e:
            logging.error(f"Error reading DOCX file {file_path}: {str(e)}")
            return []

    @staticmethod
    def read_files(file_paths):
        all_documents = []
        for file_path in file_paths:
            if file_path.endswith('.pdf'):
                documents = DocumentReader.read_pdf(file_path)
            elif file_path.endswith('.docx'):
                documents = DocumentReader.read_docx(file_path)
            else:
                logging.warning(f"Unsupported file type: {file_path}")
                continue
            all_documents.extend(documents)
        return all_documents

class TextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )

    def split_documents(self, documents):
        try:
            return self.text_splitter.split_documents(documents)
        except Exception as e:
            logging.error(f"Error splitting documents: {str(e)}")
            return []


class VectorSearch:
    def __init__(self, embeddings):
        self.embeddings = embeddings
        self.docsearch = None

    def create_vector_store(self, documents):
        try:
            self.docsearch = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logging.error(f"Error creating vector store: {str(e)}")

    def similarity_search(self, query, k=10):
        try:
            return self.docsearch.similarity_search_with_score(query, k=k)
        except Exception as e:
            logging.error(f"Error during similarity search: {str(e)}")
            return []


# Initialize the new modules
document_reader = DocumentReader()
text_splitter = TextSplitter()
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model='text-embedding-ada-002')
vector_search = VectorSearch(embeddings)

# Read files and web documents
file_paths = ["D:/Study materials/Dhananjeyan FT.pdf", "D:/Study materials/Life_draft.docx"]
combined_documents = document_reader.read_files(file_paths)

web_urls = ["https://docs.google.com/document/d/17QfYAAzvjqKLhEP39YAK4kShfV5MMbKll5O_Qvob1T4/edit"]
try:
    web_loader = UnstructuredURLLoader(urls=web_urls)
    web_documents = web_loader.load()
except Exception as e:
    logging.error(f"Error loading web URLs: {str(e)}")
    web_documents = []

# Combine all documents
all_documents = combined_documents + web_documents

# Split documents
split_docs = text_splitter.split_documents(all_documents)

# Set up vector store
vector_search.create_vector_store(split_docs)

# Define chat history
chat_history = []


### ROUTES ###
@app.route('/clear_chat', methods=['POST'])
def clear_chat():
    global chat_history
    chat_history = []
    return jsonify({'response': 'Chat history cleared.'})


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        data = request.get_json()
        question = data.get('question')
        if question:
            def generate():
                try:
                    # Build the messages for the conversation
                    messages = []
                    for q, a in chat_history:
                        messages.append({"role": "user", "content": q})
                        messages.append({"role": "assistant", "content": a})
                    messages.append({"role": "user", "content": question})

                    # Perform similarity search
                    docs_and_scores = vector_search.similarity_search(question, k=10)
                    SIMILARITY_THRESHOLD = 0.2
                    relevant_docs = [doc for doc, score in docs_and_scores if score >= SIMILARITY_THRESHOLD]

                    if not relevant_docs:
                        answer = "I'm sorry, I don't have the information about this right now."
                        yield f'data: {answer}\n\n'
                        chat_history.append((question, answer))
                        return

                    # Build context from documents
                    context = ""
                    sources = set()
                    for doc in relevant_docs:
                        source = doc.metadata.get('source', 'Unknown source')
                        sources.add(source)
                        context += f"Source: {source}\n{doc.page_content}\n\n"

                    system_message = f"""You are an expert support specialist for the -- product. Your task is to provide accurate, reliable support by answering user queries.

- If the user's input is a greeting or a common courtesy, respond appropriately.
- If you don't know the answer to a question based on the provided context, reply with "I'm sorry, I don't have the information right now."
- Make sure every response is structured in valid innerHTML format, using proper tags like <p>, <ul>, <li>, <br>, and <strong> to ensure easy parsing and readability.
- In your answer, please mention the sources of the information by appending the source names at the end of your reply in a new line without HTML tags, prefixed with 'Sources:'.

Use ONLY the following context to answer the question. Do not use prior knowledge except for greetings.

Context:
{context}
                """
                    messages.insert(0, {"role": "system", "content": system_message})

                    # Call OpenAI with streaming enabled
                    response = openai.ChatCompletion.create(
                        api_key=openai_api_key,
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0,
                        stream=True
                    )

                    answer = ''
                    for chunk in response:
                        if 'choices' in chunk and len(chunk['choices']) > 0:
                            token = chunk['choices'][0]['delta'].get('content', '')
                            answer += token
                            yield f'data: {token}\n\n'
                    chat_history.append((question, answer))

                except Exception as e:
                    logging.error(f"Error during question processing: {str(e)}")
                    yield f'data: Error processing the question\n\n'

            return Response(stream_with_context(generate()), mimetype='text/event-stream')

        return jsonify({'response': 'No question provided'}), 400
    return render_template('index.html')


@app.route('/generate_email', methods=['POST'])
def generate_email():
    if not chat_history:
        return jsonify({'response': 'No previous bot message to generate email from.'}), 400

    last_bot_message = chat_history[-1][1]

    def generate():
        try:
            custom_prompt = f"""Act like a professional customer support email writer. You have been crafting clear, empathetic, and efficient support emails for over 15 years. Ensure the email is structured in valid innerHTML format, using proper tags like <p>, <ul>, <li>, <br>, and <strong> with proper openings and closings to ensure easy parsing and readability for any systems that need to interpret or present the content.
Here is the mail content:
{last_bot_message}

Your Email:
"""
            response = openai.ChatCompletion.create(
                api_key=openai_api_key,
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": custom_prompt}],
                temperature=0,
                stream=True
            )

            email_response = ''
            for chunk in response:
                token = chunk['choices'][0]['delta'].get('content', '')
                email_response += token
                yield f'data: {token}\n\n'
            chat_history.append(("Generated Email", email_response))

        except Exception as e:
            logging.error(f"Error generating email: {str(e)}")
            yield f'data: Error generating email\n\n'

    return Response(stream_with_context(generate()), mimetype='text/event-stream')


if __name__ == '__main__':
    app.run(debug=True)
