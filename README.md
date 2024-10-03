# Smart-Support-system-RAG
A Rag based support system with the ability to draft email to customer queries.

# Chatbot with Document Retrieval and Email Generation

This repository contains a Flask-based chatbot application that integrates OpenAI's GPT-3.5-turbo model for conversational responses, document retrieval for context-aware answers, and email generation functionality.

## Features

1. **Chatbot Integration**:
   - Users can input questions and receive streamed responses from the chatbot powered by OpenAI's GPT-3.5-turbo model.
   - The chatbot uses document retrieval to search through uploaded documents/ linked websites and return context-aware answers.
   
2. **Document Retrieval**:
   - Supports loading documents from PDF, DOCX, and web URLs.
   - Uses LangChain's document loaders and FAISS vector store to perform similarity searches on the documents.
   
3. **Email Generation**:
   - Generates a customer support email based on the chatbot's last response.
   - Email is structured in valid HTML format for easy parsing.

4. **Chat History Management**:
   - Clear the chat history with a simple API call.
   - Chat history is used to maintain context during conversations.

4. **Copy button to copy bot messages**:
   - Copy the email content/ bot instruction with simple click on the bot message.
   

## Installation

### Prerequisites

- Python 3.7+
- Flask
- OpenAI API Key
- `dotenv` for environment variables
- Additional libraries (`langchain`, `FAISS`, `openai`, etc.)

### Installation Steps

1. Clone the repository:

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    ```

2. Navigate to the project directory:

    ```bash
    cd your-repo-name
    ```

3. Create a virtual environment and activate it:

    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

4. Install the dependencies:

    ```bash
    pip install -r requirements.txt
    ```

5. Create a `.env` file in the project root to store your OpenAI API key:

    ```bash
    touch .env
    ```

6. Add your OpenAI API key to the `.env` file:

    ```bash
    OPENAI_API_KEY=your-openai-api-key
    ```

7. Set up any necessary environment variables.

## Usage

### Running the Application

1. Start the Flask application:

    ```bash
    python app.py
    ```

2. Open a browser and go to `http://127.0.0.1:5000/` to interact with the chatbot.

### API Endpoints

- `/`: Handles chatbot interactions via POST requests.
- `/clear_chat`: Clears the chat history.
- `/generate_email`: Generates an email based on the last bot response.

### Document Upload

- The application supports PDF, DOCX, and web URL document loading.
- Modify the `file_paths` and `web_urls` variables in `app.py` to add your own documents for retrieval.

## Code Overview

### `app.py`

- **Flask Setup**: Initializes the Flask application and API routes.
- **DocumentReader Class**: Reads PDF, DOCX files, and web URLs.
- **TextSplitter Class**: Splits large documents into smaller chunks for easier processing.
- **VectorSearch Class**: Uses FAISS and OpenAI embeddings to perform similarity searches on the documents.
- **Routes**:
  - `POST /`: Handles the user's question, retrieves relevant documents, and returns a context-aware response.
  - `POST /clear_chat`: Clears the chat history.
  - `POST /generate_email`: Generates a support email from the bot's last response.

### `static/app.js`

- **Handles form submission**: Sends the user's question to the backend and streams the bot's response.
- **Manages UI updates**: Displays user and bot messages, provides a "Copy" button for bot responses, and handles chat scrolling.
- **Email generation**: Streams the generated email back to the user.

## Environment Variables

Make sure to include your OpenAI API key in the `.env` file:

