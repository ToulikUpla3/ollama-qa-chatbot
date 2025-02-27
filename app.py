from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms.ollama import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader  # For PDF files
import docx  # For Word files
from functools import lru_cache  # For caching

# Load environment variables
load_dotenv()

# LangSmith tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Ollama"

# Prompt template with context
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to user queries based on the following context:\n\nContext: {context}\n\n"),
        ("user", "Question: {question}")
    ]
)

# Cached function for generating responses
@lru_cache(maxsize=100)
def generate_response(question, engine, temperature, max_tokens, context=""):
    llm = Ollama(model=engine, temperature=temperature, max_tokens=max_tokens)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    try:
        for chunk in chain.stream({'question': question, 'context': context}):
            yield chunk
    except Exception as e:
        yield f"An error occurred: {str(e)}"

# Function to extract text from uploaded files
def extract_text_from_file(file):
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages])
    elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx.Document(file)
        text = "\n".join([para.text for para in doc.paragraphs])
    else:
        text = file.getvalue().decode("utf-8")  # For plain text files
    return text

# Streamlit app
st.title("Q&A Chatbot with Open Source Models")

# Sidebar for model selection and parameters
engine = st.sidebar.selectbox("Select an Ollama model", ["gemma2", "llama3.2", "mistral", "llama2"])
temperature = st.sidebar.slider('Temperature', min_value=0.0, max_value=1.0, value=0.7)
max_tokens = st.sidebar.slider('Max Tokens', min_value=50, max_value=300, value=150)

# File upload for context
uploaded_file = st.sidebar.file_uploader("Upload a file for context", type=["pdf", "txt", "docx"])
file_context = ""
if uploaded_file:
    file_context = extract_text_from_file(uploaded_file)
    st.sidebar.write("File uploaded successfully!")

# Initialize session state for conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Clear conversation button
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.experimental_rerun()

# Main interface
user_input = st.chat_input("Ask a question:")

if user_input:
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Combine conversation history and file context
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages])
    if file_context:
        context += f"\n\nFile Context:\n{file_context}"

    # Generate and stream response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""
        for chunk in generate_response(user_input, engine, temperature, max_tokens, context):
            full_response += chunk
            response_placeholder.write(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
