import os
import streamlit as st
import pdfplumber
from groq import Groq  # Import Groq client
import pandas as pd
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

os.environ["GROQ_API_KEY"] = "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"

# Streamlit page configuration
st.set_page_config(
    page_title="Chat-Bot crafter",
    page_icon="",
    layout="centered"
)
client = Groq()

# Initialize session state
if "parameters" not in st.session_state:
    st.session_state.parameters = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Parameter selection page
if not st.session_state.parameters:
    st.title("Create Your Own ChatBot🤖")
    st.markdown("Please select the parameters for your chatbot:")
    selected_model = st.selectbox("Choose LLM Model:", ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama-3-3b-8192", "gemma2-9b-it", "gemma-7b-it", "mistral-8x7b-32768"])
    chatbot_description = st.text_area("Chatbot Description:")
    chatbot_name = st.text_input("Chatbot Name:")
    st.markdown("**Use Knowledge Base?**")
    use_knowledge_base = st.radio("", ["Yes", "No"])
    uploaded_files = None

    if use_knowledge_base == "Yes":
        uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

    if st.button("Submit"):
        documents = []
        if uploaded_files:
          for file in uploaded_files:
              try:
                  filename = file.name
                  with pdfplumber.open(file) as pdf:
                      text = ""
                      for page in pdf.pages:
                          text += page.extract_text()
                  documents.append(text)
              except Exception as e:
                  st.error(f"Error loading file {filename}: {str(e)}")

        st.session_state.parameters = {
            "selected_model": selected_model,
            "chatbot_description": chatbot_description,
            "chatbot_name": chatbot_name,
            "use_knowledge_base": use_knowledge_base,
            "documents": documents  # Store the processed documents
        }
        st.success("Parameters submitted. Redirecting to your ChatBot...")

# Chat interface page
if st.session_state.parameters:
    st.title(f"Chat With {st.session_state.parameters['chatbot_name']}")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input field for user's message:
    user_prompt = st.chat_input(f"Ask {st.session_state.parameters['chatbot_name']}")

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})

        # Send user's message to the selected LLM
        messages = [
            {"role": "system", "content": st.session_state.parameters["chatbot_description"]},
            {"role": "system", "content": f"You are a helpful assistant named {st.session_state.parameters['chatbot_name']}."},
            *st.session_state.chat_history
        ]

        if st.session_state.parameters["use_knowledge_base"] == "Yes" and st.session_state.parameters["documents"]:
          # Use the pre-processed documents from parameter selection
          documents = st.session_state.parameters["documents"]
          embeddings = HuggingFaceEmbeddings()
          vectorstore = FAISS.from_texts(documents, embeddings)
          # Search knowledge base
          docs = vectorstore.similarity_search(query=user_prompt, k=3)
          context = "\n".join([doc.page_content for doc in docs])  # Modified here
          messages.append({"role": "system", "content": context})

        response = client.chat.completions.create(
            model=st.session_state.parameters["selected_model"],
            messages=messages,
            temperature=1,
            max_tokens=500,
            top_p=1,
            stream=False
        )

        assistant_response = response.choices[0].message.content
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})

        # Display the LLM's response
        with st.chat_message("assistant"):
            st.markdown(assistant_response)
