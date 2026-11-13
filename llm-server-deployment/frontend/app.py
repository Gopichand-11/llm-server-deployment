
import os
import requests
import streamlit as st


# Page configuration
st.set_page_config(
    page_title="Llama 3.2 Chatbot",
    page_icon="🤖",
    layout="centered"
)


# Backend configuration
BACKEND_URL = os.getenv(
    "BACKEND_URL",
    "http://127.0.0.1:8000"
)


# Application title
st.title("🤖 Llama 3.2 Chatbot")
st.caption("Powered by FastAPI and Ollama")


# Initialize conversation history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# User input
prompt = st.chat_input("Ask something...")


if prompt:

    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Generating response..."):

            try:
                response = requests.post(
                    f"{BACKEND_URL}/api/chat",
                    json={
                        "messages": st.session_state.messages
                    },
                    timeout=200
                )

                response.raise_for_status()

                answer = response.json()["answer"]

                st.markdown(answer)

                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })

            except requests.RequestException as error:

                st.error(
                    f"Unable to connect to backend: {error}"
                )

                # Remove failed user message
                st.session_state.messages.pop()


# Clear conversation
if st.sidebar.button("Clear Conversation"):
    st.session_state.messages = []
    st.rerun()
