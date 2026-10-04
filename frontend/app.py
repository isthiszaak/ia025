import streamlit as st
from langchain_ollama import OllamaLLM

st.set_page_config(page_title="Local LLM Chatbot", page_icon="🤖")
st.title("🤖 Local LLM Chatbot")
st.write("This is a simple chatbot interface that uses a local LLM model served by Ollama. You can type your messages below and receive responses from the model.")
st.warning("COLD START WARNING: The first time you run this app, it may take a few minutes to pull the model. Please be patient.")

# Initialize local LLM client
@st.cache_resource
def get_llm():
    return OllamaLLM(
        model="qwen2.5:1.5b",
        base_url="http://ollama:11434"
    )

llm = get_llm()

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous conversation messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle user input
if prompt := st.chat_input("Type a message..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Stream response from Ollama
    with st.chat_message("assistant"):
        response_stream = llm.stream(prompt)
        full_response = st.write_stream(response_stream)

    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})