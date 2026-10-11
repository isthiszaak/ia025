import streamlit as st
from backend.llm import get_llm
from langchain_core.prompts import PromptTemplate

# Set page configuration and title
st.set_page_config(page_title="Local LLM Chatbot", page_icon="🤖")
st.title("🤖 Local LLM Chatbot")
st.write("This is a simple chatbot interface that uses a local LLM model served by Ollama. You can type your messages below and receive responses from the model.")
st.warning("COLD START WARNING: The first time you run this app, it may take a few minutes to pull the model. Please be patient.")

# Initialize local LLM client
@st.cache_resource
def load_llm():
    return get_llm()

llm = load_llm()

template = """Your name is Bob. 
User question: {user_input}
Answer:"""

prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template=template
)


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
        
        chain = prompt_template | llm
        full_response = chain.invoke({"user_input": prompt})
        st.write(full_response)

    # Save response to history
    st.session_state.messages.append({"role": "assistant", "content": full_response})