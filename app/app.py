import streamlit as st
from backend.llm import get_llm
from backend.prompts import CHAT_TEMPLATE
from langchain_core.prompts import PromptTemplate

# Set page configuration and title
st.set_page_config(page_title="Local LLM Chatbot", page_icon="🤖")
st.title("🤖 Local LLM Chatbot")

# Initialize local LLM client
@st.cache_resource
def load_llm():
    return get_llm()

llm = load_llm()

# Initialize session state for message history and prompt template
if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt_template_text" not in st.session_state:
    st.session_state.prompt_template_text = CHAT_TEMPLATE

# --- CREATE TABS ---
chat_tab, editor_tab = st.tabs(["💬 Chat", "⚙️ Prompt Editor"])

with editor_tab:
    st.header("Prompt Template Editor")
    st.write("Modify the prompt template below. Your changes will apply immediately to the next message you send.")
    
    # Text area to edit the prompt live
    updated_template = st.text_area(
        "Edit Template (keep {user_input} as a placeholder):",
        value=st.session_state.prompt_template_text,
        height=200
    )
    
    if st.button("Save Changes"):
        st.session_state.prompt_template_text = updated_template
        st.success("Prompt template successfully updated!")

with chat_tab:
    st.write("This is a simple chatbot interface that uses a local LLM model served by Ollama.")
    st.warning("COLD START WARNING: The first time you run this app, it may take a few minutes to pull the model. Please be patient.")

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

        # Build dynamic prompt template from session state
        dynamic_prompt_template = PromptTemplate(
            input_variables=["user_input"],
            template=st.session_state.prompt_template_text
        )

        # Stream/Invoke response from Ollama
        with st.chat_message("assistant"):
            chain = dynamic_prompt_template | llm
            full_response = chain.invoke({"user_input": prompt})
            st.write(full_response)

        # Save response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})