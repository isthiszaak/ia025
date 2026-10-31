
import streamlit as st
import json
from pathlib import Path
import pandas as pd

from backend.llm import get_llm
from backend.pipeline import run_query
from backend.prompts import CHAT_TEMPLATE


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Local LLM Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Local LLM Chatbot")

# Path to evaluation dataset
CASES_PATH = (
    Path(__file__).resolve().parent.parent
    / "app"
    / "evaluation"
    / "evaluation_data"
    / "cases.jsonl"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def load_llm():
    return get_llm()


def load_cases():
    """Load evaluation cases from JSONL file."""
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def save_cases(df):
    """Overwrite the original JSONL file."""
    with open(CASES_PATH, "w", encoding="utf-8") as f:
        for case in df.to_dict(orient="records"):
            f.write(
                json.dumps(case, ensure_ascii=False) + "\n"
            )


# ============================================================
# INITIALIZATION
# ============================================================

llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt_template_text" not in st.session_state:
    st.session_state.prompt_template_text = CHAT_TEMPLATE


# ============================================================
# TABS
# ============================================================

chat_tab, editor_tab, dataset_tab = st.tabs([
    "💬 Chat",
    "⚙️ Prompt Editor",
    "📊 Evaluation Dataset"
])


# ============================================================
# TAB 1 - CHAT
# ============================================================

with chat_tab:

    st.write(
        "This is a simple chatbot interface "
        "that uses a local LLM model served by Ollama."
    )

    st.warning(
        "COLD START WARNING: The first time you run "
        "this app, it may take a few minutes to pull the model."
    )

    # Display conversation history
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User input
    if prompt := st.chat_input("Type a message..."):

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        # Invoke LLM
        with st.chat_message("assistant"):

            try:
                # Build conversation history
                chat_history = "\n".join(
                    [
                        f"{message['role']}: {message['content']}"
                        for message in st.session_state.messages[:-1]
                    ]
                )
                result = run_query(
                    prompt,
                    llm,
                    st.session_state.prompt_template_text,
                    chat_history
                )

                full_response = result.response
                st.write(full_response)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:
                st.error(f"Error processing query: {e}")


# ============================================================
# TAB 2 - PROMPT EDITOR
# ============================================================

with editor_tab:

    st.header("Prompt Template Editor")

    st.write(
        "Modify the prompt template below. "
        "Your changes will apply to the next message you send."
    )

    updated_template = st.text_area(
        "Edit Template (keep {user_input} as a placeholder):",
        value=st.session_state.prompt_template_text,
        height=250
    )

    if st.button("💾 Save Prompt Changes"):

        st.session_state.prompt_template_text = updated_template

        st.success("Prompt template successfully updated!")


# ============================================================
# TAB 3 - EVALUATION DATASET EDITOR
# ============================================================

with dataset_tab:

    st.header("Evaluation Dataset Editor")

    st.write(
        "Edit, add or remove evaluation cases. "
        "Changes will overwrite the original cases.jsonl file."
    )

    if not CASES_PATH.exists():

        st.error(f"Dataset not found: {CASES_PATH}")

    else:

        try:

            # Load dataset
            if "evaluation_df" not in st.session_state:
                cases = load_cases()
                st.session_state.evaluation_df = pd.DataFrame(cases)

            st.caption(
                f"Dataset: {CASES_PATH} | "
                f"Cases: {len(st.session_state.evaluation_df)}"
            )

            # Editable table
            edited_df = st.data_editor(
                st.session_state.evaluation_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="evaluation_dataset_editor"
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button("💾 Save Dataset", type="primary"):

                    save_cases(edited_df)

                    st.session_state.evaluation_df = edited_df

                    st.success(
                        "Evaluation dataset successfully updated!"
                    )

            with col2:

                if st.button("🔄 Reload Dataset"):

                    st.session_state.evaluation_df = pd.DataFrame(
                        load_cases()
                    )

                    st.rerun()

        except Exception as e:

            st.error(f"Error handling evaluation dataset: {e}")
