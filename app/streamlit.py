
import streamlit as st
import json
import subprocess
import sys

from pathlib import Path
from datetime import datetime, timezone
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

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

CASES_PATH = (
    APP_DIR
    / "evaluation"
    / "evaluation_data"
    / "cases.jsonl"
)

RESULTS_DIR = (
    APP_DIR
    / "evaluation"
    / "evaluation_results"
)

RUNNER_MODULE = "app.evaluation.runner"
RAGAS_MODULE = "app.evaluation.ragas_adapter"


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


def execute_script(module_name, extra_args=None):
    """Execute a Python module from the project root."""

    command = [
        sys.executable,
        "-m",
        module_name
    ]

    if extra_args:
        command.extend(extra_args)

    return subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True
    )


def display_execution_result(name, result):
    """Display execution status, stdout and stderr."""

    st.subheader(name)

    if result["returncode"] == 0:
        st.success("Execution completed successfully!")
    else:
        st.error(
            f"Execution failed (exit code: {result['returncode']})"
        )

    if result["stdout"]:
        with st.expander("Standard Output", expanded=True):
            st.code(result["stdout"], language="text")

    if result["stderr"]:
        with st.expander("Standard Error", expanded=True):
            st.code(result["stderr"], language="text")

    if not result["stdout"] and not result["stderr"]:
        st.info("No output was generated.")


def save_execution_result(session_key, process):
    """Store subprocess result in Streamlit session state."""

    st.session_state[session_key] = {
        "returncode": process.returncode,
        "stdout": process.stdout,
        "stderr": process.stderr
    }


def get_evaluation_runs():
    """Return evaluation directories sorted by newest first."""

    if not RESULTS_DIR.exists():
        return []

    return sorted(
        [
            folder
            for folder in RESULTS_DIR.iterdir()
            if folder.is_dir()
            and (folder / "predictions.jsonl").exists()
        ],
        key=lambda folder: folder.name,
        reverse=True
    )


def load_ragas_summary(summary_path):
    """Load RAGAS summary JSON."""

    with open(summary_path, "r", encoding="utf-8") as f:
        return json.load(f)


def display_ragas_summary(summary):
    """Display RAGAS evaluation summary."""

    st.divider()
    st.header("📊 RAGAS Evaluation Summary")

    # General information
    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Total Cases",
            summary.get("total_cases", 0)
        )

    with col2:
        st.metric(
            "Standard Deviation",
            summary.get("standard_deviation", "N/A")
        )

    # Metrics table
    metrics = summary.get("metrics", {})

    rows = []

    for metric_name, values in metrics.items():

        mean = values.get("mean")
        stddev = values.get("stddev")

        rows.append({
            "Metric": metric_name,
            "Mean": mean,
            "Std Dev": stddev,
            "Valid": values.get("valid_count", 0),
            "Failures": values.get("failure_count", 0),
            "Not Applicable": values.get("not_applicable_count", 0),
            "Skipped": values.get("skipped_count", 0)
        })

    metrics_df = pd.DataFrame(rows)

    st.dataframe(
        metrics_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Mean": st.column_config.NumberColumn(
                "Mean",
                format="%.4f"
            ),
            "Std Dev": st.column_config.NumberColumn(
                "Std Dev",
                format="%.4f"
            )
        }
    )

    # Detailed metric status
    with st.expander("Metric details"):

        for metric_name, values in metrics.items():

            st.markdown(f"**{metric_name}**")

            st.json(values)


# ============================================================
# INITIALIZATION
# ============================================================

llm = load_llm()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "prompt_template_text" not in st.session_state:
    st.session_state.prompt_template_text = CHAT_TEMPLATE

if "evaluation_df" not in st.session_state:
    st.session_state.evaluation_df = None

if "ragas_summary" not in st.session_state:
    st.session_state.ragas_summary = None

if "ragas_output_dir" not in st.session_state:
    st.session_state.ragas_output_dir = None


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

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Type a message..."):

        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            try:
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

    # --------------------------------------------------------
    # DATASET EDITOR
    # --------------------------------------------------------

    if not CASES_PATH.exists():

        st.error(f"Dataset not found: {CASES_PATH}")

    else:

        try:

            if st.session_state.evaluation_df is None:

                cases = load_cases()

                st.session_state.evaluation_df = pd.DataFrame(cases)

            st.caption(
                f"Dataset: {CASES_PATH} | "
                f"Cases: {len(st.session_state.evaluation_df)}"
            )

            edited_df = st.data_editor(
                st.session_state.evaluation_df,
                num_rows="dynamic",
                use_container_width=True,
                hide_index=True,
                key="evaluation_dataset_editor"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                if st.button(
                    "💾 Save Dataset",
                    type="primary",
                    use_container_width=True
                ):

                    save_cases(edited_df)

                    st.session_state.evaluation_df = edited_df

                    st.success(
                        "Evaluation dataset successfully updated!"
                    )

            with col2:

                if st.button(
                    "🔄 Reload Dataset",
                    use_container_width=True
                ):

                    st.session_state.evaluation_df = pd.DataFrame(
                        load_cases()
                    )

                    st.rerun()

            with col3:

                if st.button(
                    "🚀 Run Generation Pipeline",
                    use_container_width=True
                ):

                    with st.spinner("Running generation pipeline..."):

                        try:
                            process = execute_script(RUNNER_MODULE)

                            save_execution_result(
                                "runner_result",
                                process
                            )

                        except Exception as e:

                            st.session_state.runner_result = {
                                "returncode": -1,
                                "stdout": "",
                                "stderr": str(e)
                            }

        except Exception as e:

            st.error(f"Error handling evaluation dataset: {e}")


    # --------------------------------------------------------
    # EVALUATION RUN SELECTION
    # --------------------------------------------------------

    st.divider()

    st.header("Evaluation Execution")

    st.write(
        "Select a previous evaluation run to calculate RAGAS metrics."
    )

    evaluation_runs = get_evaluation_runs()

    selected_run_dir = None
    predictions_path = None

    if evaluation_runs:

        selected_run = st.selectbox(
            "Select an evaluation run:",
            options=evaluation_runs,
            format_func=lambda folder: folder.name,
            key="selected_evaluation_run"
        )

        selected_run_dir = selected_run
        predictions_path = selected_run_dir / "predictions.jsonl"

        st.caption(f"Selected directory: {selected_run_dir}")

        if predictions_path.exists():
            st.success("predictions.jsonl found.")
        else:
            st.warning(
                "The selected directory does not contain predictions.jsonl."
            )

    else:

        st.info(
            "No evaluation runs found. Execute the evaluation pipeline first."
        )


    # --------------------------------------------------------
    # RUN RAGAS
    # --------------------------------------------------------

    st.subheader("Run Evaluation")

    if st.button(
        "📊 Run RAGAS",
        use_container_width=True,
        disabled=(
            predictions_path is None
            or not predictions_path.exists()
        )
    ):

        timestamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%S%fZ"
        )

        ragas_output_dir = (
            selected_run_dir
            / f"ragas_{timestamp}"
        )

        summary_path = ragas_output_dir / "summary.json"

        with st.spinner(
            f"Running RAGAS on {selected_run_dir.name}..."
        ):

            try:

                process = execute_script(
                    RAGAS_MODULE,
                    extra_args=[
                        str(predictions_path),
                        "--output-dir",
                        str(ragas_output_dir)
                    ]
                )

                save_execution_result(
                    "ragas_result",
                    process
                )

                st.session_state.ragas_selected_run = (
                    selected_run_dir.name
                )

                st.session_state.ragas_output_dir = str(
                    ragas_output_dir
                )

                if process.returncode == 0:

                    if summary_path.exists():

                        st.session_state.ragas_summary = (
                            load_ragas_summary(summary_path)
                        )

                    else:

                        st.session_state.ragas_summary = None

                        st.warning(
                            f"RAGAS completed, but summary.json "
                            f"was not found at: {summary_path}"
                        )

                else:

                    st.session_state.ragas_summary = None

            except Exception as e:

                st.session_state.ragas_result = {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(e)
                }

                st.session_state.ragas_summary = None


    # --------------------------------------------------------
    # EXECUTION RESULTS
    # --------------------------------------------------------

    if "runner_result" in st.session_state:

        display_execution_result(
            "🚀 Evaluation Pipeline Results",
            st.session_state.runner_result
        )

    if "ragas_result" in st.session_state:

        st.divider()

        st.header("📊 RAGAS Execution Results")

        st.caption(
            "Source run: "
            + st.session_state.get("ragas_selected_run", "Unknown")
        )

        if st.session_state.ragas_output_dir:
            st.caption(
                f"Output directory: {st.session_state.ragas_output_dir}"
            )

        display_execution_result(
            "RAGAS Output",
            st.session_state.ragas_result
        )

    # Display the summary whenever it is available.
    if st.session_state.ragas_summary is not None:

        display_ragas_summary(
            st.session_state.ragas_summary
        )
