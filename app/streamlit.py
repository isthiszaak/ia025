
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


# Project paths
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

RETRIEVAL_RESULTS_DIR = (
    PROJECT_ROOT
    / "data"
    / "retrieval"
    / "experiments"
)

RUNNER_MODULE = "app.evaluation.runner"
RAGAS_MODULE = "app.evaluation.ragas_adapter"
RETRIEVAL_MODULE = "app.retrieval"


# Retrieval experiment options supported by experiments.py
CHUNK_SIZES = [256, 512, 1024]
TOP_K_VALUES = [1, 3, 5]
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


def validate_retrieval_cases(cases):
    """Validate the fields required by the retrieval module."""

    if not cases:
        raise ValueError("The cases.jsonl file contains no cases.")

    for index, case in enumerate(cases, start=1):

        if not isinstance(case, dict):
            raise ValueError(
                f"Case on line {index} must be a JSON object."
            )

        if (
            not isinstance(case.get("user_input"), str)
            or not case["user_input"].strip()
        ):
            raise ValueError(
                f"Case on line {index} has an invalid user_input."
            )

    return True


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


def execute_command(command):
    """Execute a complete command from the project root."""

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


def get_retrieval_runs():
    """Return retrieval experiment directories sorted by newest first."""

    if not RETRIEVAL_RESULTS_DIR.exists():
        return []

    return sorted(
        [
            folder
            for folder in RETRIEVAL_RESULTS_DIR.iterdir()
            if folder.is_dir()
            and (folder / "manifest.json").exists()
        ],
        key=lambda folder: folder.name,
        reverse=True
    )


def load_json_file(path):
    """Load a JSON file."""

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl_file(path):
    """Load a JSONL file."""

    with open(path, "r", encoding="utf-8") as f:
        return [
            json.loads(line)
            for line in f
            if line.strip()
        ]


def run_retrieval_experiment(
    chunk_overlap,
    fixed_chunk_size,
    fixed_top_k,
    selection,
    limit,
    build_indexes
):
    """Execute the retrieval comparison experiment."""

    if not CASES_PATH.exists():
        raise FileNotFoundError(
            f"Cases file not found: {CASES_PATH}"
        )

    # Validate the saved dataset before starting the experiment.
    cases = load_cases()
    validate_retrieval_cases(cases)

    RETRIEVAL_RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    command = [
        sys.executable,
        "-m",
        RETRIEVAL_MODULE,
        "compare",

        "--cases",
        str(CASES_PATH),

        "--output-dir",
        str(RETRIEVAL_RESULTS_DIR),

        "--chunk-unit",
        "tokens",

        "--chunk-overlap",
        str(chunk_overlap),

        "--chunk-size",
        str(fixed_chunk_size),

        "--top-k",
        str(fixed_top_k),

        "--selection",
        str(selection)
    ]

    if limit is not None:
        command.extend([
            "--limit",
            str(limit)
        ])

    if build_indexes:
        command.append("--build-indexes")

    return execute_command(command)


def display_retrieval_run(run_dir):
    """Display the results of a selected retrieval experiment."""

    summary_path = run_dir / "manifest.json"

    if not summary_path.exists():
        st.error("Experiment manifest not found.")
        return

    summary = load_json_file(summary_path)

    st.subheader("Experiment Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Questions",
        summary.get("cases_count", 0)
    )

    col2.metric(
        "Retrievals",
        summary.get("retrieval_count", 0)
    )

    col3.metric(
        "Errors",
        summary.get("errors", 0)
    )

    st.caption(
        f"Run ID: {summary.get('run_id', run_dir.name)}"
    )

    st.caption(
        f"Timestamp (UTC): {summary.get('timestamp_utc', 'Unknown')}"
    )

    st.caption(
        f"Output directory: {run_dir}"
    )

    st.divider()

    st.subheader("Compared Configurations")

    runs = summary.get("runs", [])

    if not runs:
        st.warning("No comparison results found.")
        return

    comparison_df = pd.DataFrame(runs)

    st.dataframe(
        comparison_df,
        width='stretch',
        hide_index=True
    )

    st.divider()

    st.subheader("Detailed Retrieval Results")

    selected_setting = st.selectbox(
        "Select a configuration:",
        options=[
            item["setting"]
            for item in runs
        ],
        key="selected_retrieval_setting"
    )

    retrieval_path = (
        run_dir
        / selected_setting
        / "retrieval.jsonl"
    )

    if not retrieval_path.exists():
        st.warning("No retrieval.jsonl found for this configuration.")
        return

    results = load_jsonl_file(retrieval_path)

    if not results:
        st.warning("No retrieval results found.")
        return

    # Compact overview of all questions.
    overview = []

    for result in results:
        overview.append({
            "case_id": result.get("case_id"),
            "status": result.get("status"),
            "chunk_size": result.get("chunk_size"),
            "top_k": result.get("top_k"),
            "latency_seconds": result.get("latency_seconds"),
            "retrieved_chunks": len(
                result.get("retrieved_chunks", [])
            )
        })

    st.dataframe(
        pd.DataFrame(overview),
        width='stretch',
        hide_index=True
    )

    st.divider()

    st.subheader("Inspect Individual Questions")

    for index, result in enumerate(results):

        case_id = result.get(
            "case_id",
            f"Case {index + 1}"
        )

        question = result.get(
            "user_input",
            "Question not available"
        )

        with st.expander(
            f"{case_id} — {question[:100]}"
        ):

            st.markdown("**Question**")
            st.write(question)

            st.write(
                f"Status: {result.get('status', 'Unknown')}"
            )

            if result.get("latency_seconds") is not None:
                st.write(
                    "Latency: "
                    f"{result['latency_seconds']:.4f} seconds"
                )

            if result.get("status") == "error":

                st.error(
                    result.get("error", "Unknown error")
                )

                continue

            chunks = result.get(
                "retrieved_chunks",
                []
            )

            st.write(
                f"Retrieved chunks: {len(chunks)}"
            )

            for chunk_index, chunk in enumerate(chunks):

                st.markdown(
                    f"**Retrieved chunk {chunk_index + 1}**"
                )

                st.json(chunk)
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
        width='stretch',
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

chat_tab, editor_tab, dataset_tab, retrieval_tab = st.tabs([
    "💬 Chat",
    "⚙️ Prompt Editor",
    "📊 Evaluation Dataset",
    "🔎 Retrieval Experiments"
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
                width='stretch',
                hide_index=True,
                key="evaluation_dataset_editor"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                if st.button(
                    "💾 Save Dataset",
                    type="primary",
                    width='stretch'
                ):

                    save_cases(edited_df)

                    st.session_state.evaluation_df = edited_df

                    st.success(
                        "Evaluation dataset successfully updated!"
                    )

            with col2:

                if st.button(
                    "🔄 Reload Dataset",
                    width='stretch'
                ):

                    st.session_state.evaluation_df = pd.DataFrame(
                        load_cases()
                    )

                    st.rerun()

            with col3:

                if st.button(
                    "🚀 Run Generation Pipeline",
                    width='stretch'
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
        width="stretch",
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

        st.subheader("📊 RAGAS Evaluation Results")

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



# ============================================================
# TAB 4 - RETRIEVAL EXPERIMENTS
# ============================================================

with retrieval_tab:

    st.header("Retrieval Experiments")

    st.write(
        "Configure and execute retrieval experiments using "
        "the questions defined in cases.jsonl."
    )

    st.info(
        "This experiment evaluates document retrieval only. "
        "It does not generate answers or calculate RAGAS metrics."
    )

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    st.subheader("Experiment Dataset")

    if not CASES_PATH.exists():

        st.error(
            f"Dataset not found: {CASES_PATH}"
        )

        retrieval_cases = []

    else:

        try:

            retrieval_cases = load_cases()
            validate_retrieval_cases(retrieval_cases)

            st.success(
                f"{len(retrieval_cases)} valid questions loaded."
            )

            st.caption(
                f"Dataset: {CASES_PATH}"
            )

        except Exception as e:

            st.error(
                f"Could not load the experiment dataset: {e}"
            )

            retrieval_cases = []


    st.divider()


    # --------------------------------------------------------
    # EXPERIMENT PARAMETERS
    # --------------------------------------------------------

    st.subheader("Experiment Parameters")

    st.write(
        "The experiment uses a one-factor-at-a-time comparison. "
        "Five configurations will be executed."
    )

    col1, col2 = st.columns(2)

    with col1:

        fixed_chunk_size = st.selectbox(
            "Reference chunk size (tokens)",
            options=CHUNK_SIZES,
            index=1,
            help=(
                "The reference value used when comparing top-k. "
                "The experiment also tests all three chunk sizes."
            )
        )

        fixed_top_k = st.selectbox(
            "Reference top-k",
            options=TOP_K_VALUES,
            index=1,
            help=(
                "The reference value used when comparing chunk sizes. "
                "The experiment also tests top-k values 1, 3 and 5."
            )
        )

    with col2:

        chunk_overlap = st.number_input(
            "Chunk overlap (tokens)",
            min_value=0,
            max_value=255,
            value=50,
            step=1,
            help=(
                "The overlap is kept constant across all configurations."
            )
        )

        corpus_selection = st.radio(
            "Corpus selection",
            options=[
                "baseline",
                "all"
            ],
            format_func=lambda value: (
                "Baseline IRPF corpus"
                if value == "baseline"
                else "Entire prepared corpus"
            ),
            help=(
                "Choose whether to use the document selection "
                "defined in baseline_irpf.json or the entire corpus."
            )
        )


    # --------------------------------------------------------
    # CASE LIMIT
    # --------------------------------------------------------

    st.subheader("Questions to Execute")

    use_limit = st.checkbox(
        "Limit the number of questions",
        value=False
    )

    case_limit = None

    if use_limit:

        case_limit = st.number_input(
            "Maximum number of questions",
            min_value=1,
            max_value=max(1, len(retrieval_cases)),
            value=min(20, max(1, len(retrieval_cases))),
            step=1
        )

        st.caption(
            f"The first {case_limit} questions in cases.jsonl will be used."
        )

    else:

        st.caption(
            f"All {len(retrieval_cases)} questions will be used."
        )


    # --------------------------------------------------------
    # INDEX BUILDING
    # --------------------------------------------------------

    st.subheader("Index Configuration")

    build_indexes = st.checkbox(
        "Build or resume the required indexes automatically",
        value=True,
        help=(
            "Builds or resumes the indexes for chunk sizes "
            "256, 512 and 1024. Existing compatible indexes are reused."
        )
    )

    if not build_indexes:

        st.warning(
            "If an index for one of the configurations does not exist, "
            "that configuration will fail to open."
        )


    st.divider()


    # --------------------------------------------------------
    # EXPERIMENT PREVIEW
    # --------------------------------------------------------

    st.subheader("Experiment Preview")

    preview_df = pd.DataFrame([
        {
            "Configuration": f"Chunk {size}",
            "Chunk size": size,
            "Top-k": fixed_top_k
        }
        for size in CHUNK_SIZES
    ] + [
        {
            "Configuration": f"Top-k {top_k}",
            "Chunk size": fixed_chunk_size,
            "Top-k": top_k
        }
        for top_k in TOP_K_VALUES
        if top_k != fixed_top_k
    ])

    # The reference combination appears in both comparison groups.
    # experiments.py removes duplicated configurations internally.

    st.dataframe(
        preview_df,
        width='stretch',
        hide_index=True
    )

    st.caption(
        "The reference combination is shared by both comparison groups "
        "and is executed only once."
    )


    # --------------------------------------------------------
    # RUN EXPERIMENT
    # --------------------------------------------------------

    st.divider()

    st.subheader("Run Experiment")

    if st.button(
        "🚀 Run Retrieval Experiment",
        type="primary",
        width='stretch',
        disabled=not retrieval_cases
    ):

        selection = (
            "all"
            if corpus_selection == "all"
            else str(PROJECT_ROOT / "data" / "retrieval" /"baseline_irpf.json")
        )

        limit = (
            int(case_limit)
            if use_limit
            else None
        )

        with st.spinner(
            "Running retrieval experiment. "
            "This may take some time while indexes are built..."
        ):

            try:

                process = run_retrieval_experiment(
                    chunk_overlap=int(chunk_overlap),
                    fixed_chunk_size=int(fixed_chunk_size),
                    fixed_top_k=int(fixed_top_k),
                    selection=selection,
                    limit=limit,
                    build_indexes=build_indexes
                )

                save_execution_result(
                    "retrieval_result",
                    process
                )

                # The compare CLI prints a JSON summary.
                try:

                    output = json.loads(process.stdout)

                    if output.get("output_dir"):

                        st.session_state.retrieval_selected_run = (
                            Path(output["output_dir"])
                        )

                except (json.JSONDecodeError, TypeError):
                    pass

            except Exception as e:

                st.session_state.retrieval_result = {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": str(e)
                }


    # --------------------------------------------------------
    # EXECUTION RESULTS
    # --------------------------------------------------------

    if "retrieval_result" in st.session_state:

        display_execution_result(
            "🔎 Retrieval Experiment Execution",
            st.session_state.retrieval_result
        )


    # --------------------------------------------------------
    # PREVIOUS EXPERIMENTS
    # --------------------------------------------------------

    st.divider()

    st.header("Previous Retrieval Experiments")

    retrieval_runs = get_retrieval_runs()

    if retrieval_runs:

        selected_retrieval_run = st.selectbox(
            "Select an experiment:",
            options=retrieval_runs,
            format_func=lambda folder: folder.name,
            key="retrieval_run_selector"
        )

        display_retrieval_run(selected_retrieval_run)

    else:

        st.info(
            "No retrieval experiments found. "
            "Run an experiment to see its results here."
        )
