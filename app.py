
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Configuration
# --------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "all_cases.csv"
MODEL_PATH = BASE_DIR / "bail_prediction_model.joblib"
RESULTS_DIR = BASE_DIR / "final_results"

st.set_page_config(
    page_title="Judgment Law AI",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Custom styling
# --------------------------------------------------
st.markdown(
    """
    <style>

/* Light mode: keep the hero gradient and make cards readable */
[data-testid="stAppViewContainer"] .hero-title {
    color: #f8fafc !important;
}

[data-testid="stAppViewContainer"] .hero-subtitle {
    color: #cbd5e1 !important;
}

[data-testid="stAppViewContainer"] .hero-eyebrow,
[data-testid="stAppViewContainer"] .hero-footer {
    color: #93c5fd !important;
}

[data-testid="stAppViewContainer"] div[data-testid="stMetric"] {
    background: #ffffff !important;
    border: 1px solid #dbe3ef !important;
}

[data-testid="stAppViewContainer"] div[data-testid="stMetricLabel"] {
    color: #475569 !important;
}

[data-testid="stAppViewContainer"] div[data-testid="stMetricValue"] {
    color: #172033 !important;
}

[data-testid="stAppViewContainer"] .metadata-label {
    color: #64748b !important;
}

[data-testid="stAppViewContainer"] .metadata-value {
    color: #172033 !important;
}



    .block-container {
        max-width: 1440px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    /* Main hero header */
    .hero-card {
        padding: 2rem;
        border: 1px solid #293957;
        border-radius: 20px;
        background: linear-gradient(
            120deg,
            #172554 0%,
            #111827 65%,
            #172033 100%
        );
        margin-bottom: 1.3rem;
    }

    .hero-topline {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        justify-content: space-between;
        gap: 0.7rem;
        margin-bottom: 1.1rem;
    }

    .hero-eyebrow {
        color: #93c5fd;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }

    .hero-badge {
        display: inline-block;
        color: #fde68a;
        background: rgba(245, 158, 11, 0.10);
        border: 1px solid rgba(245, 158, 11, 0.35);
        border-radius: 999px;
        padding: 0.35rem 0.7rem;
        font-size: 0.75rem;
        font-weight: 650;
    }

    .hero-title {
        color: #f8fafc;
        font-size: clamp(2.1rem, 4vw, 3.2rem);
        font-weight: 800;
        letter-spacing: -0.045em;
        line-height: 1.1;
        margin: 0;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1rem;
        line-height: 1.7;
        margin-top: 0.9rem;
        max-width: 780px;
    }

    .hero-footer {
        color: #93c5fd;
        font-size: 0.82rem;
        margin-top: 1.2rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid #293548;
        border-radius: 15px;
        padding: 1rem 1.2rem;
    }

    div[data-testid="stMetricLabel"] {
        color: #a8b5c9;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
        font-weight: 750;
    }

    /* Case metadata */
    .metadata-label {
        color: #94a3b8;
        font-size: 0.76rem;
        font-weight: 700;
        letter-spacing: 0.07em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .metadata-value {
        color: #f1f5f9;
        font-size: 1rem;
        font-weight: 550;
        overflow-wrap: anywhere;
    }

    .research-note {
        background: rgba(245, 158, 11, 0.07);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-left: 4px solid #f59e0b;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        color: #e5e7eb;
        font-size: 0.9rem;
        line-height: 1.65;
        margin: 0.6rem 0 1.4rem 0;
    }

    .result-card {
        background: linear-gradient(135deg, #14213d, #111827);
        border: 1px solid #334766;
        border-radius: 15px;
        padding: 1.25rem 1.4rem;
        margin: 0.5rem 0 1rem 0;
    }

    .result-label {
        color: #93c5fd;
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .result-value {
        color: #f8fafc;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.025em;
    }

    .result-description {
        color: #cbd5e1;
        font-size: 0.9rem;
        line-height: 1.6;
        margin-top: 0.45rem;
    }

    /* Inputs and buttons */
    div[data-baseweb="select"] > div {
        border-radius: 10px;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        border-radius: 10px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.8rem;
    }

    /* Responsive spacing */
    @media (max-width: 768px) {
        .block-container {
            padding: 1rem 1rem 2rem 1rem;
        }

        .hero-card {
            padding: 1.25rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------
# Load dataset and saved model
# --------------------------------------------------
@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)

    required = [
        "doc_id",
        "title",
        "court",
        "gender",
        "Facts",
        "conclusion",
        "bail_granted",
    ]

    missing = [col for col in required if col not in df.columns]

    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")

    df["text"] = (
        df["Facts"].fillna("").astype(str)
        + " "
        + df["conclusion"].fillna("").astype(str)
    )

    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)




# --------------------------------------------------
# Hero header
# --------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-topline">
            <span class="hero-eyebrow">
                LEGAL TECHNOLOGY · MACHINE LEARNING
            </span>
            <span class="hero-badge">
                Academic research prototype
            </span>
        </div>
        <h1 class="hero-title">Judgment Law AI</h1>
        <div class="hero-subtitle">
            Explore legal judgments, review case details, and examine
            bail outcome predictions generated by a saved machine-learning
            model.
        </div>
        <div class="hero-footer">
            Case exploration · Judgment review · Experimental prediction
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)



# --------------------------------------------------
# Initialize dataset and model
# --------------------------------------------------
if not DATA_PATH.exists():
    st.error(f"Dataset file not found: {DATA_PATH.name}")
    st.stop()

if not MODEL_PATH.exists():
    st.error(f"Saved model file not found: {MODEL_PATH.name}")
    st.stop()

try:
    df = load_dataset()
    model = load_model()
except Exception as exc:
    st.error(f"Could not initialize the application: {exc}")
    st.stop()

# --------------------------------------------------
# Workspace overview
# --------------------------------------------------
st.markdown("## Workspace overview")
st.caption("A quick overview of the data and model available in this session.")

json_files = (
    sorted(RESULTS_DIR.glob("*.json"))
    if RESULTS_DIR.exists()
    else []
)

c1, c2, c3 = st.columns(3, gap="medium")

with c1:
    st.metric(
        "Judgments in dataset",
        f"{len(df):,}",
        help="Number of records loaded from all_cases.csv.",
    )

with c2:
    st.metric(
        "Courts represented",
        f"{df['court'].nunique(dropna=True):,}",
        help="Number of distinct non-missing court values in the dataset.",
    )

with c3:
    st.metric(
        "Existing analyses",
        f"{len(json_files):,}",
        help="Number of JSON result files available in final_results.",
    )

st.caption(
    "The saved model was loaded successfully. This does not indicate "
    "that its predictions are accurate or independently validated."
)

st.divider()

# --------------------------------------------------
# Choose a case
# --------------------------------------------------
st.markdown("## 1 · Explore a judgment")
st.caption(
    "Select a case to inspect its metadata, review the judgment text, "
    "and run an experimental prediction."
)


def case_label(index):
    row = df.loc[index]
    return f"{row['doc_id']} — {row['title']}"


with st.container(border=True):
    st.markdown("### Select a case")

    selected_index = st.selectbox(
        "Available judgments",
        options=df.index.tolist(),
        format_func=case_label,
        key="selected_judgment",
        help="Choose a judgment by its case ID and title.",
    )

    case = df.loc[selected_index]

    st.caption(
        f"Selected record: {df.index.get_loc(selected_index) + 1:,} "
        f"of {len(df):,}"
    )

    st.divider()

    st.markdown("### Case information")

    left, right = st.columns(2, gap="large")

    with left:
        st.markdown(
            '<div class="metadata-label">Case title</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metadata-value">{case["title"]}</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            '<div class="metadata-label">Court</div>',
            unsafe_allow_html=True,
        )
        court_display = (
            "Not available"
            if pd.isna(case["court"]) or not str(case["court"]).strip()
            else str(case["court"])
        )
        st.markdown(
            f'<div class="metadata-value">{court_display}</div>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            '<div class="metadata-label">Case ID</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="metadata-value">{case["doc_id"]}</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        st.markdown(
            '<div class="metadata-label">Gender field in dataset</div>',
            unsafe_allow_html=True,
        )
        gender_display = (
            "Not available"
            if pd.isna(case["gender"]) or not str(case["gender"]).strip()
            else str(case["gender"])
        )
        st.markdown(
            f'<div class="metadata-value">{gender_display}</div>',
            unsafe_allow_html=True,
        )

# --------------------------------------------------
# Judgment input
# --------------------------------------------------
st.divider()
st.markdown("## 2 · Review judgment text")
st.caption(
    "Review the selected record's facts and conclusion. "
    "You can edit the text to explore how the model responds to changes."
)

with st.container(border=True):
    judgment_text = st.text_area(
        "Facts and conclusion",
        value=str(case["text"]),
        height=280,
        key=f"judgment_text_{selected_index}",
        help="The selected record's facts and conclusion are combined here.",
    )

    input_left, input_right = st.columns(2, gap="medium")

    with input_left:
        court = st.text_input(
            "Court feature",
            value="" if pd.isna(case["court"]) else str(case["court"]),
            key=f"court_{selected_index}",
            help="Court value passed to the saved model.",
        )

    with input_right:
        gender = st.text_input(
            "Gender feature",
            value="" if pd.isna(case["gender"]) else str(case["gender"]),
            key=f"gender_{selected_index}",
            help="Gender value passed to the saved model.",
        )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.divider()
st.markdown("## 3 · Experimental bail prediction")
st.caption(
    "Run the existing saved model on the judgment text and selected features."
)

st.warning(
    "Academic demonstration only. This output is not legal advice and "
    "must not be used to make decisions about an actual person's liberty."
)

if st.button("Run Bail Prediction", type="primary", use_container_width=True):
    if not judgment_text.strip():
        st.error("Please enter judgment text before predicting.")
        st.stop()

    features = pd.DataFrame(
        [
            {
                "text": judgment_text,
                "court": court,
                "gender": gender,
            }
        ]
    )

    try:
        prediction = model.predict(features)[0]
        label = "GRANTED" if int(prediction) == 1 else "REJECTED"

        st.session_state["prediction_result"] = {
            "label": label,
            "features": features,
            "case_id": str(case["doc_id"]),
            "case_title": str(case["title"]),
        }

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(features)[0]

            st.session_state["prediction_probabilities"] = {
                str(int(cls)): float(prob)
                for cls, prob in zip(model.classes_, probabilities)
            }
        else:
            st.session_state.pop("prediction_probabilities", None)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

result = st.session_state.get("prediction_result")

if result:
    st.markdown("### Latest prediction")

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-label">Predicted outcome</div>
            <div class="result-value">{result["label"]}</div>
            <div class="result-description">
                Case ID: {result["case_id"]}<br>
                This is the saved model's output for the submitted features,
                not a legal recommendation.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    probabilities = st.session_state.get("prediction_probabilities")

    if probabilities:
        p_rejected = probabilities.get("0", 0.0)
        p_granted = probabilities.get("1", 0.0)

        a, b = st.columns(2, gap="medium")

        with a:
            st.metric(
                "REJECTED · Model score",
                f"{p_rejected:.2%}",
            )

        with b:
            st.metric(
                "GRANTED · Model score",
                f"{p_granted:.2%}",
            )

        st.caption(
            "These values are the model's predict_proba outputs. They are "
            "not necessarily calibrated probabilities of a real legal "
            "outcome, and they should not be interpreted as confidence "
            "that a court will reach a particular decision."
        )

# --------------------------------------------------
# Existing processed results
# --------------------------------------------------
st.divider()
st.markdown("## 4 · Existing processed judgments")
st.caption(
    "Browse the JSON analysis files already available in your project."
)

if RESULTS_DIR.exists():
    json_files = sorted(RESULTS_DIR.glob("*.json"))

    st.metric("JSON result files available", len(json_files))

    if json_files:
        with st.container(border=True):
            selected_result = st.selectbox(
                "Choose an existing analysis",
                options=json_files,
                format_func=lambda p: p.name,
                key="selected_existing_result",
            )

            if st.button("Load Existing Result", type="secondary"):
                try:
                    with selected_result.open("r", encoding="utf-8") as file:
                        result_data = json.load(file)

                    st.session_state["loaded_json_result"] = result_data
                    st.session_state["loaded_json_filename"] = selected_result.name

                except Exception as exc:
                    st.error(f"Could not read result: {exc}")

        if "loaded_json_result" in st.session_state:
            st.markdown(
                f"### Analysis: {st.session_state['loaded_json_filename']}"
            )
            st.json(st.session_state["loaded_json_result"])
else:
    st.info("The final_results directory has not been created yet.")

# --------------------------------------------------
# Footer
# --------------------------------------------------
st.divider()

st.caption(
    "Judgment Law AI · Academic research prototype · "
    "Predictions require independent validation."
)
