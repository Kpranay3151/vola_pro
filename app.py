"""Streamlit UI for the Transaction RAG Pipeline."""

import os
import sys

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(override=True)  # always pick up latest .env values on restart

import streamlit as st

from src.utils import load_transaction_data
from src.pipeline import TransactionRAGPipeline
from ui.session import init_session_state
from ui.sidebar import render_sidebar
from ui.chat import render_chat

st.set_page_config(
    page_title="Vola Pro - Financial Analysis",
    page_icon="\U0001f4c8",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_pipeline():
    """Load CSV data and create the pipeline (cached across reruns)."""
    csv_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "assessment_transaction_data.xlsx - Transactions.csv",
    )
    df = load_transaction_data(csv_path)
    pipeline = TransactionRAGPipeline(df=df)
    return pipeline, df


def main():
    init_session_state()

    try:
        pipeline, df = get_pipeline()
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}")
        st.stop()
    except ValueError as e:
        st.error(f"Configuration error: {e}")
        st.info("Set GEMINI_API_KEY or OPENROUTER_API_KEY in your .env file.")
        st.stop()

    render_sidebar(pipeline, df)
    render_chat(pipeline)


if __name__ == "__main__":
    main()
