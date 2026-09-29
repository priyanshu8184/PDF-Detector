# UI Helpers: CSS load karna aur session state set karna
import os
import streamlit as st


def load_custom_css(path: str = "static/style.css") -> None:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_session_state() -> None:
    for k, v in {"raw_text": "", "chunks": [], "index": None, "history": []}.items():
        if k not in st.session_state:
            st.session_state[k] = v
