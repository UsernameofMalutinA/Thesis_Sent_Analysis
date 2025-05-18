# app.py
import streamlit as st

st.set_page_config(page_title="Sentiment Multi-Page App", layout="wide")
st.title("Добро пожаловать в анализатор отзывов")
st.markdown(
    "Выберите нужную страницу в меню слева:\n\n"
    "- **01_simple** — простой анализ настроения и тем\n"
    "- **02_absa** — abstract-based sentiment analysis\n"
    "- **03_models_info** — информация о моделях"
)