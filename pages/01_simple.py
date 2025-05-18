import streamlit as st
import pandas as pd
import io
from src.back import preprocess_text, analyze_sentiment, extract_topics



st.title("Простой анализ настроения и тем")
topic_choice = st.selectbox("Topic modeling:", ["None", "BERTopic", "LDA"])

st.subheader("Анализ одного отзыва")
text = st.text_area("Введите текст:")
if st.button("Анализировать текст"):
    if not text:
        st.warning("Введите текст для анализа.")
    else:
        proc = preprocess_text(text)
        sent = analyze_sentiment(proc)
        st.write(f"**Настроение:** {sent}")
        if topic_choice != "None":
            kws = extract_topics(text, topic_choice)
            st.write("**Темы:**", ", ".join(kws) if kws else "—")

st.subheader("Пакетная обработка файла")
file = st.file_uploader("CSV или Excel", type=["csv", "xlsx"])
if file:
    df = pd.read_csv(file) if file.name.endswith('.csv') else pd.read_excel(file)
    df['Sentiment'] = df.iloc[:, 0].astype(str).apply(lambda t: analyze_sentiment(preprocess_text(t)))
    if topic_choice != "None":
        df['Topics'] = df.iloc[:, 0].astype(str).apply(lambda t: ", ".join(extract_topics(t, topic_choice)))
    st.dataframe(df.head())
    buf = io.BytesIO()
    if file.name.endswith('.csv'):
        df.to_csv(buf, index=False)
        mime, fname = "text/csv", "result.csv"
    else:
        df.to_excel(buf, index=False)
        mime, fname = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "result.xlsx"
    buf.seek(0)
    st.download_button("Скачать", data=buf, file_name=fname, mime=mime)
