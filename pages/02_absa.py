import streamlit as st
import pandas as pd
import io
from src.back import (
    analyze_topic_sentiment_gpt4,
    analyze_topic_sentiment_deepseek,
    generate_apology_response
)

st.title("Аспектно-ориентированный анализ настроения (ABSA)")

model_choice = st.radio(
    "Выберите модель ABSA:",
    ["GPT-4.1 mini", "DeepSeek R1"]
)

api_key = ""
if model_choice in ("GPT-4.1 mini", "DeepSeek R1"):
    api_key = st.text_input("API Key", type="password")

aspects = st.text_input(
    "Введите до 5 аспектов (через запятую):",
    "Еда, Сервис, Интерьер"
)
aspect_list = [a.strip() for a in aspects.split(",") if a.strip()][:5]

st.subheader("Анализ одного отзыва")
text_absa = st.text_area("Введите текст:")
if st.button("ABSA анализ текста"):
    if not text_absa:
        st.warning("Введите текст.")
    else:
        if model_choice == "GPT-4.1 mini" and api_key:
            res = analyze_topic_sentiment_gpt4(text_absa, aspect_list, api_key)
        elif model_choice == "DeepSeek R1" and api_key:
            res = analyze_topic_sentiment_deepseek(text_absa, aspect_list, api_key)
        else:
            res = {}
        for asp, sentiment in res.items():
            st.write(f"**{asp}:** {sentiment}")

st.subheader("Пакетный ABSA по файлу")
file_absa = st.file_uploader("CSV или Excel", type=["csv", "xlsx"])
if file_absa:
    df = (
        pd.read_csv(file_absa)
        if file_absa.name.endswith(".csv")
        else pd.read_excel(file_absa)
    )

    def calc_absa(row_text: str) -> dict[str, str]:
        if model_choice == "GPT-4.1 mini" and api_key:
            return analyze_topic_sentiment_gpt4(row_text, aspect_list, api_key)
        elif model_choice == "DeepSeek R1" and api_key:
            return analyze_topic_sentiment_deepseek(row_text, aspect_list, api_key)
        return {}

    df_absa = pd.DataFrame(
        df.iloc[:, 0].astype(str).apply(calc_absa).tolist(),
        index=df.index
    )
    df = pd.concat([df, df_absa], axis=1)

    st.dataframe(df.head())

    buf = io.BytesIO()
    if file_absa.name.endswith(".csv"):
        df.to_csv(buf, index=False)
        mime, fname = "text/csv", "absa.csv"
    else:
        df.to_excel(buf, index=False)
        mime, fname = (
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "absa.xlsx",
        )
    buf.seek(0)
    st.download_button("Скачать", data=buf, file_name=fname, mime=mime)

st.subheader("Сгенерировать ответ для отзыва")
apology_text = st.text_area("Введите отзыв:", key="apology")
if st.button("Сгенерировать ответ"):
    if not apology_text:
        st.warning("Введите текст отзыва.")
    elif not api_key:
        st.warning("Введите API Key.")
    else:
        provider = "deepseek" if model_choice == "DeepSeek R1" else "gpt4"
        apology = generate_apology_response(apology_text, api_key, provider)
        st.markdown(f"**Ваш ответ:**\n\n{apology}")



st.subheader("Ответы по файлу")
file_apology = st.file_uploader("CSV или Excel (тексты отзывов в первом столбце)", type=["csv", "xlsx"], key="apologyfile")

if file_apology:
    df_apology = (
        pd.read_csv(file_apology)
        if file_apology.name.endswith(".csv")
        else pd.read_excel(file_apology)
    )

    if st.button("Сгенерировать ответы для всех отзывов"):
        if not api_key:
            st.warning("Введите API Key.")
        else:
            provider = "deepseek" if model_choice == "DeepSeek R1" else "gpt4"
            # Генерируем ответы для всех отзывов из первого столбца
            apologies = []
            for text in df_apology.iloc[:, 0].astype(str):
                apology = generate_apology_response(text, api_key, provider)
                apologies.append(apology)
            df_apology["Извинение"] = apologies
            st.dataframe(df_apology.head())
            buf = io.BytesIO()
            if file_apology.name.endswith(".csv"):
                df_apology.to_csv(buf, index=False)
                mime, fname = "text/csv", "apologies.csv"
            else:
                df_apology.to_excel(buf, index=False)
                mime, fname = (
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    "apologies.xlsx",
                )
            buf.seek(0)
            st.download_button("Скачать ответы", data=buf, file_name=fname, mime=mime)

