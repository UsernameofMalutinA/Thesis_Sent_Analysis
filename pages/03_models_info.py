# pages/03_models_info.py
import streamlit as st
from pathlib import Path

st.title("Информация о моделях")

st.markdown("""
### Sentiment Analysis  
**rubert-base-cased-sentiment** – Модель на основе RuBERT, обученная классифицировать текст как «Положительное», «Нейтральное» или «Негативное». Она по умолчанию используется для простого анализа настроений.

### Topic Modeling  
**BERTopic** – современная алгоритмическая модель, 
            использующая эмбеддинги и кластеризацию для извлечения ключевых слов-тем.
             Выдает списки из 5–10 ключевых слов.
            
**LDA (sklearn)** – Классический метод латентного размещения Дирихле. На больших объемах данных может работать быстрее, чем BERTopic.

### ABSA  
**GPT-4.1 mini** – облачная модель OpenAI (версия 4.1 mini) для чат-анализа. 
            Делает аспектно-ориентированный анализ: по каждой заданной теме возвращает «Положительное», «Нейтральное» или «Негативное». 

**DeepSeek R1** – альтернативный облачный сервис для чат-анализа отзывов, модель R1 на платформе DeepSeek. Схож по возможностям с GPT, можно задавать инструкции и примеры, но работает дольше. 
            При этом стоимость выгодно отличается от GPT.
            
### Полезные файлы
""")

project_root = Path(__file__).parent.parent
docs_dir = project_root / "docs"

html_files = {
    "Карта субъектов Российской Федерации по среднему рейтингу в отзывах": docs_dir / "region_rating_map_grouped.html",
    "Гексагоновая карта Нижегородской области": docs_dir / "hexagon_map_NN.html",
    "Гексагоновая карта Свердловской области": docs_dir / "hexagon_map_EKB.html"
}

for label, path in html_files.items():
    if path.exists():
        with open(path, "rb") as f:
            st.download_button(
                label=f"Скачать: {label}",
                data=f.read(),
                file_name=path.name,
                mime="text/html"
            )
    else:
        st.warning(f"Не найден файл: {path.name} (ожидался в {docs_dir})")
