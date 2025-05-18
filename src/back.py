import os
import re
import nltk
from nltk.corpus import stopwords
from pymorphy3 import MorphAnalyzer
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from openai import OpenAI
from bertopic import BERTopic
import joblib
import deepseek
import requests
from sentence_transformers import SentenceTransformer


try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

morph = MorphAnalyzer()
russian_stopwords = set(stopwords.words('russian'))

device = 0 if torch.cuda.is_available() else -1


sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="blanchefort/rubert-base-cased-sentiment",
    tokenizer="blanchefort/rubert-base-cased-sentiment",
    device=device
)


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^а-яё\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return " ".join(
        morph.parse(tok)[0].normal_form
        for tok in text.split()
        if tok not in russian_stopwords and len(tok) > 2
    )
    

def analyze_sentiment(text: str) -> str:
    if not text or len(text) < 3:
        return "Нейтральное"
    label = sentiment_analyzer(text)[0]["label"]
    return {
        "POSITIVE": "Положительное",
        "NEUTRAL":  "Нейтральное",
        "NEGATIVE": "Негативное"
    }.get(label, "Нейтральное")



# ──BERTopic ──────────────────────────────────────────


bert_model = None

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir   = os.path.join(project_root, "bertopic_final_model")

if os.path.isdir(model_dir):
    try:
        embedding_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        bert_model = BERTopic.load(model_dir, embedding_model=embedding_model)
        print(f"Loaded BERTopic model from {model_dir}")
    except Exception as e:
        print(f"Failed to load BERTopic model: {e}")
else:
    print(f"⚠️ Model directory not found: {model_dir}")


def deduplicate_topic_words(topic_words):
    filtered = []
    for w in topic_words:
        if not any(w != other and w in other.split() for other in topic_words):
            filtered.append(w)
    return filtered


def extract_topics_bertopic(text: str, top_n: int = 5) -> list[str]:
    if bert_model is None:
        return []
    topics, _ = bert_model.transform([text])
    kws = bert_model.get_topic(topics[0])
    top_words = [w for w, _ in kws][:max(10, top_n)]
    cleaned = deduplicate_topic_words(top_words)
    return cleaned[:top_n]





# ── TOPIC MODELLING ───────────────────────────────


PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
VECT_PATH  = os.path.join(PROJECT_ROOT, "sklearn_lda_vectorizer_genvers.joblib")
MODEL_PATH = os.path.join(PROJECT_ROOT, "sklearn_lda_model_genvers.joblib")

try:
    vectorizer = joblib.load(VECT_PATH)
    lda_model  = joblib.load(MODEL_PATH)
    print(f"✅ Loaded sklearn LDA and vectorizer from {VECT_PATH}, {MODEL_PATH}")
except Exception as e:
    vectorizer = None
    lda_model  = None
    print(f"⚠️ Could not load sklearn LDA/vectorizer: {e}")


def extract_topics_lda(text: str, top_n: int = 5) -> list[str]:
    if vectorizer is None or lda_model is None:
        return []
    bow = vectorizer.transform([text])
    topic_probs = lda_model.transform(bow)[0]
    if topic_probs.size == 0:
        return []
    top_topic = int(topic_probs.argmax())
    comp = lda_model.components_[top_topic]
    feat_names = vectorizer.get_feature_names_out()
    top_idxs = comp.argsort()[-top_n:][::-1]
    return [feat_names[i] for i in top_idxs]

def extract_topics(text: str, model_type: str, top_n: int = 5) -> list[str]:
    if model_type == "BERTopic":
        return extract_topics_bertopic(text, top_n)
    if model_type == "LDA":
        return extract_topics_lda(text, top_n)
    return []

# ── ABSA FUNCTIONS ─────────────────────────────


def analyze_topic_sentiment_flan():
    return []


_PROMPT_INTRO = (
    "Ты — эксперт по анализу тональности текстов русскоязычных отзывов о заведениях общественного питания. Ты работаешь на владельцев ресторанов и кафе, помогаешь им анализировать отзывы."
    "Твоя задача — по списку тем определить для каждой тональность: "
    "Положительное, Нейтральное или Негативное."
    "Формат вывода должен быть [Тема1]: [Тональность], для каждой данной темы. Объяснение в скобках писать не надо"
    "В случае если по какому-то топику нет информации в тексте, то просто заполняй значением 'Нейтральное'"
    "От того как ты определишь тональность зависит улучшат ли владельцы бизнеса условия для их клиентов"
)

_PROMPT_EXAMPLES = """

Отзыв: 'Еда была ужасная, но атмосфера отличная.'
Темы: food, ambiance
Ответ:
Еда: Негативное (еда была ужасная)
атмосфера: Положительное (атмосфера отличная)

Отзыв: 'Очень медленное обслуживание, но вкусно.'
Темы: сервис, еда
Ответ:
сервис: Негативное (обслуживание медленное)
Еда: Положительное (вкусно)

Отзыв: 'Прекрасный интерьер, хорошее обслуживание, быстро, ненавязчиво. Прекрасный интерьер, хорошее обслуживание, быстро, ненавязчиво. Кухня понравилась.'
Темы: сервис, Еда, интерьер, атмосфера.
Ответ:
сервис: Положительное (хорошее обслуживание, быстро, ненавязчиво)
Еда: Положительное (кухня понравилась)
интерьер: Положительно (прекрасный интерьер)
атмосфера: Нейтрально (отзыв не выделяет атмосферу явно)

Отзыв: 'В ресторане было очень шумно, но официанты старались, а еда понравилась.'
Темы: атмосфера, сервис, еда
Ответ:
атмосфера: Негативно (было очень шумно)
сервис: Положительно (официанты старались)
Еда: Положительно (еда понравилась)

Отзыв: 'Интерьер современный, но блюда принесли холодными. Ждали заказ очень долго, ужас.' 
Темы: интерьер, еда, сервис
Ответ:
интерьер: Положительно (интерьер современный)
Еда: Негативно (блюда холодные)
сервис: Негативно (ждали долго, ужас)

Отзыв: 'Очень уютная атмосфера, вкусные десерты, но персонал невнимательный.'
Темы: атмосфера, еда, сервис
Ответ:
атмосфера: Положительно (очень уютная атмосфера)
Еда: Положительно (вкусные десерты)
сервис: Негативно (персонал невнимательный)

Отзыв: 'Обслуживание быстрое, интерьер обычный, блюда свежие и недорогие.'
Темы: сервис, интерьер, еда
Ответ:
сервис: Положительно (обслуживание быстрое)
интерьер: Нейтрально (интерьер обычный)
Еда: Положительно (блюда свежие и недорогие)

Отзыв: 'Официанты были грубыми, но салат был свежий и вкусный.'
Темы: сервис, еда
Ответ:
сервис: Негативно (официанты грубые)
Еда: Положительно (салат свежий и вкусный)

Отзыв: 'Интерьер скучный, зато обслуживание вежливое, кухня ничем не выделяется.'
Темы: интерьер, сервис, еда
Ответ:
интерьер: Негативно (интерьер скучный)
сервис: Положительно (обслуживание вежливое)
Еда: Нейтрально (кухня ничем не выделяется)

Отзыв: 'В кафе уютно, официанты очень доброжелательные, но паста была пересолена и напитки дорогие.'
Темы: атмосфера, сервис, еда, напитки
Ответ:
Еда: Негативно (паста была пересолена)
напитки: Негативно (напитки дорогие)
атмосфера: Положительно (в кафе уютно)
сервис: Положительно (официанты доброжелательные)

"""


def analyze_topic_sentiment_gpt4(
    text: str,
    topics: list[str],
    api_key: str
) -> dict[str, str]:

    client = OpenAI(api_key=api_key)

    topics_str = ", ".join(topics)
    user_prompt = (
        f"Отзыв: '{text}'\n"
        f"Темы: {topics_str}\n"
        "Ответ в формате 'тема: настроение' для каждой темы."
    )

    messages = [
        {"role": "system", "content": _PROMPT_INTRO},
        {"role": "system", "content": _PROMPT_EXAMPLES.strip()},
        {"role": "user",   "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=messages,
        temperature=0
    )

    content = response.choices[0].message.content.strip()
    result = {}
    for line in content.splitlines():
        if ":" in line:
            topic, sentiment = line.split(":", 1)
            result[topic.strip()] = sentiment.strip()
    return result


def analyze_topic_sentiment_deepseek(
    text: str,
    topics: list[str],
    api_key: str
) -> dict[str, str]:
    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

    topics_str = ", ".join(topics)
    user_prompt = (
        f"Отзыв: '{text}'\n"
        f"Темы: {topics_str}\n"
        "Определите настроение для каждой темы в формате 'тема: настроение'."
    )

    messages = [
        {"role": "system", "content": _PROMPT_INTRO},
        {"role": "system", "content": _PROMPT_EXAMPLES.strip()},
        {"role": "user",   "content": user_prompt}
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        temperature=0,
        stream=False
    )

    content = response.choices[0].message.content.strip()
    result = {}
    for line in content.splitlines():
        if ":" in line:
            topic, sentiment = line.split(":", 1)
            result[topic.strip()] = sentiment.strip()
    return result