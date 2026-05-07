import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# Download NLTK resources
# ----------------------------
nltk.download('punkt')
nltk.download('stopwords')

# ----------------------------
# Streamlit Page Config
# ----------------------------
st.set_page_config(
    page_title="Legal NLP Analyzer",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Legal NLP Analysis App")
st.markdown("### NLP + Machine Learning + Visualization")

# ----------------------------
# Load Dataset
# ----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("legal_nlp_dataset_3000_rows.csv")
    return df

df = load_data()

# ----------------------------
# Dataset Overview
# ----------------------------
st.subheader("📂 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Rows", df.shape[0])
col2.metric("Columns", df.shape[1])
col3.metric("Missing Values", df.isnull().sum().sum())

st.dataframe(df.head())

# ----------------------------
# NLP Preprocessing
# ----------------------------
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    return " ".join(tokens)

with st.spinner("Preprocessing Text..."):
    df['clean_text'] = df['document_text'].apply(preprocess)

st.success("Text preprocessing completed!")

# ----------------------------
# TF-IDF
# ----------------------------
st.subheader("📊 TF-IDF Feature Extraction")

tfidf = TfidfVectorizer(max_features=200)
X_tfidf = tfidf.fit_transform(df['clean_text']).toarray()

feature_names = tfidf.get_feature_names_out()

st.write("### Top Features")
st.write(feature_names[:20])

# ----------------------------
# Word Frequency
# ----------------------------
st.subheader("📈 Top Frequent Words")

all_words = " ".join(df['clean_text']).split()
word_freq = Counter(all_words)

top_words = word_freq.most_common(10)

words = [w[0] for w in top_words]
counts = [w[1] for w in top_words]

fig, ax = plt.subplots(figsize=(10,5))
ax.bar(words, counts)
plt.xticks(rotation=45)

st.pyplot(fig)

# ----------------------------
# Bigram Analysis
# ----------------------------
st.subheader("🔍 Bigram Analysis")

bigram = CountVectorizer(ngram_range=(2,2), max_features=10)
X_bigram = bigram.fit_transform(df['clean_text'])

bigrams = bigram.get_feature_names_out()
bigram_counts = X_bigram.sum(axis=0).A1

bigram_df = pd.DataFrame({
    "Bigram": bigrams,
    "Count": bigram_counts
})

st.dataframe(bigram_df)

# ----------------------------
# Cosine Similarity
# ----------------------------
st.subheader("🧠 Cosine Similarity")

similarity_matrix = cosine_similarity(X_tfidf)

st.write(f"Similarity between Document 1 and 2: {similarity_matrix[0][1]:.4f}")
st.write(f"Average Similarity: {np.mean(similarity_matrix):.4f}")

# ----------------------------
# ML Model
# ----------------------------
st.subheader("🤖 Machine Learning Model")

X = X_tfidf
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# ----------------------------
# Metrics
# ----------------------------
col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

# ----------------------------
# Anomaly Detection
# ----------------------------
st.subheader("🚨 Anomaly Detection")

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

df['anomaly'] = iso.fit_predict(X_tfidf)

normal_docs = (df['anomaly'] == 1).sum()
anomalies = (df['anomaly'] == -1).sum()

st.write(f"Normal Documents: {normal_docs}")
st.write(f"Anomalies Detected: {anomalies}")

# ----------------------------
# User Prediction
# ----------------------------
st.subheader("✍️ Predict Risk Label")

user_input = st.text_area("Enter Legal Text")

if st.button("Predict"):

    cleaned = preprocess(user_input)

    vector = tfidf.transform([cleaned]).toarray()

    prediction = model.predict(vector)[0]

    st.success(f"Predicted Risk Label: {prediction}")

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit + NLP + Machine Learning")
