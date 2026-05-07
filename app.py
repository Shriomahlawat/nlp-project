import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Legal NLP Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# ============================================
# NLTK DOWNLOAD FIX
# ============================================

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# ============================================
# TITLE
# ============================================

st.title("⚖️ Legal NLP Analysis App")
st.markdown("### NLP + Machine Learning + Data Visualization")

# ============================================
# LOAD DATASET
# ============================================

@st.cache_data
def load_data():
    df = pd.read_csv("legal_nlp_dataset_3000_rows.csv")
    return df

df = load_data()

# ============================================
# DATASET OVERVIEW
# ============================================

st.subheader("📂 Dataset Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Rows", df.shape[0])
col2.metric("Total Columns", df.shape[1])
col3.metric("Missing Values", int(df.isnull().sum().sum()))

st.dataframe(df.head())

# ============================================
# NLP PREPROCESSING
# ============================================

stop_words = set(stopwords.words('english'))

def preprocess(text):

    # Convert to lowercase
    text = str(text).lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Simple tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]

    return " ".join(tokens)

with st.spinner("Preprocessing text data..."):
    df['clean_text'] = df['document_text'].apply(preprocess)

st.success("✅ Text preprocessing completed successfully!")

# ============================================
# TF-IDF FEATURE EXTRACTION
# ============================================

st.subheader("📊 TF-IDF Feature Extraction")

tfidf = TfidfVectorizer(max_features=200)

X_tfidf = tfidf.fit_transform(df['clean_text']).toarray()

feature_names = tfidf.get_feature_names_out()

st.write("### Top TF-IDF Features")
st.write(feature_names[:20])

# ============================================
# WORD FREQUENCY ANALYSIS
# ============================================

st.subheader("📈 Most Frequent Words")

all_words = " ".join(df['clean_text']).split()

word_freq = Counter(all_words)

top_words = word_freq.most_common(10)

words = [w[0] for w in top_words]
counts = [w[1] for w in top_words]

fig, ax = plt.subplots(figsize=(10, 5))

ax.bar(words, counts)

plt.xticks(rotation=45)

st.pyplot(fig)

# ============================================
# BIGRAM ANALYSIS
# ============================================

st.subheader("🔍 Bigram Analysis")

bigram = CountVectorizer(
    ngram_range=(2, 2),
    max_features=10
)

X_bigram = bigram.fit_transform(df['clean_text'])

bigrams = bigram.get_feature_names_out()

bigram_counts = X_bigram.sum(axis=0).A1

bigram_df = pd.DataFrame({
    "Bigram": bigrams,
    "Count": bigram_counts
})

st.dataframe(bigram_df)

# ============================================
# COSINE SIMILARITY
# ============================================

st.subheader("🧠 Cosine Similarity Analysis")

similarity_matrix = cosine_similarity(X_tfidf)

st.write(
    f"Similarity Between Document 1 & 2: "
    f"{similarity_matrix[0][1]:.4f}"
)

st.write(
    f"Average Similarity Score: "
    f"{np.mean(similarity_matrix):.4f}"
)

# ============================================
# DOCUMENT LENGTH STATISTICS
# ============================================

st.subheader("📏 Document Statistics")

df['doc_length'] = df['clean_text'].apply(
    lambda x: len(x.split())
)

col1, col2, col3 = st.columns(3)

col1.metric(
    "Average Length",
    f"{df['doc_length'].mean():.2f}"
)

col2.metric(
    "Maximum Length",
    int(df['doc_length'].max())
)

col3.metric(
    "Minimum Length",
    int(df['doc_length'].min())
)

# ============================================
# MACHINE LEARNING MODEL
# ============================================

st.subheader("🤖 Machine Learning Model")

X = X_tfidf
y = df['risk_label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
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

precision = precision_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average='weighted',
    zero_division=0
)

# ============================================
# MODEL METRICS
# ============================================

st.subheader("📊 Model Performance")

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

# ============================================
# ANOMALY DETECTION
# ============================================

st.subheader("🚨 Anomaly Detection")

iso = IsolationForest(
    contamination=0.05,
    random_state=42
)

df['anomaly'] = iso.fit_predict(X_tfidf)

normal_docs = (df['anomaly'] == 1).sum()
anomalies = (df['anomaly'] == -1).sum()

col1, col2 = st.columns(2)

col1.metric("Normal Documents", int(normal_docs))
col2.metric("Anomalies", int(anomalies))

# ============================================
# USER PREDICTION SECTION
# ============================================

st.subheader("✍️ Predict Legal Risk")

user_input = st.text_area(
    "Enter Legal Document Text"
)

if st.button("Predict Risk Label"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        cleaned_text = preprocess(user_input)

        vector = tfidf.transform([cleaned_text]).toarray()

        prediction = model.predict(vector)[0]

        st.success(
            f"✅ Predicted Risk Label: {prediction}"
        )

# ============================================
# RISK LABEL DISTRIBUTION
# ============================================

st.subheader("📌 Risk Label Distribution")

if 'risk_label' in df.columns:

    risk_counts = df['risk_label'].value_counts()

    fig2, ax2 = plt.subplots(figsize=(8, 5))

    ax2.bar(
        risk_counts.index.astype(str),
        risk_counts.values
    )

    plt.xticks(rotation=30)

    st.pyplot(fig2)

# ============================================
# FOOTER
# ============================================

st.markdown("---")

st.markdown(
    "### ✅ Streamlit NLP Project Successfully Running"
)

st.markdown(
    "Made with ❤️ using NLP + Machine Learning + Streamlit"
)
