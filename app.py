
import re
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="Keyword Frequency Analysis",
    page_icon="🔎",
    layout="wide"
)

st.title("Search Engine Keyword Frequency Analysis")
st.write("Upload a CSV file to analyse keyword frequency.")

def analyse_keywords(texts):
    words = []

    for text in texts:
        text = str(text).lower()
        words.extend(re.findall(r"\b[a-z0-9]+\b", text))

    stop_words = {
        "the", "is", "a", "an", "and", "or", "to",
        "of", "in", "for", "on", "with", "how", "what"
    }

    words = [word for word in words if word not in stop_words]
    return Counter(words)

uploaded_file = st.file_uploader(
    "Upload search queries CSV",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    if text_columns:
        column = st.selectbox(
            "Choose the search query column",
            text_columns
        )

        frequencies = analyse_keywords(df[column].dropna())

        if frequencies:
            results = pd.DataFrame(
                frequencies.most_common(),
                columns=["Keyword", "Frequency"]
            )

            st.subheader("Keyword Frequency Results")
            st.dataframe(results, use_container_width=True)

            st.subheader("Top 10 Keywords")
            top_words = results.head(10)

            fig, ax = plt.subplots()
            ax.bar(top_words["Keyword"], top_words["Frequency"])
            ax.set_xlabel("Keyword")
            ax.set_ylabel("Frequency")
            ax.set_title("Most Frequent Keywords")
            plt.xticks(rotation=45)
            plt.tight_layout()

            st.pyplot(fig)

            st.download_button(
                "Download Results CSV",
                results.to_csv(index=False),
                "keyword_frequency_results.csv",
                "text/csv"
            )
        else:
            st.warning("No keywords found.")
    else:
        st.error("The CSV file must contain a text column.")
