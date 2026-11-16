
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
st.write(
    "Analyse search queries, find popular keywords, "
    "and visualise their frequencies."
)


def analyse_keywords(texts):
    words = []

    stop_words = {
        "the", "is", "a", "an", "and", "or", "to",
        "of", "in", "for", "on", "with", "how", "what"
    }

    for text in texts:
        text = str(text).lower()
        found_words = re.findall(r"\b[a-z0-9]+\b", text)
        words.extend(
            word for word in found_words
            if word not in stop_words
        )

    return Counter(words)


uploaded_file = st.file_uploader(
    "Upload search queries CSV",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.subheader("Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        text_columns = df.select_dtypes(
            include=["object", "string"]
        ).columns.tolist()

        if not text_columns:
            st.error("The CSV file must contain a text column.")
            st.stop()

        column = st.selectbox(
            "Select the search query column",
            text_columns
        )

        frequencies = analyse_keywords(df[column].dropna())

        if not frequencies:
            st.warning("No keywords found in this dataset.")
            st.stop()

        results = pd.DataFrame(
            frequencies.most_common(),
            columns=["Keyword", "Frequency"]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Words Analysed", sum(frequencies.values()))
        col2.metric("Unique Keywords", len(frequencies))
        col3.metric("Search Queries", int(df[column].count()))

        st.subheader("Keyword Analysis")

        search_keyword = st.text_input(
            "Search for a keyword",
            placeholder="Example: python"
        )

        filtered_results = results[
            results["Keyword"].str.contains(
                search_keyword,
                case=False,
                regex=False
            )
        ]

        st.subheader("Keyword Frequency Table")
        st.dataframe(
            filtered_results,
            use_container_width=True,
            hide_index=True
        )

        top_n = st.selectbox(
            "Number of top keywords in chart",
            [5, 10, 15, 20],
            index=1
        )

        chart_data = filtered_results.head(top_n)

        st.subheader(f"Top {top_n} Keywords")

        if not chart_data.empty:
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.bar(
                chart_data["Keyword"],
                chart_data["Frequency"]
            )
            ax.set_xlabel("Keyword")
            ax.set_ylabel("Frequency")
            ax.set_title("Keyword Frequency Analysis")
            ax.tick_params(axis="x", labelrotation=45)

            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info("No keywords match your search.")

        csv_data = filtered_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "Download Filtered Results",
            data=csv_data,
            file_name="keyword_frequency_results.csv",
            mime="text/csv"
        )

    except Exception as error:
        st.error(f"Could not process the CSV file: {error}")

else:
    st.info("Upload a CSV file to begin keyword analysis.")
