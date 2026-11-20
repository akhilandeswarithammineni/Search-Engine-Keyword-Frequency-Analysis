
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from collections import Counter
from wordcloud import WordCloud


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Keyword Frequency Analysis",
    page_icon="🔍",
    layout="wide"
)


# TITLE
st.title("🔍 Search Engine Keyword Frequency Analysis")
st.write(
    "Analyze search queries to identify frequently "
    "used keywords and understand search patterns."
)

st.divider()


# STOP WORDS
STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "and", "or", "but", "to", "of", "in", "on",
    "for", "with", "by", "at", "from", "how",
    "what", "which", "who", "where", "when",
    "why", "can", "could", "should", "would",
    "i", "me", "my", "we", "our", "you", "your",
    "it", "this", "that", "be", "do", "does"
}


# CSV UPLOAD
uploaded_file = st.file_uploader(
    "Upload a CSV file containing search queries",
    type=["csv"]
)


if uploaded_file is not None:

    try:
        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("The uploaded CSV file is empty.")

        else:

            # DATASET PREVIEW
            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            # SELECT COLUMN
            text_column = st.selectbox(
                "Select the column containing search queries",
                options=df.columns
            )

            queries = df[text_column].dropna().astype(str)

            # KEYWORD ANALYSIS
            frequencies = Counter()
            all_phrases = Counter()

            for query in queries:

                words = re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    query.lower()
                )

                clean_words = [
                    word for word in words
                    if word not in STOP_WORDS
                ]

                frequencies.update(clean_words)

                # TWO-WORD PHRASE ANALYSIS
                phrases = [
                    f"{clean_words[i]} {clean_words[i+1]}"
                    for i in range(len(clean_words) - 1)
                ]

                all_phrases.update(phrases)

            if not frequencies:
                st.warning("No keywords found.")

            else:

                total_words = sum(frequencies.values())

                # RESULTS TABLE
                results = pd.DataFrame(
                    frequencies.most_common(),
                    columns=["Keyword", "Frequency"]
                )

                results.insert(
                    0, "Rank", range(1, len(results) + 1)
                )

                results["Percentage"] = (
                    results["Frequency"] / total_words * 100
                ).round(2)

                # ANALYSIS SUMMARY
                st.subheader("Analysis Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Words", total_words)
                col2.metric("Unique Keywords", len(frequencies))
                col3.metric("Search Queries", len(queries))

                st.divider()

                # KEYWORD SEARCH
                st.subheader("Keyword Frequency Results")

                search_keyword = st.text_input(
                    "Search for a keyword",
                    placeholder="Enter a keyword..."
                )

                filtered_results = results[
                    results["Keyword"].str.contains(
                        search_keyword,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

                top_n = st.selectbox(
                    "Select number of top keywords",
                    [5, 10, 15, 20],
                    index=1
                )

                display_results = filtered_results.head(top_n)

                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True
                )

                # BAR CHART
                st.subheader("📊 Keyword Frequency Chart")

                if not display_results.empty:

                    chart_data = display_results.sort_values(
                        "Frequency",
                        ascending=True
                    )

                    fig, ax = plt.subplots(figsize=(10, 5))

                    ax.barh(
                        chart_data["Keyword"],
                        chart_data["Frequency"]
                    )

                    ax.set_xlabel("Frequency")
                    ax.set_ylabel("Keyword")
                    ax.set_title("Keyword Frequency Analysis")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                else:
                    st.info("No matching keywords found.")

                # PIE CHART - NEW FEATURE
                st.divider()
                st.subheader("🥧 Keyword Distribution Analysis")

                top_keywords = frequencies.most_common(5)

                pie_labels = [
                    keyword for keyword, count in top_keywords
                ]

                pie_values = [
                    count for keyword, count in top_keywords
                ]

                other_count = (
                    total_words - sum(pie_values)
                )

                if other_count > 0:
                    pie_labels.append("Others")
                    pie_values.append(other_count)

                fig, ax = plt.subplots(figsize=(8, 6))

                ax.pie(
                    pie_values,
                    labels=pie_labels,
                    autopct="%1.1f%%",
                    startangle=90
                )

                ax.set_title("Top 5 Keyword Distribution")
                ax.axis("equal")

                st.pyplot(fig)
                plt.close(fig)

                # WORD CLOUD
                st.divider()
                st.subheader("☁️ Keyword Word Cloud")

                wordcloud = WordCloud(
                    width=1200,
                    height=500,
                    background_color="white",
                    colormap="viridis",
                    max_words=100
                ).generate_from_frequencies(frequencies)

                fig, ax = plt.subplots(figsize=(14, 6))

                ax.imshow(wordcloud, interpolation="bilinear")
                ax.axis("off")

                st.pyplot(fig)
                plt.close(fig)

                # DOWNLOAD RESULTS
                st.divider()
                st.subheader("⬇️ Download Analysis Results")

                csv_data = filtered_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Keyword Results as CSV",
                    data=csv_data,
                    file_name="keyword_frequency_results.csv",
                    mime="text/csv"
                )

                # TWO-WORD PHRASE ANALYSIS
                st.divider()
                st.subheader("🔎 Popular Search Phrases")

                if all_phrases:

                    phrase_results = pd.DataFrame(
                        all_phrases.most_common(10),
                        columns=["Search Phrase", "Frequency"]
                    )

                    st.dataframe(
                        phrase_results,
                        use_container_width=True,
                        hide_index=True
                    )

                else:
                    st.info("Not enough words to generate phrases.")

    except Exception as e:
        st.error(f"Unable to process CSV file: {e}")


else:

    st.info("Upload a CSV file to start the analysis.")

    st.write("Expected CSV format:")

    sample_data = pd.DataFrame({
        "search_query": [
            "best python courses",
            "python programming tutorials",
            "best python projects",
            "java programming courses",
            "machine learning projects"
        ]
    })

    st.dataframe(sample_data, use_container_width=True)


# FOOTER
st.caption(
    "Search Engine Keyword Frequency Analysis | Python and Streamlit"
)
