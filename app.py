
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from collections import Counter
from wordcloud import WordCloud


# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="Search Engine Keyword Frequency Analysis",
    page_icon="🔍",
    layout="wide"
)


# ----------------------------------
# TITLE AND DESCRIPTION
# ----------------------------------

st.title("🔍 Search Engine Keyword Frequency Analysis")

st.write(
    "Analyze search queries to identify frequently "
    "used keywords and understand search patterns."
)

st.divider()


# ----------------------------------
# STOP WORDS
# ----------------------------------

STOP_WORDS = {
    "a", "an", "the", "is", "are", "was", "were",
    "and", "or", "but", "to", "of", "in", "on",
    "for", "with", "by", "at", "from", "how",
    "what", "which", "who", "where", "when",
    "why", "can", "could", "should", "would",
    "i", "me", "my", "we", "our", "you", "your",
    "it", "this", "that", "be", "do", "does"
}


# ----------------------------------
# CSV FILE UPLOAD
# ----------------------------------

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

            # ----------------------------------
            # DATASET PREVIEW
            # ----------------------------------

            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            # ----------------------------------
            # SELECT TEXT COLUMN
            # ----------------------------------

            text_column = st.selectbox(
                "Select the column containing search queries",
                options=df.columns
            )

            queries = df[text_column].dropna().astype(str)

            # ----------------------------------
            # KEYWORD FREQUENCY CALCULATION
            # ----------------------------------

            frequencies = Counter()

            for query in queries:

                words = re.findall(
                    r"\b[a-zA-Z0-9]+\b",
                    query.lower()
                )

                words = [
                    word for word in words
                    if word not in STOP_WORDS
                ]

                frequencies.update(words)

            if not frequencies:

                st.warning(
                    "No keywords found. Upload a CSV "
                    "with valid text."
                )

            else:

                total_words = sum(frequencies.values())

                # ----------------------------------
                # CREATE RESULTS TABLE
                # ----------------------------------

                results = pd.DataFrame(
                    frequencies.most_common(),
                    columns=["Keyword", "Frequency"]
                )

                results.insert(
                    0,
                    "Rank",
                    range(1, len(results) + 1)
                )

                results["Percentage"] = (
                    results["Frequency"] /
                    total_words * 100
                ).round(2)

                # ----------------------------------
                # ANALYSIS SUMMARY
                # ----------------------------------

                st.subheader("Analysis Summary")

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Total Words",
                    total_words
                )

                col2.metric(
                    "Unique Keywords",
                    len(frequencies)
                )

                col3.metric(
                    "Search Queries",
                    len(queries)
                )

                st.divider()

                # ----------------------------------
                # KEYWORD SEARCH
                # ----------------------------------

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

                # ----------------------------------
                # TOP KEYWORD SELECTION
                # ----------------------------------

                top_n = st.selectbox(
                    "Select number of top keywords",
                    options=[5, 10, 15, 20],
                    index=1
                )

                display_results = filtered_results.head(top_n)

                st.dataframe(
                    display_results,
                    use_container_width=True,
                    hide_index=True
                )

                # ----------------------------------
                # BAR CHART
                # ----------------------------------

                st.subheader("Top Keyword Frequency Chart")

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

                # ----------------------------------
                # WORD CLOUD VISUALIZATION
                # ----------------------------------

                st.divider()

                st.subheader("☁️ Keyword Word Cloud")

                st.write(
                    "Frequently occurring keywords appear larger "
                    "in the word cloud."
                )

                wordcloud = WordCloud(
                    width=1200,
                    height=500,
                    background_color="white",
                    colormap="viridis",
                    max_words=100,
                    contour_width=0
                ).generate_from_frequencies(frequencies)

                fig, ax = plt.subplots(figsize=(14, 6))

                ax.imshow(
                    wordcloud,
                    interpolation="bilinear"
                )

                ax.axis("off")

                st.pyplot(fig)

                plt.close(fig)

                # ----------------------------------
                # DOWNLOAD RESULTS
                # ----------------------------------

                st.divider()

                st.subheader("Download Analysis Results")

                csv_data = filtered_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    label="⬇️ Download Results as CSV",
                    data=csv_data,
                    file_name="keyword_frequency_results.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(
            f"Unable to process the CSV file: {e}"
        )


else:

    st.info(
        "Upload a CSV file to start the analysis."
    )

    st.write("**Expected CSV format:**")

    sample_data = pd.DataFrame({
        "search_query": [
            "best python courses",
            "python programming tutorials",
            "best python projects",
            "java programming courses",
            "machine learning projects"
        ]
    })

    st.dataframe(
        sample_data,
        use_container_width=True
    )


# ----------------------------------
# FOOTER
# ----------------------------------

st.caption(
    "Search Engine Keyword Frequency Analysis | "
    "Python and Streamlit"
)
