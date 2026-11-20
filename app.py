
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
    "Analyze search queries, discover popular keywords "
    "and understand user search behavior."
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


# SEARCH INTENT CLASSIFICATION
def classify_intent(query):

    q = query.lower()

    transactional = [
        "buy", "purchase", "order", "download",
        "subscribe", "sign up", "register",
        "book", "price", "coupon", "discount"
    ]

    navigational = [
        "official website", "login", "log in",
        "homepage", "website", "contact",
        "customer support"
    ]

    commercial = [
        "best", "top", "review", "reviews",
        "comparison", "compare", "vs",
        "alternative", "alternatives",
        "rating", "cheap"
    ]

    informational = [
        "what", "why", "how", "when", "where",
        "who", "meaning", "definition", "tutorial",
        "guide", "learn", "examples", "explain"
    ]

    if any(word in q for word in transactional):
        return "Transactional"

    elif any(word in q for word in navigational):
        return "Navigational"

    elif any(word in q for word in commercial):
        return "Commercial"

    elif any(word in q for word in informational):
        return "Informational"

    else:
        return "Unclassified"


# CSV UPLOAD
uploaded_file = st.file_uploader(
    "Upload your search query CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("Your CSV file is empty.")

        else:

            # DATASET PREVIEW
            st.subheader("Dataset Preview")

            st.dataframe(
                df.head(10),
                use_container_width=True
            )

            # COLUMN SELECTION
            text_column = st.selectbox(
                "Select search query column",
                df.columns
            )

            queries = df[text_column].dropna().astype(str)

            if queries.empty:
                st.warning("No valid search queries found.")

            else:

                # KEYWORD FREQUENCY
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

                    phrases = [
                        f"{clean_words[i]} {clean_words[i+1]}"
                        for i in range(len(clean_words) - 1)
                    ]

                    all_phrases.update(phrases)

                # SEARCH INTENT RESULTS
                df["Search Intent"] = df[text_column].apply(
                    lambda x: classify_intent(str(x))
                    if pd.notna(x) else "Unclassified"
                )

                # SUMMARY
                total_words = sum(frequencies.values())

                st.subheader("Analysis Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric("Total Words", total_words)
                c2.metric("Unique Keywords", len(frequencies))
                c3.metric("Search Queries", len(queries))

                st.divider()

                # KEYWORD TABLE
                st.subheader("Keyword Frequency Results")

                search_keyword = st.text_input(
                    "Search for a keyword"
                )

                top_n = st.selectbox(
                    "Select top keywords",
                    [5, 10, 15, 20],
                    index=1
                )

                results = pd.DataFrame(
                    frequencies.most_common(),
                    columns=["Keyword", "Frequency"]
                )

                results.insert(
                    0, "Rank", range(1, len(results) + 1)
                )

                if total_words > 0:
                    results["Percentage"] = (
                        results["Frequency"] /
                        total_words * 100
                    ).round(2)

                filtered_results = results[
                    results["Keyword"].str.contains(
                        search_keyword,
                        case=False,
                        na=False,
                        regex=False
                    )
                ]

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
                        "Frequency"
                    )

                    fig, ax = plt.subplots(figsize=(10, 5))

                    ax.barh(
                        chart_data["Keyword"],
                        chart_data["Frequency"]
                    )

                    ax.set_xlabel("Frequency")
                    ax.set_title("Top Keyword Frequency")

                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # PIE CHART
                st.subheader("🥧 Keyword Distribution")

                top_keywords = frequencies.most_common(5)

                pie_labels = [
                    word for word, count in top_keywords
                ]

                pie_values = [
                    count for word, count in top_keywords
                ]

                other_count = total_words - sum(pie_values)

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
                st.subheader("☁️ Keyword Word Cloud")

                if frequencies:

                    wordcloud = WordCloud(
                        width=1200,
                        height=500,
                        background_color="white",
                        colormap="viridis",
                        max_words=100
                    ).generate_from_frequencies(frequencies)

                    fig, ax = plt.subplots(figsize=(14, 6))

                    ax.imshow(
                        wordcloud,
                        interpolation="bilinear"
                    )

                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                # POPULAR PHRASES
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

                # SEARCH INTENT ANALYSIS
                st.divider()

                st.header("🎯 Search Intent Analysis")

                st.write(
                    "Understand the purpose behind users' "
                    "search queries."
                )

                intent_counts = df["Search Intent"].value_counts()

                a, b, c, d = st.columns(4)

                a.metric(
                    "Informational",
                    int(intent_counts.get("Informational", 0))
                )

                b.metric(
                    "Navigational",
                    int(intent_counts.get("Navigational", 0))
                )

                c.metric(
                    "Transactional",
                    int(intent_counts.get("Transactional", 0))
                )

                d.metric(
                    "Commercial",
                    int(intent_counts.get("Commercial", 0))
                )

                # INTENT CHART
                fig, ax = plt.subplots(figsize=(9, 5))

                intent_counts.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_title("Search Intent Distribution")
                ax.set_xlabel("Search Intent")
                ax.set_ylabel("Number of Queries")

                plt.xticks(rotation=20)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # INTENT TABLE
                st.subheader("Search Query Classification")

                st.dataframe(
                    df[[text_column, "Search Intent"]],
                    use_container_width=True,
                    hide_index=True
                )

                # DOWNLOAD
                st.divider()

                st.subheader("⬇️ Download Analysis Results")

                download_data = results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "Download Keyword Results",
                    data=download_data,
                    file_name="keyword_results.csv",
                    mime="text/csv"
                )

                intent_download = df[
                    [text_column, "Search Intent"]
                ].to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download Search Intent Results",
                    data=intent_download,
                    file_name="search_intent_results.csv",
                    mime="text/csv"
                )

    except Exception as e:

        st.error(f"Error processing file: {e}")


else:

    st.info("Upload a CSV file to begin.")

    sample = pd.DataFrame({
        "search_query": [
            "what is python",
            "best python courses",
            "buy python course",
            "python official website",
            "learn machine learning"
        ]
    })

    st.dataframe(sample, use_container_width=True)


# FOOTER
st.caption(
    "Search Engine Keyword Frequency Analysis | "
    "Python and Streamlit"
)
