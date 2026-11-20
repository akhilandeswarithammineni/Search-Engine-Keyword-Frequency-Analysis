
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from collections import Counter
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# PAGE CONFIGURATION
st.set_page_config(
    page_title="Keyword Frequency Analysis",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Search Engine Keyword Frequency Analysis")
st.write("Analyze keywords, discover search patterns and predict user search intent.")
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


# MACHINE LEARNING TRAINING DATA
training_data = [
    # INFORMATIONAL
    ("what is python", "Informational"),
    ("how to learn programming", "Informational"),
    ("python programming tutorial", "Informational"),
    ("explain machine learning", "Informational"),
    ("what is artificial intelligence", "Informational"),
    ("learn data science", "Informational"),
    ("python beginner guide", "Informational"),
    ("how does deep learning work", "Informational"),
    ("meaning of data analytics", "Informational"),
    ("examples of python programming", "Informational"),
    ("how to use pandas", "Informational"),
    ("machine learning algorithms explained", "Informational"),
    ("learn web development", "Informational"),
    ("what is neural network", "Informational"),
    ("python coding lessons", "Informational"),

    # NAVIGATIONAL
    ("python official website", "Navigational"),
    ("google homepage", "Navigational"),
    ("github login", "Navigational"),
    ("python documentation website", "Navigational"),
    ("facebook login page", "Navigational"),
    ("youtube official website", "Navigational"),
    ("openai homepage", "Navigational"),
    ("microsoft official website", "Navigational"),
    ("stackoverflow website", "Navigational"),
    ("google account login", "Navigational"),
    ("github official page", "Navigational"),
    ("python website", "Navigational"),
    ("linkedin login", "Navigational"),
    ("amazon customer support", "Navigational"),
    ("google contact page", "Navigational"),

    # TRANSACTIONAL
    ("buy python programming course", "Transactional"),
    ("download python software", "Transactional"),
    ("purchase online course", "Transactional"),
    ("register for coding classes", "Transactional"),
    ("subscribe to learning platform", "Transactional"),
    ("order programming books", "Transactional"),
    ("book python training", "Transactional"),
    ("sign up for python course", "Transactional"),
    ("buy machine learning course", "Transactional"),
    ("download data science software", "Transactional"),
    ("purchase online tutorials", "Transactional"),
    ("get discount on coding course", "Transactional"),
    ("buy programming books online", "Transactional"),
    ("register for web development course", "Transactional"),
    ("download coding application", "Transactional"),

    # COMMERCIAL
    ("best python courses", "Commercial"),
    ("top programming languages", "Commercial"),
    ("python vs java comparison", "Commercial"),
    ("best machine learning tools", "Commercial"),
    ("reviews of coding courses", "Commercial"),
    ("compare python courses", "Commercial"),
    ("best laptops for programming", "Commercial"),
    ("top data science platforms", "Commercial"),
    ("python course ratings", "Commercial"),
    ("best online learning platforms", "Commercial"),
    ("cheap programming courses", "Commercial"),
    ("best web development tools", "Commercial"),
    ("alternatives to python", "Commercial"),
    ("machine learning software comparison", "Commercial"),
    ("best coding bootcamps", "Commercial")
]

train_queries = [item[0] for item in training_data]
train_labels = [item[1] for item in training_data]


# TRAIN MACHINE LEARNING MODEL
@st.cache_resource
def train_model():

    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2)
    )

    X = vectorizer.fit_transform(train_queries)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    model.fit(X, train_labels)

    return vectorizer, model


vectorizer, model = train_model()


# CSV UPLOAD
uploaded_file = st.file_uploader(
    "Upload your search query CSV file",
    type=["csv"]
)


if uploaded_file is not None:

    try:

        df = pd.read_csv(uploaded_file)

        if df.empty:
            st.warning("CSV file is empty.")

        else:

            st.subheader("Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True)

            text_column = st.selectbox(
                "Select search query column",
                df.columns
            )

            queries = df[text_column].dropna().astype(str)

            if queries.empty:
                st.warning("No valid queries found.")

            else:

                # KEYWORD FREQUENCY
                frequencies = Counter()
                phrases = Counter()

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

                    phrases.update([
                        f"{clean_words[i]} {clean_words[i+1]}"
                        for i in range(len(clean_words) - 1)
                    ])

                total_words = sum(frequencies.values())

                # SUMMARY
                st.subheader("Analysis Summary")

                c1, c2, c3 = st.columns(3)

                c1.metric("Total Words", total_words)
                c2.metric("Unique Keywords", len(frequencies))
                c3.metric("Search Queries", len(queries))

                st.divider()

                # KEYWORD TABLE
                st.subheader("Keyword Frequency Results")

                results = pd.DataFrame(
                    frequencies.most_common(),
                    columns=["Keyword", "Frequency"]
                )

                results.insert(
                    0, "Rank", range(1, len(results) + 1)
                )

                if total_words:
                    results["Percentage"] = (
                        results["Frequency"] / total_words * 100
                    ).round(2)

                search_keyword = st.text_input("Search a keyword")

                filtered = results[
                    results["Keyword"].str.contains(
                        search_keyword,
                        case=False,
                        regex=False
                    )
                ]

                top_n = st.selectbox(
                    "Number of top keywords",
                    [5, 10, 15, 20],
                    index=1
                )

                st.dataframe(
                    filtered.head(top_n),
                    use_container_width=True,
                    hide_index=True
                )

                # BAR CHART
                st.subheader("📊 Keyword Frequency Chart")

                chart_data = filtered.head(top_n).sort_values(
                    "Frequency"
                )

                if not chart_data.empty:

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

                pie_labels = [x[0] for x in top_keywords]
                pie_values = [x[1] for x in top_keywords]

                other = total_words - sum(pie_values)

                if other > 0:
                    pie_labels.append("Others")
                    pie_values.append(other)

                fig, ax = plt.subplots(figsize=(8, 6))

                ax.pie(
                    pie_values,
                    labels=pie_labels,
                    autopct="%1.1f%%",
                    startangle=90
                )

                ax.set_title("Top Keyword Distribution")
                ax.axis("equal")

                st.pyplot(fig)
                plt.close(fig)

                # WORD CLOUD
                st.subheader("☁️ Keyword Word Cloud")

                if frequencies:

                    wc = WordCloud(
                        width=1200,
                        height=500,
                        background_color="white",
                        colormap="viridis"
                    ).generate_from_frequencies(frequencies)

                    fig, ax = plt.subplots(figsize=(14, 6))
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")

                    st.pyplot(fig)
                    plt.close(fig)

                # POPULAR PHRASES
                st.subheader("🔎 Popular Search Phrases")

                if phrases:

                    phrase_df = pd.DataFrame(
                        phrases.most_common(10),
                        columns=["Phrase", "Frequency"]
                    )

                    st.dataframe(
                        phrase_df,
                        use_container_width=True,
                        hide_index=True
                    )

                # MACHINE LEARNING PREDICTION
                st.divider()
                st.header("🤖 Machine Learning Search Intent Prediction")

                st.write(
                    "The TF-IDF and Logistic Regression model "
                    "predicts the intent behind each search query."
                )

                new_query = st.text_input(
                    "Enter a search query to predict its intent",
                    placeholder="Example: best Python courses"
                )

                if new_query.strip():

                    vector = vectorizer.transform([new_query])

                    prediction = model.predict(vector)[0]

                    probabilities = model.predict_proba(vector)[0]

                    confidence = max(probabilities) * 100

                    st.success(
                        f"Predicted Intent: {prediction}"
                    )

                    st.metric(
                        "Model Confidence",
                        f"{confidence:.2f}%"
                    )

                    st.caption(
                        "Confidence is the model's estimated "
                        "probability, not a guarantee of correctness."
                    )

                # CLASSIFY UPLOADED QUERIES
                st.subheader("Search Intent Classification")

                X_queries = vectorizer.transform(queries)

                predictions = model.predict(X_queries)

                probabilities = model.predict_proba(X_queries)

                confidence_scores = probabilities.max(axis=1) * 100

                intent_df = pd.DataFrame({
                    "Search Query": queries.values,
                    "Predicted Intent": predictions,
                    "Confidence (%)": confidence_scores.round(2)
                })

                st.dataframe(
                    intent_df,
                    use_container_width=True,
                    hide_index=True
                )

                # INTENT DISTRIBUTION
                st.subheader("📈 Predicted Search Intent Distribution")

                intent_counts = intent_df[
                    "Predicted Intent"
                ].value_counts()

                fig, ax = plt.subplots(figsize=(9, 5))

                intent_counts.plot(
                    kind="bar",
                    ax=ax
                )

                ax.set_xlabel("Search Intent")
                ax.set_ylabel("Number of Queries")
                ax.set_title("Machine Learning Intent Predictions")

                plt.xticks(rotation=20)
                plt.tight_layout()

                st.pyplot(fig)
                plt.close(fig)

                # DOWNLOAD KEYWORDS
                st.divider()
                st.subheader("⬇️ Download Results")

                st.download_button(
                    "Download Keyword Results",
                    results.to_csv(index=False).encode("utf-8"),
                    "keyword_results.csv",
                    "text/csv"
                )

                st.download_button(
                    "Download ML Intent Predictions",
                    intent_df.to_csv(index=False).encode("utf-8"),
                    "ml_intent_predictions.csv",
                    "text/csv"
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


st.caption(
    "Search Engine Keyword Frequency Analysis | "
    "Python, Streamlit and Machine Learning"
)
