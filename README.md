# Search Engine Keyword Frequency Analysis

## Project Description
Search Engine Keyword Frequency Analysis is a Python-based project that analyzes search queries and identifies the most frequently used keywords. It helps users understand common search terms through tables and visualizations.

## Objectives
- Analyze search query data from CSV files.
- Identify frequently occurring keywords.
- Calculate keyword frequency and percentage.
- Display results using tables and bar charts.
- Allow users to download analysis results.

## Technologies Used
- Python
- Streamlit
- Pandas
- Matplotlib
- Regular Expressions

## Features
- CSV file upload
- Keyword frequency counting
- Keyword ranking
- Percentage calculation
- Interactive keyword search
- Bar chart visualization
- CSV result download

## How It Works
1. Upload a CSV file containing search queries.
2. Select the column containing the queries.
3. The application processes the text and removes common words.
4. It calculates keyword frequencies and percentages.
5. Results are displayed in a table and chart.
6. Users can download the results.

## How to Run
Install the required packages:

`pip install -r requirements.txt`

Run the application:

`streamlit run app.py`

## Note
The application analyzes the dataset provided by the user. It does not collect live search engine data automatically.
