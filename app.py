import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="PH Housing Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# 2. Main Title & Introduction
st.title("Predicting Philippine Housing Prices: A Data Science Journey")

st.markdown("""
Welcome to this interactive walkthrough of the Philippine Housing Price Prediction project. 
This dashboard outlines the end-to-end methodology used to process raw real estate data and build robust predictive models.

### Our Methodology Flow:
1. **Exploratory Data Analysis (EDA)** 📍 *You are here*
2. **Data Preprocessing** (Outlier removal, log transformations)
3. **Modeling and Evaluation** (Algorithm training and feature importance)

---
""")

# 3. EDA Section
st.header("Phase 1: Exploratory Data Analysis")

st.markdown("""
Before making any alterations to the dataset, it is crucial to understand its initial state. 
Using automated data profiling, we generated a comprehensive overview of the 1,500 properties in our dataset. 

The full report contains interactive feature distributions, missing value matrixes, and outlier warnings 
(such as the extreme 43-bathroom property) that informed our preprocessing strategy.
""")

# 4. "Open in New Tab" Button
# Notice the href is now "app/static/ph_housing_eda.html"
st.markdown("""
    <br>
    <a href="app/static/ph_housing_eda.html" target="_blank" style="
        display: inline-block;
        padding: 0.5em 1em;
        color: #FFFFFF;
        background-color: #FF4B4B;
        border-radius: 0.5rem;
        text-decoration: none;
        font-weight: 600;
        text-align: center;
        border: 1px solid transparent;
        transition: 0.3s;
    ">
        📊 Open Full Interactive EDA Report (New Tab)
    </a>
""", unsafe_allow_html=True)