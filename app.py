import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

Medyo naa diyud parts sa data na kanang need i modify kay naga create og unrealistic scenarios and need nato i handle 
para better ang accuracy og generalization sa model.

""")

# 4. "Open in New Tab" Button
# Notice the href is now "app/static/ph_housing_eda.html"
st.markdown("""
    <br>
    <a href="https://franzsazon.github.io/phHousingProject/static/ph_housing_eda.html" target="_blank" style="
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

st.markdown("---")

# 5. Phase 2: Data Preprocessing Section
st.header("Phase 2: Data Preprocessing")

st.markdown("""
Raw data is rarely ready for machine learning right out of the box. As noted in the EDA, we had extreme outliers 
and heavily skewed distributions that would confuse our predictive algorithms. 

Here is how we transformed the dataset to ensure our models learn the true market trends rather than memorizing extreme anomalies:
""")

# Load the raw data (Streamlit will cache this so it doesn't reload constantly)
@st.cache_data
def load_data():
    return pd.read_csv('PH_Housing.csv')

raw_df = load_data()

# 1. Missing Values & Outlier Removal (The IQR Method)
st.subheader("1. Cleaning the Anomalies")
st.markdown("""
First, we removed incomplete records and dropped categorical columns that wouldn't be used in our numerical models. 
Next, we applied the **Interquartile Range (IQR)** method to filter out extreme structural outliers—like properties listing 40+ bathrooms.
""")

# Perform the cleaning steps
df_clean = raw_df.copy()
df_clean.drop(["HouseID", "Location", "Description"], axis=1, inplace=True, errors='ignore')
df_clean = df_clean.dropna()

Q1_bed, Q3_bed = df_clean['Bedrooms'].quantile(0.25), df_clean['Bedrooms'].quantile(0.75)
IQR_bed = Q3_bed - Q1_bed
df_filtered = df_clean[
    (df_clean['Bedrooms'] >= Q1_bed - 1.5 * IQR_bed) & 
    (df_clean['Bedrooms'] <= Q3_bed + 1.5 * IQR_bed)
].copy()

Q1_bath, Q3_bath = df_filtered['Bathrooms'].quantile(0.25), df_filtered['Bathrooms'].quantile(0.75)
IQR_bath = Q3_bath - Q1_bath
df_filtered = df_filtered[
    (df_filtered['Bathrooms'] >= Q1_bath - 1.5 * IQR_bath) & 
    (df_filtered['Bathrooms'] <= Q3_bath + 1.5 * IQR_bath)
]

# Display before and after metrics using Streamlit's native metric component
col1, col2, col3 = st.columns(3)
col1.metric(label="Original Dataset Size", value=f"{len(raw_df)} properties")
col2.metric(label="Cleaned Dataset Size", value=f"{len(df_filtered)} properties", delta=f"{len(df_filtered) - len(raw_df)} outliers removed", delta_color="inverse")
col3.metric(label="Features Retained", value="5 predictors")

# 2. Log Transformations
st.subheader("2. Taming the Skew (Log Transformations)")
st.markdown("""
Real estate prices and land areas naturally suffer from extreme **right-skewness**. Most houses cluster around a typical price range, 
but a few ultra-luxury estates stretch the axis out to the billions. 

To fix this, we applied a **Base-10 Logarithmic Transformation**. Notice how the transformation pulls the data into a beautiful, predictable bell curve below:
""")

# Apply log transformations
df_filtered['Log_Price'] = np.log10(df_filtered['Price'])
df_filtered['Log_Floor_Area'] = np.log10(df_filtered['Floor Area'])
df_filtered['Log_Land_Area'] = np.log10(df_filtered['Land Area'])

# Create Before & After visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Before: Raw Price Distribution
sns.histplot(df_filtered['Price'], bins=50, kde=True, ax=axes[0], color='salmon')
axes[0].set_title('BEFORE: Raw Price Distribution (Right-Skewed)')
axes[0].set_xlabel('Price (PHP)')
axes[0].set_ylabel('Count')

# After: Log Price Distribution
sns.histplot(df_filtered['Log_Price'], bins=50, kde=True, ax=axes[1], color='mediumseagreen')
axes[1].set_title('AFTER: Log10 Price Distribution (Normal Curve)')
axes[1].set_xlabel('Log10(Price)')
axes[1].set_ylabel('Count')

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)
