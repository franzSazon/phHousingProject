import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from matplotlib.ticker import FuncFormatter

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
1. **Exploratory Data Analysis (EDA)
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

st.markdown("---")

# 6. Phase 3: Modeling & Evaluation Section
st.header("Phase 3: Modeling and Evaluation")

st.markdown("""
With a clean, normally distributed dataset, we split the properties into a **Training Set (80%)** to teach the algorithms, 
and a **Testing Set (20%)** to evaluate how accurately they predict unseen real estate prices. 

We trained four distinct machine learning architectures. Explore the tabs below to see how each model interprets the market and performs against the testing data.
""")

# Cache the model training so it doesn't re-run on every page load
@st.cache_resource
def train_and_evaluate_models(df):
    # Final Feature Selection 
    X = df.drop(['Log_Price', 'Price', 'Floor Area', 'Land Area'], axis=1, errors='ignore')
    y = df['Log_Price']
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models_data = {}
    
    # Model 1: Linear Regression
    lr = LinearRegression().fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    models_data['Linear Regression'] = {
        'model': lr,
        'MAE': mean_absolute_error(y_test, lr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, lr_pred)),
        'R²': r2_score(y_test, lr_pred),
        'predictions': lr_pred,
        'description': "Our baseline algorithm. It assumes a strict straight-line relationship between house features and price. While stable, it struggles to adapt to the geographic complexities of Latitude and Longitude."
    }
    
    # Model 2: Polynomial Regression (Degree 2)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    pr = LinearRegression().fit(X_train_poly, y_train)
    pr_pred = pr.predict(X_test_poly)
    models_data['Polynomial Regression'] = {
        'model': pr,
        'transformer': poly,
        'MAE': mean_absolute_error(y_test, pr_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, pr_pred)),
        'R²': r2_score(y_test, pr_pred),
        'predictions': pr_pred,
        'description': "An upgrade over the baseline. By multiplying features together (e.g., Bedrooms × Floor Area), this model begins to capture mathematical interactions that drive up property values."
    }
    
    # Model 3: Decision Tree
    dt = DecisionTreeRegressor(random_state=42).fit(X_train, y_train)
    dt_pred = dt.predict(X_test)
    models_data['Decision Tree'] = {
        'model': dt,
        'MAE': mean_absolute_error(y_test, dt_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, dt_pred)),
        'R²': r2_score(y_test, dt_pred),
        'predictions': dt_pred,
        'description': "A rule-based algorithm that essentially draws 'bounding boxes' around the data. It can easily memorize the training data, but as seen below, it struggles slightly when generalizing to new, unseen houses."
    }
    
    # Model 4: Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    models_data['Random Forest'] = {
        'model': rf,
        'MAE': mean_absolute_error(y_test, rf_pred),
        'RMSE': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'R²': r2_score(y_test, rf_pred),
        'predictions': rf_pred,
        'description': "The undisputed champion of our analysis. By training 100 separate decision trees and averaging their results, it completely eliminates the overfitting problem and seamlessly maps geographic zones."
    }
    
    # Create the leaderboard dataframe
    metrics_only = {name: {'MAE': d['MAE'], 'RMSE': d['RMSE'], 'R²': d['R²']} for name, d in models_data.items()}
    results_df = pd.DataFrame(metrics_only).T.sort_values(by='R²', ascending=False)
    
    return models_data, results_df, rf, X.columns, y_test

# Execute the modeling pipeline
models_data, leaderboard_df, winning_model, feature_names, y_test = train_and_evaluate_models(df_filtered)

# Helper function to format axis labels into Millions (M)
def millions_formatter(x, pos):
    return f'{x / 1e6:.0f}M'

# Updated Helper function with toggle logic
def plot_predictions(y_true, y_pred, model_name, show_raw):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    if show_raw:
        # Reverse the log10 transformation
        plot_y_true = 10 ** y_true
        plot_y_pred = 10 ** y_pred
        title_suffix = "(Raw Price in PHP)"
        xlabel = 'Actual Price (PHP)'
        ylabel = 'Predicted Price (PHP)'
    else:
        # Keep the log10 values
        plot_y_true = y_true
        plot_y_pred = y_pred
        title_suffix = "(Log10 Scale)"
        xlabel = 'Actual Price (Log10)'
        ylabel = 'Predicted Price (Log10)'

    ax.scatter(plot_y_true, plot_y_pred, alpha=0.6, color='dodgerblue', edgecolor='k')
    
    # Define bounds for the perfect prediction line
    min_val = min(plot_y_true.min(), plot_y_pred.min())
    max_val = max(plot_y_true.max(), plot_y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_title(f'{model_name}: Actual vs. Predicted {title_suffix}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Apply the millions formatter if raw prices are shown
    if show_raw:
        ax.xaxis.set_major_formatter(FuncFormatter(millions_formatter))
        ax.yaxis.set_major_formatter(FuncFormatter(millions_formatter))
        
    return fig

# --- The Interactive Model Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest"])

tabs = [tab1, tab2, tab3, tab4]
model_names = ["Linear Regression", "Polynomial Regression", "Decision Tree", "Random Forest"]

for tab, model_name in zip(tabs, model_names):
    with tab:
        data = models_data[model_name]
        st.markdown(f"**Methodology:** {data['description']}")
        
        # Display individual metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("R-Squared (R²)", f"{data['R²']:.4f}")
        col2.metric("RMSE", f"{data['RMSE']:.4f}")
        col3.metric("MAE", f"{data['MAE']:.4f}")
        
        # Add the toggle switch
        # We use a unique key for each toggle so they don't interfere with one another
        show_raw = st.toggle("Show in Raw Price (PHP)", value=False, key=f"toggle_{model_name}")
        
        # Display the scatter plot
        fig = plot_predictions(y_test, data['predictions'], model_name, show_raw)
        st.pyplot(fig)


# --- Final Summary & Feature Importance ---
st.markdown("---")
st.subheader("The Final Leaderboard")
st.markdown("Comparing all algorithms side-by-side confirms the power of ensemble modeling in real estate valuation:")

# Display the leaderboard dynamically
st.dataframe(
    leaderboard_df.style.format("{:.4f}").highlight_max(subset=['R²'], color='lightgreen').highlight_min(subset=['MAE', 'RMSE'], color='lightgreen'),
    use_container_width=True
)

st.subheader("What actually drives Philippine property prices?")
st.markdown("""
Using our top-performing Random Forest model, we can extract the exact "importance" it assigned to each variable when predicting a house's value. 
""")

# Calculate and plot feature importances
importances = winning_model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance (%)': importances * 100
}).sort_values(by='Importance (%)', ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(10, 5))
sns.barplot(x='Importance (%)', y='Feature', data=importance_df, palette='viridis', ax=ax_imp)
ax_imp.set_title('Random Forest Feature Importance')
ax_imp.set_xlabel('Contribution to Model Prediction (%)')
ax_imp.set_ylabel('')
st.pyplot(fig_imp)

st.markdown("""
### The Multicollinearity Insight
As seen above, **Floor Area** accounts for an overwhelming majority of the prediction power, while **Bedrooms** and **Bathrooms** register surprisingly low. 

Does this mean bedrooms don't matter? **No.** Earlier correlation analysis revealed high *multicollinearity* between floor area and room counts (i.e., larger houses naturally have more rooms). Rather than splitting credit between them, the Random Forest efficiently grouped their predictive weight into the "Floor Area" feature, treating it as the ultimate proxy for a property's overall capacity.
""")

st.markdown("---")

# 7. Phase 4: Interactive Prediction Sandbox
st.header("Phase 4: Interactive Market Simulator")
st.markdown("""
Put the models to the test! Adjust the property parameters below to see how each algorithm estimates the market value in real-time. 
*Note: The models were trained on Log10 areas, but you can enter the raw Square Meters below and the dashboard will handle the complex math automatically.*
""")

# Create a clean 3-column layout for the input controls
col_in1, col_in2, col_in3 = st.columns(3)

with col_in1:
    st.markdown("**Physical Layout**")
    sim_bed = st.slider("Bedrooms", min_value=1, max_value=10, value=3)
    sim_bath = st.slider("Bathrooms", min_value=1, max_value=10, value=2)

with col_in2:
    st.markdown("**Space (sqm)**")
    sim_floor = st.number_input("Floor Area", min_value=20, max_value=2000, value=120)
    sim_land = st.number_input("Land Area", min_value=20, max_value=2000, value=150)

with col_in3:
    st.markdown("**Location (Coordinates)**")
    # Bounds set roughly to the Philippines geography
    sim_lat = st.slider("Latitude (North/South)", min_value=5.0, max_value=19.0, value=14.55, step=0.01)
    sim_long = st.slider("Longitude (East/West)", min_value=117.0, max_value=126.0, value=121.02, step=0.01)

# Format inputs mathematically exactly as the models expect them
sim_data = pd.DataFrame({
    'Bedrooms': [sim_bed],
    'Bathrooms': [sim_bath],
    'Log_Floor_Area': [np.log10(sim_floor)],
    'Log_Land_Area': [np.log10(sim_land)],
    'Latitude': [sim_lat],
    'Longitude': [sim_long]
})

# Ensure column order matches the training data exactly
sim_data = sim_data[feature_names]

# Generate predictions and reverse the log (10 ** pred) to get raw Pesos
pred_lr = 10 ** models_data['Linear Regression']['model'].predict(sim_data)[0]
pred_dt = 10 ** models_data['Decision Tree']['model'].predict(sim_data)[0]
pred_rf = 10 ** models_data['Random Forest']['model'].predict(sim_data)[0]

# Polynomial requires the transformer first
poly_transformer = models_data['Polynomial Regression']['transformer']
poly_model = models_data['Polynomial Regression']['model']
pred_pr = 10 ** poly_model.predict(poly_transformer.transform(sim_data))[0]

st.markdown("<br>", unsafe_allow_html=True)
st.subheader("Live Valuation Estimates")

# Display results in large, stylized metric cards
col_out1, col_out2, col_out3, col_out4 = st.columns(4)

def format_price(val):
    return f"₱ {val:,.0f}"

col_out1.metric("Linear Regression", format_price(pred_lr))
col_out2.metric("Polynomial Reg.", format_price(pred_pr))
col_out3.metric("Decision Tree", format_price(pred_dt))
col_out4.metric("🏆 Random Forest", format_price(pred_rf))
