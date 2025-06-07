import streamlit as st
import numpy as np
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd  # Required for EDA


# ✅ Set page config FIRST before any other Streamlit command
st.set_page_config(page_title="Box Office Revenue Prediction", page_icon="🎮", layout="wide")

st.markdown(
    """
    <style>
        html, body, [class*="st-"] {
            background-color: white !important;
            color: black !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Load dataset for EDA (Modify path as needed)
data_path = r"C:\Users\nincy\Downloads\BORP-PROJECT\boxoffice.csv"  # Adjust the path
if os.path.exists(data_path):
    df = pd.read_csv(data_path)
else:
    df = None
    st.warning("⚠️ Dataset not found! Ensure 'box_office_data.csv' is in the directory.")

# Display an Image (Check if File Exists)
image_path = "Innomatics-Logo1.png"
if os.path.exists(image_path):
    st.image(image_path, width=700)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Box Office Revenue Prediction Using ML", ["Introduction", "EDA Visualization", "Prediction App"])

# 🎮 Introduction Page
if page == "Introduction":
    st.markdown(
        "<h1 style='text-align: center;'>Box Office Revenue Prediction</h1>",
        unsafe_allow_html=True
    )
    st.markdown(
        "<h2 style='text-align: center;'>🎮 Overview of the Project</h2>",
        unsafe_allow_html=True
    )

    # Load and show the Box Office image
    image_path_intro = "BOX OFFICE PIC.jpeg"
    if os.path.exists(image_path_intro):
        st.image(image_path_intro, caption="Box Office Earnings", use_container_width=True)

   
    st.header("📌 Introduction")
    st.write(
        "The Box Office Revenue Prediction Project uses machine learning to predict how much money a movie might earn. By analyzing details like financial information, genre, budget, and audience interest, it helps filmmakers plan marketing, budgets, and release strategies more effectively. This project combines technology with creativity to make smarter decisions in the film industry."
    )

    st.header("💼 Business Problem")
    st.write("- In the entertainment industry, predicting a movie's box office revenue is crucial for production companies, investors, and distributors.")
    st.write("- Accurately estimating revenue can help in decision-making related to budgeting, marketing strategies, and release planning.")
    st.write("- However, box office success is influenced by multiple factors such as genre, budget, distributor, opening revenue, MPAA rating, and number of release days.")
    st.write("- The challenge is to build a model that can predict the worldwide box office revenue of a movie based on various attributes available before and shortly after its release.")

    st.header("🎯 Business Objective")
    st.write("- The goal is to develop a predictive model that estimates a movie's world revenue based on its attributes like domestic revenue, opening revenue, budget, MPAA rating, distributor, genre, and release days.")
    st.write("- This model can help:")
    st.write("  1. **Producers & Investors** – Decide which movies to fund and how much to invest.")
    st.write("  2. **Distributors & Studios** – Optimize marketing budgets and release strategies.")
    st.write("  3. **Theaters & Streaming Services** – Predict demand for specific movie types.")

    st.header("⚖️ Business Constraints")
    st.write("- **Accuracy vs. Interpretability** – While accuracy is important, the model should also be interpretable for business stakeholders.")
    st.write("- **Limited Data Availability** – Some movie features (like word-of-mouth impact) may not be available before release.")
    st.write("- **Dynamic Market Trends** – Audience preferences and external factors (e.g., competition, streaming impact) constantly change.")
    st.write("- **Computation Time** – The model should provide predictions efficiently for real-time decision-making.")


    


elif page == "EDA Visualization":
    if df is None:
        st.error("⚠️ Dataset not found! Please upload the correct CSV file.")
        st.stop()

    st.title("🎬 Box Office Revenue Data Analysis")

    # 🔹 Histogram
    st.header("📊 Histogram Distributions")
    valid_cols = [col for col in ["budget", "domestic_revenue", "world_revenue"] if col in df.columns]
    fig, axes = plt.subplots(1, len(valid_cols), figsize=(6 * len(valid_cols), 5), facecolor="black")

    colors = ["cyan", "lime", "orange"]
    for i, (col, color) in enumerate(zip(valid_cols, colors)):
        axes[i].set_facecolor("black")  # Dark background
        sns.histplot(df[col].dropna(), bins=30, kde=True, ax=axes[i], color=color)
        axes[i].set_title(f"{col.replace('_', ' ').title()} Distribution", color="white")
        axes[i].set_xlabel(col.replace('_', ' ').title(), color="white")
        axes[i].set_ylabel("Count", color="white")
        axes[i].spines[:].set_color("white")
        axes[i].tick_params(axis="both", colors="white")  # White tick labels

    plt.tight_layout()
    st.pyplot(fig)

    
    # 📌 Insights
    st.markdown("""
    **Insights**
    - Budgets, Domestic, and World Revenues are fairly evenly distributed, indicating a wide variety of film scales and earnings.
    - Most films fall within mid-to-high budget and revenue ranges.
    """)

    # 🔹 Scatter Plot (Budget vs. World Revenue)
    if "budget" in df.columns and "world_revenue" in df.columns:
        st.header("📉 Scatter Plot")
        fig, ax = plt.subplots(figsize=(8,5), facecolor="black")
        ax.set_facecolor("black")  # Dark background
        sns.scatterplot(x=df["budget"], y=df["world_revenue"], alpha=0.5, color="cyan", ax=ax)
        ax.set_title("Budget vs. World Revenue", color="white")
        ax.set_xlabel("Budget", color="white")
        ax.set_ylabel("World Revenue", color="white")
        ax.spines[:].set_color("white")
        ax.tick_params(axis="both", colors="white")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)  # Subtle grid
        st.pyplot(fig)

     # 📌 Insights
    st.markdown("""
    **Insights**
    - There is no strong linear correlation between budget and world revenue.
    - Movies with both low and high budgets can achieve high or low global revenues, indicating that budget alone does not guarantee success.
    - The distribution is widely scattered, suggesting that other factors also play a key role in a movie’s global performance.
    """)   

    # 🔹 Count Plot (MPAA Ratings)
    if "MPAA" in df.columns:
        st.header("🎭 Count Plot")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="black")
        ax.set_facecolor("black")  
        sns.countplot(x=df["MPAA"].dropna(), palette="coolwarm", ax=ax)
        ax.set_title("Movie Count by MPAA Rating", color="white")
        ax.set_xlabel("MPAA Rating", color="white")
        ax.set_ylabel("Count", color="white")
        ax.spines[:].set_color("white")
        ax.tick_params(axis="both", colors="white")
        st.pyplot(fig)

     # 📌 Insights
    st.markdown("""
    **Insights**
    - Movies are fairly evenly distributed across all MPAA ratings.
    - R-rated films have the highest count, suggesting they are the most frequently produced.
    - PG-rated movies have the lowest count but are still significant in number.
    - This indicates a balanced production across different audience groups, with a slight lean toward mature content.
    """)   
   

    # 🔹 Box Plot (MPAA Ratings vs Revenue)
    if "MPAA" in df.columns and "world_revenue" in df.columns:
        st.header("📦 Box Plot")
        fig, ax = plt.subplots(figsize=(8, 5), facecolor="black")
        ax.set_facecolor("black")  
        sns.boxplot(x=df["MPAA"].dropna(), y=df["world_revenue"], palette="muted", ax=ax)
        ax.set_title("World Revenue by MPAA Rating", color="white")
        ax.set_xlabel("MPAA", color="white")
        ax.set_ylabel("World Revenue", color="white")
        ax.spines[:].set_color("white")
        ax.tick_params(axis="both", colors="white")
        st.pyplot(fig)

    # 📌 Insights
    st.markdown("""
    **Insights**
    - World revenue is fairly consistent across all MPAA ratings, with similar medians and wide ranges.
                 No rating shows a clear advantage in earning potential globally.
    """)       

    # 🔹 Bar Chart (Top 15 Movies by Revenue)
    if "world_revenue" in df.columns and "title" in df.columns:
        st.header("🎥 Bar Chart")
        top_movies = df.nlargest(15, "world_revenue").dropna(subset=["title"])
        fig, ax = plt.subplots(figsize=(10, 5), facecolor="black")
        ax.set_facecolor("black")  
        ax.bar(top_movies["title"], top_movies["world_revenue"], color="cyan", label="World Revenue")
        ax.bar(top_movies["title"], top_movies["domestic_revenue"], color="orange", label="Domestic Revenue")
        ax.set_xticks(range(len(top_movies["title"])))
        ax.set_xticklabels(top_movies["title"], rotation=45, ha="right", color="white")
        ax.set_title("Top Movies by Revenue", color="white")
        ax.set_xlabel("Movie Title", color="white")
        ax.set_ylabel("Revenue", color="white")
        ax.legend(facecolor="red", edgecolor="white")
        ax.spines[:].set_color("white")
        ax.tick_params(axis="both", colors="white")
        ax.grid(color="gray", linestyle="--", linewidth=0.5)
        st.pyplot(fig)
    


    # 🔹 Genre Distribution (Pie Chart)
    if "genres" in df.columns:
        st.header("🔘 Pie Chart")
        genre_counts = df["genres"].value_counts()
        dark_colors = ["#6A0DAD", "#2E8B57", "#DC143C", "#1E90FF", "#FFD700", "#8B0000", "#008080", "#A52A2A", "#FF8C00", "#4682B4"]
        
        fig, ax = plt.subplots(figsize=(8, 8), facecolor="black")
        ax.set_facecolor("black")
        wedges, texts, autotexts = ax.pie(
            genre_counts.values,
            labels=genre_counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=dark_colors[:len(genre_counts)]
        )
        for text in texts + autotexts:
            text.set_color("white")
        ax.set_title("Movie Genre Distribution", color="white", fontweight="bold")
        st.pyplot(fig)


    # 📌 Insights
    st.markdown("""
    **Insights**
    - This chart helps me understand the popularity of different genres. 
    - Comedy and Animation are the top genres, while Horror and Drama are slightly less common. 
    - But overall, all genres are quite balanced, and no genre is completely dominating the dataset.            
    """)  


    # 🔹 Average World Revenue by Distributor (Line Plot)
    if "distributor" in df.columns and "world_revenue" in df.columns:
        st.header("🏢 Line Plot")
        avg_rev_by_distributor = df.groupby("distributor")["world_revenue"].mean().sort_values(ascending=False).head(10)
        
        fig, ax = plt.subplots(figsize=(10, 6), facecolor="black")
        ax.set_facecolor("black")
        sns.lineplot(x=avg_rev_by_distributor.index, y=avg_rev_by_distributor.values, marker="o", linewidth=2.5, color="orange", ax=ax)
        ax.set_title("Average World Revenue by Distributors", fontweight='bold', color="white")
        ax.set_xlabel("Distributor", color="white")
        ax.set_ylabel("Average World Revenue", color="white")
        ax.tick_params(axis="x", colors="white", rotation=45)
        ax.tick_params(axis="y", colors="white")
        ax.spines[:].set_color("white")
        ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
        st.pyplot(fig)

    # 📌 Insights
    st.markdown("""
    **Insights**
    - Disney has the highest average world revenue.
    - Universal has the lowest among the top 5 shown.
    - Disney leads in global revenue generation, suggesting their strong market impact and popularity among audiences.
    """) 

    # 🔹 Correlation Heatmap
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    if not numeric_df.empty:
        st.header("🔍 Correlation Heatmap")
        correlation_matrix = numeric_df.corr()
        
        fig, ax = plt.subplots(figsize=(10, 8), facecolor="black")
        ax.set_facecolor("black")  

        heatmap = sns.heatmap(
            correlation_matrix, 
            annot=True, 
            cmap="coolwarm", 
            fmt=".2f", 
            linewidths=0.5, 
            ax=ax
        )

        # Set title and tick labels to white
        ax.set_title("Correlation Heatmap of Numerical Features", color="white")
        ax.tick_params(axis="x", colors="white", rotation=45)
        ax.tick_params(axis="y", colors="white", rotation=0)

        # Set labels manually if needed
        ax.set_xticklabels(ax.get_xticklabels(), color="white")
        ax.set_yticklabels(ax.get_yticklabels(), color="white")

        st.pyplot(fig)
    
    # 📌 Insights
    st.markdown("""
    **Insights**
    - The diagonal values are all 1.00 because a column is always perfectly correlated with itself.
    - Other values are very close to 0, meaning there is no strong correlation between most features.
    """) 




# 🎥 Prediction App Page
elif page == "Prediction App":
    st.title("🎬 Box Office Revenue Prediction App")

    # Load trained model and scaler
    model_path = "random_forest_model.pkl"
    scaler_path = "scaler.pkl"

    if os.path.exists(model_path) and os.path.exists(scaler_path):
        with open(model_path, "rb") as model_file:
            model = pickle.load(model_file)
        with open(scaler_path, "rb") as scaler_file:
            scaler = pickle.load(scaler_file)
    else:
        st.error("⚠️ Model or scaler file not found! Ensure they are in the directory.")
        st.stop()

    expected_features = [
        'domestic_revenue', 'opening_revenue', 'opening_theaters', 'budget', 'release_days',
        'Animation', 'Action', 'Horror', 'Comedy', 'Drama', 'Thriller',
        'opening_revenue_ratio', 'budget_to_revenue_ratio', 'domestic_to_world_ratio',
        'revenue_per_theater', 'MPAA_G', 'MPAA_NC17', 'MPAA_PG', 'MPAA_PG13', 'MPAA_R',
        'Distributor_Disney', 'Distributor_Paramount', 'Distributor_Sony',
        'Distributor_Universal', 'Distributor_Warner Bros.'
    ]

    # 📝 User Input (Step by Step)
    with st.form("movie_input_form"):
        st.subheader("📌 Step 1: Enter Financial Information")
        domestic_revenue = st.number_input("Domestic Revenue ($)", value=10000000, step=100000)
        opening_revenue = st.number_input("Opening Revenue ($)", value=5000000, step=100000)
        opening_theaters = st.number_input("Opening Theaters", value=2000, step=10)
        budget = st.number_input("Budget ($)", value=50000000, step=1000000)
        release_days = st.number_input("Release Days", value=90, step=1)

        st.subheader("🎭 Step 2: Select Genre")
        genre_options = ["Animation", "Action", "Horror", "Comedy", "Drama", "Thriller"]
        selected_genre = st.selectbox("Select Movie Genre", genre_options)


        st.subheader("📊 Step 3: Additional Financial Ratios")
        opening_revenue_ratio = st.number_input("Opening Revenue Ratio", value=0.1, step=0.01, format="%.2f")
        budget_to_revenue_ratio = st.number_input("Budget to Revenue Ratio", value=0.5, step=0.01, format="%.2f")
        domestic_to_world_ratio = st.number_input("Domestic to World Ratio", value=0.3, step=0.01, format="%.2f")
        revenue_per_theater = st.number_input("Revenue Per Theater", value=5000, step=100)

       
        st.subheader("🎟️ Step 4: Select MPAA Rating") 
        mpaa_rating = st.selectbox("MPAA Rating", ["G", "NC-17", "PG", "PG-13", "R"])
    
        

        st.subheader("🏢 Step 5: Select Distributor")
        distributor = st.selectbox(
             "Select Distributor", 
             ["Disney", "Paramount", "Sony", "Universal", "Warner Bros."]
        )
    
        submit_button = st.form_submit_button("🎥 Predict Box Office Revenue")

        # 🏗️ Prepare Input Data
        if submit_button:
            # Create input array in the exact order expected by the model
            input_data = np.array([
               domestic_revenue, opening_revenue, opening_theaters, budget, release_days,
               *(1 if genre == selected_genre else 0 for genre in genre_options),  # Corrected genre encoding
               opening_revenue_ratio, budget_to_revenue_ratio, domestic_to_world_ratio, revenue_per_theater,
               1 if mpaa_rating == "G" else 0, 1 if mpaa_rating == "NC-17" else 0, 1 if mpaa_rating == "PG" else 0,
               1 if mpaa_rating == "PG-13" else 0, 1 if mpaa_rating == "R" else 0,
               1 if distributor == "Disney" else 0, 1 if distributor == "Paramount" else 0,
               1 if distributor == "Sony" else 0, 1 if distributor == "Universal" else 0,
               1 if distributor == "Warner Bros." else 0
            ]).reshape(1, -1)

            # Scale input
            input_scaled = scaler.transform(input_data)

            # Predict
            prediction = model.predict(input_scaled)
    
            # Display Prediction
            st.success(f"🎬 Predicted Box Office Revenue: **${prediction[0]:,.2f}**")