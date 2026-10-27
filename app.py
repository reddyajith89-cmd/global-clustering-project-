# Global Clustering Project - Streamlit App
# How to run:  streamlit run app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------
# Step 1: Page title
# ---------------------------------------------------
st.set_page_config(page_title="Global Clustering Project", layout="wide")
st.title("Global Clustering Project")
st.write("See how the GDP of each country changed every year.")

# ---------------------------------------------------
# Step 2: Load the data
# (country_results.csv is created by the last steps of the notebook)
# ---------------------------------------------------
try:
    data = pd.read_csv("country_results.csv")
except FileNotFoundError:
    st.error("country_results.csv is not found. Please run the notebook first.")
    st.stop()

# GDP is very big, so we show it in Billion dollars
data["GDP (Billion $)"] = data["GDP"] / 1000000000

all_countries = sorted(data["Country"].unique())
all_years = sorted(data["Year"].unique())

# ---------------------------------------------------
# Step 3: Left side (sidebar) - select countries
# ---------------------------------------------------
st.sidebar.header("Select Countries")

# Show these 3 countries first (only if they are in the data)
first_countries = [c for c in ["India", "China", "United States"] if c in all_countries]

selected_countries = st.sidebar.multiselect(
    "Choose one or more countries",
    all_countries,
    default=first_countries
)

what_to_show = st.sidebar.radio(
    "What do you want to see?",
    ["GDP (Billion $)", "GDP Growth (% per year)"]
)

# ---------------------------------------------------
# Step 4: Graph - every country has a different colour
# ---------------------------------------------------
st.subheader("Country growth for every year")

if len(selected_countries) == 0:
    st.info("Please select at least one country from the left side.")
else:
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, name in enumerate(selected_countries):

        # Take only this country's rows
        one_country = data[data["Country"] == name].sort_values("Year")

        if what_to_show == "GDP (Billion $)":
            y_values = one_country["GDP (Billion $)"]
        else:
            # Growth % = (this year - last year) / last year * 100
            y_values = one_country["GDP (Billion $)"].pct_change(fill_method=None) * 100

        # Different colour for each country
        ax.plot(one_country["Year"], y_values, marker="o",
                label=name, color=plt.cm.tab20(i % 20))

    ax.set_xlabel("Year")
    ax.set_ylabel(what_to_show)
    ax.set_title(what_to_show + " of selected countries")
    ax.set_xticks(all_years)
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)

# ---------------------------------------------------
# Step 5: Type a country name - see GDP of every year and month
# ---------------------------------------------------
st.subheader("Type a country name to see its GDP")

typed_name = st.text_input("Country name", value="India")

if typed_name.strip() == "":
    st.info("Please type a country name.")
else:
    # Find the countries which have the typed text in their name
    found = data[data["Country"].str.lower().str.contains(typed_name.strip().lower(), regex=False)]

    if len(found) == 0:
        st.warning("No country found. Please check the spelling.")
    else:
        table = found[["Country", "Year", "GDP (Billion $)", "Cluster_KMeans"]].copy()

        # The data has only yearly GDP, so month GDP = yearly GDP / 12
        table["GDP per Month (Billion $)"] = table["GDP (Billion $)"] / 12

        table = table.sort_values(["Country", "Year"]).round(2)
        table = table.rename(columns={"Cluster_KMeans": "KMeans Cluster"})

        st.dataframe(table, hide_index=True)
        st.caption("The data has only yearly GDP. GDP per Month = yearly GDP divided by 12 (average). "
                   "An empty GDP means the data is not available for that year.")
