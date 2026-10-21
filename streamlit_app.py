
from pathlib import Path
import gzip
import re

import pandas as pd
import streamlit as st
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIRS = [BASE_DIR / "Dataset", BASE_DIR / "dataset"]

st.set_page_config(
    page_title="HomeMate | Real Estate Analytics",
    page_icon="🏡",
    layout="wide",
)

st.title("🏡 HomeMate")
st.caption("Real Estate Analytics & Property Price Explorer")

@st.cache_data(show_spinner="Loading property dataset...")
def load_data():
    filenames = [
        "compressed_data.csv.gz",
        "HomeMate_residential.csv",
        "compressed_data.csv",
        "Makaan_Properties_Buy.csv",
    ]

    for folder in DATASET_DIRS:
        for filename in filenames:
            path = folder / filename
            if path.exists():
                try:
                    df = pd.read_csv(path, compression="infer", low_memory=False)
                    if not df.empty:
                        return df, filename
                except Exception:
                    continue

    return pd.DataFrame(), None


def find_column(df, candidates):
    lookup = {str(c).strip().lower(): c for c in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


df, source_file = load_data()

if df.empty:
    st.error("Dataset not found or could not be loaded.")
    st.info(
        "Check that your CSV file is inside the Dataset folder "
        "in your GitHub repository."
    )
    st.stop()

city_col = find_column(df, ["City_name", "city", "city_name"])
price_col = find_column(df, ["Price", "price", "price_in_rupees"])
size_col = find_column(df, ["Size", "size", "area"])
bhk_col = find_column(df, ["No_of_BHK", "BHK", "bhk"])
type_col = find_column(df, ["Property_type", "property_type", "type"])
status_col = find_column(df, ["Property_status", "property_status", "status"])
furnished_col = find_column(df, ["is_furnished", "furnished"])
ready_col = find_column(df, ["is_ready_to_move", "ready_to_move"])

if price_col:
    df[price_col] = pd.to_numeric(
        df[price_col].astype(str).str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False).str.strip(),
        errors="coerce",
    )

st.sidebar.header("🔎 Property Filters")
st.sidebar.caption(f"Dataset: {source_file}")

filtered = df.copy()

if city_col:
    cities = sorted(filtered[city_col].dropna().astype(str).unique())
    selected_cities = st.sidebar.multiselect(
        "Select cities", cities, default=cities
    )
    filtered = filtered[
        filtered[city_col].astype(str).isin(selected_cities)
    ]

if type_col:
    types = sorted(filtered[type_col].dropna().astype(str).unique())
    selected_types = st.sidebar.multiselect(
        "Property type", types, default=types
    )
    filtered = filtered[
        filtered[type_col].astype(str).isin(selected_types)
    ]

if bhk_col:
    bhks = sorted(
        filtered[bhk_col].dropna().astype(str).unique()
    )
    selected_bhks = st.sidebar.multiselect(
        "BHK", bhks, default=bhks
    )
    filtered = filtered[
        filtered[bhk_col].astype(str).isin(selected_bhks)
    ]

tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard", "🏙️ City Comparison", "🔍 Explore Data"]
)

with tab1:
    st.subheader("Property Market Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Properties", f"{len(filtered):,}")
    c2.metric(
        "Cities",
        f"{filtered[city_col].nunique():,}" if city_col else "N/A",
    )
    c3.metric(
        "Median Price",
        f"₹{filtered[price_col].median():,.0f}"
        if price_col and filtered[price_col].notna().any()
        else "N/A",
    )
    c4.metric(
        "Average Price",
        f"₹{filtered[price_col].mean():,.0f}"
        if price_col and filtered[price_col].notna().any()
        else "N/A",
    )

    left, right = st.columns(2)

    with left:
        if city_col and price_col:
            city_prices = (
                filtered.dropna(subset=[city_col, price_col])
                .groupby(city_col, as_index=False)[price_col]
                .median()
                .sort_values(price_col, ascending=False)
            )
            if not city_prices.empty:
                fig = px.bar(
                    city_prices,
                    x=city_col,
                    y=price_col,
                    title="Median Property Price by City",
                    labels={price_col: "Median Price (₹)", city_col: "City"},
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("City or price column was not found.")

    with right:
        if type_col:
            counts = (
                filtered[type_col].fillna("Unknown")
                .astype(str).value_counts()
                .rename_axis("Property Type")
                .reset_index(name="Count")
            )
            fig = px.pie(
                counts,
                names="Property Type",
                values="Count",
                title="Property Type Distribution",
                hole=0.4,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Property type column was not found.")

    if price_col:
        valid_prices = filtered[price_col].dropna()
        if not valid_prices.empty:
            fig = px.histogram(
                valid_prices.to_frame(),
                x=price_col,
                nbins=40,
                title="Property Price Distribution",
                labels={price_col: "Price (₹)"},
            )
            st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Compare Cities")

    if city_col and price_col:
        comparison = (
            filtered.dropna(subset=[city_col, price_col])
            .groupby(city_col)[price_col]
            .agg(["count", "median", "mean", "min", "max"])
            .reset_index()
            .rename(columns={
                city_col: "City",
                "count": "Properties",
                "median": "Median Price (₹)",
                "mean": "Average Price (₹)",
                "min": "Minimum Price (₹)",
                "max": "Maximum Price (₹)",
            })
            .sort_values("Median Price (₹)", ascending=False)
        )

        st.dataframe(
            comparison,
            use_container_width=True,
            hide_index=True,
        )

        fig = px.bar(
            comparison,
            x="City",
            y="Median Price (₹)",
            title="City-wise Median Price",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Dataset needs both city and price columns.")

with tab3:
    st.subheader("Explore Properties")

    search = st.text_input("Search dataset", placeholder="Enter a keyword...")
    view = filtered

    if search.strip():
        mask = view.astype(str).apply(
            lambda col: col.str.contains(
                re.escape(search.strip()), case=False, na=False
            )
        ).any(axis=1)
        view = view[mask]

    st.write(f"Showing {min(len(view), 1000):,} of {len(view):,} matching rows")
    st.dataframe(
        view.head(1000),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = view.head(10000).to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download CSV (up to 10,000 rows)",
        data=csv_data,
        file_name="homemate_properties.csv",
        mime="text/csv",
    )

st.divider()
st.caption("HomeMate | Real Estate Analytics")
