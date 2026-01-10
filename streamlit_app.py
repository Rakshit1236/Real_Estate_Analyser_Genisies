import os
import re
import pandas as pd
import streamlit as st

from scraper import (
    scrape_listings,
    save_to_csv,
    BASE_URL_SALE,
    BASE_URL_RENT,
    OUTPUT_FILE,
)
from rag_app import RealEstateRAG, CSV_FILE


def _get_csv_mtime(path: str) -> float:
    """Return modification time of the CSV, or 0.0 if it doesn't exist.

    This is used to invalidate the cached RAG object when the CSV changes.
    """
    try:
        return os.path.getmtime(path)
    except OSError:
        return 0.0


def parse_area(area_str):
    """Parse area string to sqft float."""
    if not isinstance(area_str, str):
        return 0.0
    area_str = area_str.lower().strip()
    # Match number followed by unit
    match = re.search(r"([\d\.]+)\s*(sqft|sqyrd|sqm|sq m|sq ft|sq yard)", area_str)
    if not match:
        return 0.0
    val = float(match.group(1))
    unit = match.group(2)
    
    if "sqyrd" in unit or "sq yard" in unit:
        return val * 9.0
    elif "sqm" in unit or "sq m" in unit:
        return val * 10.764
    return val


@st.cache_resource(show_spinner="Building RAG index (first time can take a bit)...")
def load_rag(csv_path: str, csv_mtime: float) -> RealEstateRAG:
    """Load the RAG system and build the FAISS index.

    csv_mtime is only used so that whenever the CSV file is overwritten,
    Streamlit will treat this as a new cache key and rebuild the index.
    """
    return RealEstateRAG(csv_path)


# --------- Streamlit UI ----------

st.set_page_config(page_title="Real Estate Scraper + Local RAG", layout="wide")

# --- Sidebar / Settings ---
st.sidebar.title("App Settings")
gemini_key = st.sidebar.text_input("Gemini API Key (optional)", type="password", help="Provide a Gemini API key to enable conversational answers from Gemini.")
top_k = st.sidebar.slider("Top-K results", 1, 10, 3)
show_csv_preview = st.sidebar.checkbox("Show CSV preview after scraping", value=True)
model_id = st.sidebar.text_input("Gemini Model ID (optional)", value="gemini-1.5", help="Model identifier to use for Gemini/GenAI clients. Leave default if unsure.")
st.sidebar.markdown("---")
st.sidebar.markdown("Example questions:\n- Find me a 3 BHK flat for rent under 1 Cr\n- Show 2 BHK apartments for sale near Surat")

st.markdown("# Real Estate Scraper")
st.markdown(
    "This app lets you:\n1. Scrape real-estate listings from MagicBricks-like URLs and save them into a CSV.\n2. Run a **local** semantic search + RAG on that CSV using SentenceTransformers + FAISS."
)
st.markdown("---")

scrape_tab, rag_tab, analysis_tab = st.tabs(["1. Scrape Data", "2. Ask Questions (RAG)", "3. Analysis"])


# --------- Scrape Data ----------

with scrape_tab:
    st.header("Scrape Listings")

    st.write("Provide base URLs for **Sale** and/or **Rent** listings.")
    st.write(
        "You can paste MagicBricks search URLs or similar listing pages. "
        "The scraper will iterate a few pages and save results to the CSV."
    )

    col1, col2 = st.columns(2)

    with col1:
        sale_url = st.text_input(
            "Sale listings URL (optional)",
            value=BASE_URL_SALE,
            help="Leave blank if you don't want to scrape sale listings.",
        )

    with col2:
        rent_url = st.text_input(
            "Rent listings URL (optional)",
            value=BASE_URL_RENT,
            help="Leave blank if you don't want to scrape rent listings.",
        )

    if st.button("Scrape and Save to CSV"):
        if not sale_url and not rent_url:
            st.error("Please provide at least one URL (Sale or Rent).")
        else:
            all_listings = []

            with st.spinner("Scraping data. This may take a few minutes..."):
                if sale_url:
                    st.write("Scraping SALE listings...")
                    sale_listings = scrape_listings(sale_url, "Sale")
                    all_listings.extend(sale_listings)

                if rent_url:
                    st.write("Scraping RENT listings...")
                    rent_listings = scrape_listings(rent_url, "Rent")
                    all_listings.extend(rent_listings)

                if all_listings:
                    save_to_csv(all_listings)
                    st.success(f"Scraped and saved {len(all_listings)} listings to {OUTPUT_FILE}.")

                    if show_csv_preview:
                        try:
                            df = pd.read_csv(OUTPUT_FILE)
                            st.subheader("Sample of scraped data")
                            st.dataframe(df.head(20))
                        except Exception as e:
                            st.warning(f"Data saved but could not preview CSV: {e}")
                else:
                    st.warning("No listings scraped. Please check the URLs or try again.")


# ---------  RAG via ----------

with rag_tab:
    st.header("Ask Questions About the Scraped Properties")

    if not os.path.exists(CSV_FILE):
        st.info(
            f"CSV file '{CSV_FILE}' not found. Please run the scraper in the first tab to generate data before using the RAG assistant."
        )
    else:
        csv_mtime = _get_csv_mtime(CSV_FILE)

        # Build or load cached RAG object
        rag = load_rag(CSV_FILE, csv_mtime)

        st.success(f"Loaded data from '{CSV_FILE}'. You can now ask questions.")

        query = st.text_input(
            "Ask a question (e.g. 'Find me a 3 BHK flat for rent under 1 Cr')",
            value="",
        )

        if st.button("Search and Generate Answer"):
            if not query.strip():
                st.error("Please enter a question.")
            else:
                with st.spinner("Retrieving most relevant properties..."):
                    # Show retrieved items first
                    context_list = rag.retrieve(query.strip(), k=top_k)

                    if context_list:
                        st.subheader("Top matching properties")
                        for i, prop in enumerate(context_list, 1):
                            with st.expander(f"{i}. {prop}"):
                                st.write(prop)

                        # Provide buy vs rent analysis for the retrieved properties (guarded)
                        if hasattr(rag, "buy_rent_analysis"):
                            analysis = rag.buy_rent_analysis(context_list)
                        else:
                            analysis = "Buy vs Rent analysis not available (method missing on RAG object)."
                        st.subheader("Buy vs Rent Analysis")
                        st.text_area("Analysis", value=analysis, height=300)
                    else:
                        st.info("No matching properties found.")

                    # Generate answer (if API key provided in sidebar use it)
                    answer = rag.generate_answer(query.strip(), k=top_k, api_key=gemini_key, model_id=model_id)

                st.subheader("Answer")
                st.text_area("Result", value=answer, height=300)

        st.caption(
            "Note: If you provide a Gemini API key in the sidebar, answers will be generated by Gemini; otherwise a local formatted result is shown."
        )


# --------- Analysis Tab ----------

with analysis_tab:
    st.header("Data Analysis")
    
    if not os.path.exists(OUTPUT_FILE):
        st.info("No data found. Please scrape some data first.")
    else:
        try:
            df = pd.read_csv(OUTPUT_FILE)
            if df.empty:
                st.warning("CSV file is empty.")
            else:
                # Preprocessing
                df["Normalized Area (sqft)"] = df["Area of Property"].apply(parse_area)
                
                # Metrics
                total_listings = len(df)
                
                # Filter for Sale vs Rent
                df_sale = df[df["For Sale/Rent"] == "Sale"]
                df_rent = df[df["For Sale/Rent"] == "Rent"]
                
                avg_price_sale = df_sale["Numeric Price (INR)"].mean() if not df_sale.empty else 0
                avg_rent = df_rent["Numeric Price (INR)"].mean() if not df_rent.empty else 0
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Listings", total_listings)
                col2.metric("Avg Sale Price", f"₹{avg_price_sale:,.0f}")
                col3.metric("Avg Monthly Rent", f"₹{avg_rent:,.0f}")
                
                st.markdown("---")
                
                # Charts
                st.subheader("1. Price vs Area Analysis")
                st.markdown("This scatter plot helps identify value-for-money properties. Points lower and to the right are generally better value (lower price for larger area).")
                
                # Use st.scatter_chart (requires Streamlit 1.29+)
                # We plot only Sale properties for price analysis to avoid skewing with rent prices
                if not df_sale.empty:
                    st.scatter_chart(
                        df_sale,
                        x="Normalized Area (sqft)",
                        y="Numeric Price (INR)",
                        color="Bedrooms",
                        size="Bedrooms",
                    )
                else:
                    st.info("No sale listings to plot Price vs Area.")

                st.markdown("---")

                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.subheader("2. Property Type Distribution")
                    type_counts = df["For Sale/Rent"].value_counts()
                    st.bar_chart(type_counts)
                
                with col_b:
                    st.subheader("3. Bedroom Configuration")
                    bed_counts = df["Bedrooms"].value_counts()
                    st.bar_chart(bed_counts)

                st.markdown("---")
                
                st.subheader("4. Buy vs Rent Recommendations")
                st.markdown("Based on the 20-year financial analysis (for Sale properties):")
                
                if "Recommendation (Buy/Rent)" in df.columns:
                    rec_counts = df["Recommendation (Buy/Rent)"].value_counts()
                    st.bar_chart(rec_counts, color="#ffaa00") 
                else:
                    st.info("Recommendation data not available.")

        except Exception as e:
            st.error(f"Error loading or processing data: {e}")
