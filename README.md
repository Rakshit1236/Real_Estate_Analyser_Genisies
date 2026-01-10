 # Genesis - Real Estate Scraper & AI Assistant

 Real Estate using RAG is a  comprehensive tool for scraping real estate listings, performing financial "Buy vs. Rent" analysis, and interacting with the data using an AI-powered RAG (Retrieval-Augmented Generation) system. Built with Streamlit, it offers a user-friendly interface to analyze property markets effectively.

## 🚀 Features

-   **Web Scraper**: Automated scraping of real estate listings (Sale & Rent) from major property portals (configured for MagicBricks style URLs).
-   **Financial Analysis Engine**:
    -   Calculates EMI schedules, tax savings (Section 24(b) & 80C), and property appreciation.
    -   Compares "Buying" vs. "Renting & Investing" strategies over a 20-year horizon.
    -   Provides clear recommendations: **Buy**, **Rent**, or **Neutral**.
-   **RAG AI Assistant**:
    -   **Semantic Search**: Uses `sentence-transformers` and `FAISS` to find relevant properties based on natural language queries (e.g., "Find affordable 2 BHK near station").
    -   **Generative AI**: Integrates with **Google Gemini** to generate human-like answers and summaries.
-   **Interactive Dashboard**:
    -   Data visualization (Price vs. Area, Bedroom distribution).
    -   CSV data preview and management.

## 📋 Prerequisites

-   **Python 3.8** or higher.
-   **Google Gemini API Key** (optional): Required for the AI generation features. Get it from [Google AI Studio](https://aistudio.google.com/).

## 🛠️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Rakshit1236/Real_Estate_Analyser_Genisies
```

### 2. Create a Virtual Environment
It is recommended to use a virtual environment to manage dependencies.

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
Install the required Python packages using `pip`:
```bash
pip install -r requirements.txt
```

*Note: If you encounter issues with `faiss-cpu`, ensure you have the C++ redistributables installed on Windows.*

## 🏃‍♂️ Usage

### Start the Application
Run the Streamlit app from your terminal:
```bash
streamlit run streamlit_app.py
```

The app will open in your default web browser at `http://localhost:8501`.

### Workflow
1.  **Scrape Data (Tab 1)**:
    -   Enter the Search URLs for **Sale** and **Rent** properties (defaults provided for Surat city).
    -   Click **"Scrape and Save to CSV"**.
    -   The system will scrape listings and calculate financial metrics automatically.

2.  **Ask Questions / RAG (Tab 2)**:
    -   (Optional) Enter your **Gemini API Key** in the sidebar for generative answers.
    -   Type a question like: *"Show me 3 BHK flats in Vesu under 1.5 Cr"* or *"Which properties have the best rental yield?"*.
    -   The system retrieves the most relevant listings and provides an AI-generated answer.

3.  **Analysis (Tab 3)**:
    -   View visual insights including Price vs. Area scatter plots, bedroom distributions, and Buy/Rent recommendation summaries.

## 📂 Project Structure

```
Genisis/
├── rag_app.py           # RAG logic (Embeddings, FAISS index, Gemini integration)
├── scraper.py           # Web scraping logic & Financial calculations
├── streamlit_app.py     # Main Streamlit application (UI entry point)
├── real_estate_data.csv # Storage for scraped data (generated)
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```


