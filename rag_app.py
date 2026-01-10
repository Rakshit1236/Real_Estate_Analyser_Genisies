import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import os
try:
    import google.generativeai as genai_modern
except Exception:
    genai_modern = None

try:
    import google.generativeai as genai_legacy
except Exception:
    genai_legacy = None

CSV_FILE = "real_estate_data.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_GEMINI = "gemini-1.5"
TOP_K = 3

class RealEstateRAG:
    def __init__(self, csv_file, model_name=MODEL_NAME):
        """Initialize the RAG system."""
        print("Loading data...")
        try:
            self.df = pd.read_csv(csv_file)
            print(f"Loaded {len(self.df)} records.")
        except FileNotFoundError:
            print(f"Error: {csv_file} not found. Please run the scraper first.")
            exit(1)

        print("Initializing embedding model (this may take a moment first time)...")
        self.model = SentenceTransformer(model_name)
        
        print("Creating embeddings and index...")
        self.index, self.texts = self._build_index()
        print("System ready!")

    def _build_index(self):
        """Create FAISS index from dataframe."""
        texts = []
        for _, row in self.df.iterrows():
            bedrooms = int(row['Bedrooms']) if pd.notna(row['Bedrooms']) else '?'
            area = str(row['Area of Property']).split('.')[0] if pd.notna(row['Area of Property']) else 'Unknown'
            property_type = row['For Sale/Rent'] if pd.notna(row['For Sale/Rent']) else 'Unknown'
            
            text = (
                f"{row['Property']} - "
                f"₹{row['Price']} - "
                f"{bedrooms} BHK - "
                f"{area} sqft - "
                f"For {property_type}"
            )
            texts.append(text)

        #embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        
        # FAISS Index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings)
        
        return index, texts

    def retrieve(self, query, k=TOP_K):
        """Retrieve top k similar documents."""
        
        query_vector = self.model.encode([query], convert_to_numpy=True)

        k = min(k, len(self.texts)) if self.texts else k
        distances, indices = self.index.search(query_vector, k)
        
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.texts[idx])
        return results

    def generate_answer(self, query, k=TOP_K, api_key: str = None, model_id: str = None):
        """Generate answer using Google Gemini.

        Parameters:
        - query: user question
        - k: number of retrieved documents to include
        - api_key: optional Gemini API key (overrides GEMINI_API_KEY env var)
        """

        # 1. Retrieve context
        context_list = self.retrieve(query, k=k)
        context_str = "\n".join([f"- {text}" for text in context_list])

        if not context_str:
            return "I couldn't find any relevant properties in the database."

        # 2. Check for API Key (allow direct pass-through)
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            # If no API key, still return formatted results
            formatted_results = "Here are the top matching properties:\n\n"
            for i, text in enumerate(context_list, 1):
                formatted_results += f"{i}. {text}\n\n"
            return formatted_results

        # 3. Configure Gemini
        system_prompt = (
            "You are a helpful real estate assistant. Your job is to answer the user's real estate question "
            "based on the property listings provided. Format your answer as a concise, conversational response "
            "highlighting price, bedrooms and area for each recommended property."
        )

        user_prompt = f"{system_prompt}\n\nAvailable Properties:\n{context_str}\n\nUser's Question: {query}\n\nAnswer:"

    
        model_to_use = model_id or MODEL_GEMINI

        def _format_fallback(err_msg: str):
            formatted_results = f"Could not call Gemini/GenAI (Error: {err_msg}). Here are the top matching properties:\n\n"
            for i, text in enumerate(context_list, 1):
                formatted_results += f"{i}. {text}\n\n"
            return formatted_results

        # modern google.genai client
        try:
            if genai_modern is not None:
                genai_modern.configure(api_key=api_key)
                # try common patterns
                try:
                    # Prefer a direct generate_text call if available
                    resp = genai_modern.generate_text(model=model_to_use, prompt=user_prompt)
                    text = getattr(resp, 'text', None) or (resp.candidates[0].content if getattr(resp, 'candidates', None) else str(resp))
                    return text
                except Exception as e:
                    legacy_err = str(e)
            else:
                legacy_err = "modern client not installed"
        except Exception as e:
            legacy_err = str(e)

        # Try legacy google.generativeai 
        try:
            if genai_legacy is not None:
                genai_legacy.configure(api_key=api_key)
                try:
                    model = genai_legacy.GenerativeModel(model_to_use)
                    resp = model.generate(user_prompt)
                    text = getattr(resp, 'text', None) or str(resp)
                    return text
                except Exception as e:
                    # try generate_content fallback
                    try:
                        resp2 = genai_legacy.GenerativeModel(model_to_use).generate_content(user_prompt)
                        text = getattr(resp2, 'text', None) or str(resp2)
                        return text
                    except Exception as e2:
                        # If model not found, attempt to list models (best-effort)
                        err_msg = str(e2)
                        try:
                            if genai_modern is not None:
                                # try modern list models
                                models = None
                                try:
                                    models = genai_modern.list_models()
                                except Exception:
                                    try:
                                        models = genai_modern.Models.list()
                                    except Exception:
                                        models = None

                                if models:
                                    # extract model ids (best-effort)
                                    model_names = []
                                    for m in (getattr(models, 'models', None) or models or []):
                                        model_names.append(getattr(m, 'name', str(m)))
                                    sample = '\n'.join(model_names[:20])
                                    return f"Model '{model_to_use}' not found. Available models (sample):\n{sample}"
                        except Exception:
                            pass

                        return _format_fallback(err_msg[:200])
            else:
                # modern not available and legacy failed
                # try to help by listing models from modern client if present
                try:
                    if genai_modern is not None:
                        models = None
                        try:
                            models = genai_modern.list_models()
                        except Exception:
                            try:
                                models = genai_modern.Models.list()
                            except Exception:
                                models = None

                        if models:
                            model_names = []
                            for m in (getattr(models, 'models', None) or models or []):
                                model_names.append(getattr(m, 'name', str(m)))
                            sample = '\n'.join(model_names[:20])
                            return f"Model '{model_to_use}' not found. Available models (sample):\n{sample}"
                except Exception:
                    pass

                return _format_fallback(legacy_err)
        except Exception as e:
            return _format_fallback(str(e)[:200])

    def buy_rent_analysis(self, context_list):
        """Provide a concise buy vs rent analysis for the given list of property summary texts.

        This uses columns from the CSV when available (e.g. Final Wealth - Buy (INR),
        Final Wealth - Rent (INR), Initial Monthly Rent (INR), Numeric Price (INR),
        Monthly EMI (INR), Recommendation (Buy/Rent)).
        """
        if not context_list:
            return "No properties provided for analysis."

        lines = []
        for text in context_list:
            # Extract property name
            name = text.split(" - ₹")[0].strip()

            # Try to find matching row in dataframe
            row = None
            try:
                matches = self.df[self.df['Property'] == name]
                if len(matches) == 0:
                    matches = self.df[self.df['Property'].str.contains(name.split(',')[0], na=False, case=False)]
                if len(matches) > 0:
                    row = matches.iloc[0]
            except Exception:
                row = None

            if row is None:
                lines.append(f"{name}: No detailed financial data available.")
                continue

            # Safely read fields if they exist
            price = row.get('Numeric Price (INR)', None)
            rent = row.get('Initial Monthly Rent (INR)', None)
            emi = row.get('Monthly EMI (INR)', None)
            final_buy = row.get('Final Wealth - Buy (INR)', None)
            final_rent = row.get('Final Wealth - Rent (INR)', None)
            rec = row.get('Recommendation (Buy/Rent)', None)

            # Build concise analysis for this property
            parts = [f"Property: {name}"]
            if price is not None:
                try:
                    parts.append(f"Price: ₹{int(price):,}")
                except Exception:
                    parts.append(f"Price: {price}")
            if rent is not None and not pd.isna(rent):
                try:
                    parts.append(f"Monthly Rent: ₹{int(rent):,}")
                except Exception:
                    parts.append(f"Monthly Rent: {rent}")
            if emi is not None and not pd.isna(emi):
                try:
                    parts.append(f"Estimated Monthly EMI: ₹{int(emi):,}")
                except Exception:
                    parts.append(f"Estimated Monthly EMI: {emi}")

            # Compare final wealth if available
            rationale = None
            if final_buy is not None and final_rent is not None and not pd.isna(final_buy) and not pd.isna(final_rent):
                try:
                    fb = float(final_buy)
                    fr = float(final_rent)
                    diff = fb - fr
                    pct = (diff / fr) * 100 if fr != 0 else 0
                    if diff > 0:
                        rationale = f"Buy appears better: final wealth higher by ₹{int(diff):,} ({pct:.1f}%)."
                    elif diff < 0:
                        rationale = f"Rent appears better: final wealth higher by ₹{int(-diff):,} ({abs(pct):.1f}%)."
                    else:
                        rationale = "Buy and Rent result in similar final wealth over the horizon."
                except Exception:
                    rationale = None

            # CSV recommendation
            if rec and not pd.isna(rec):
                parts.append(f"CSV Recommendation: {rec}")

            if rationale:
                parts.append(rationale)

            lines.append(" | ".join(parts))

        # buy rent recc
        buy_count = sum(1 for l in lines if 'Buy appears better' in l or 'CSV Recommendation: Buy' in l)
        rent_count = sum(1 for l in lines if 'Rent appears better' in l or 'CSV Recommendation: Rent' in l)
        summary = f"Summary: {buy_count} recommend Buy, {rent_count} recommend Rent out of {len(lines)} properties."

        return summary + "\n\n" + "\n".join(lines)

def main():
    rag = RealEstateRAG(CSV_FILE)
    
    print("\n--- Real Estate AI Assistant ---")
    print("Ask a question (or type 'exit' to quit).")
    print("Example: 'Find me a 3 BHK flat for sale under 1.5 Cr'")
    
    while True:
        query = input("\nYou: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        answer = rag.generate_answer(query)
        print(f"AI: {answer}")

if __name__ == "__main__":
    main()
