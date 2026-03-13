# ============================================================
# 🌐 Streamlit Batch NAICS Enrichment Agent
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import faiss
import json
import os
import re
from io import BytesIO
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
# from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun

# ------------------------------------------------------------
# 1. Load Environment Variables
# ------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# ------------------------------------------------------------
# 2. Page Setup
# ------------------------------------------------------------
st.set_page_config(page_title="Batch NAICS Enrichment Agent", page_icon="🌐", layout="wide")
st.title("🌐 Batch NAICS Enrichment Agent (RAG + ChatGroq)")
st.markdown("### Upload a file of organizations to predict their NAICS 2022 codes using AI + Web Search + RAG")

# ------------------------------------------------------------
# 3. Load NAICS Dataset
# ------------------------------------------------------------
@st.cache_data(show_spinner=True)
def load_naics_data():
    df = pd.read_excel("naics_code.xlsx")
    df.columns = df.columns.str.strip()
    return df

df_naics = load_naics_data()
st.success(f"✅ Loaded {len(df_naics)} NAICS records.")

# ------------------------------------------------------------
# 4. Prepare FAISS + Embeddings
# ------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def setup_retriever():
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    descriptions = df_naics["Description"].tolist()
    embeddings = embed_model.encode(descriptions, convert_to_numpy=True).astype(np.float32)

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return embed_model, index

embed_model, index = setup_retriever()
st.info("✅ Embedding model and FAISS index ready.")

# ------------------------------------------------------------
# 5. Initialize LLM and Search Tool
# ------------------------------------------------------------
search_tool = DuckDuckGoSearchRun()
chat = ChatGroq(model="llama-3.1-8b-instant", temperature=0.2, api_key=GROQ_API_KEY)

# ------------------------------------------------------------
# 6. NAICS Prediction Function (Single Row)
# ------------------------------------------------------------
def predict_naics_agent(org_name, org_description="", country="", top_k=5):
    query = f"{org_name} company profile {country}"
    try:
        web_info = search_tool.run(query)
    except Exception as e:
        web_info = ""
        st.warning(f"⚠️ Web search failed: {e}")

    query_text = f"{org_name} {org_description} {country} {web_info[:1000]}"
    query_vec = embed_model.encode([query_text], convert_to_numpy=True).astype(np.float32)

    D, I = index.search(query_vec, top_k)
    retrieved = df_naics.iloc[I[0]][["Naics code", "Description"]].to_dict(orient="records")
    retrieved_json = json.dumps(retrieved, indent=2)

    prompt = f"""
You are an expert NAICS classification analyst.
Analyze the company's main business and select the best matching NAICS 2022 code.

Organization Name: {org_name}
Country: {country}
Organization Description: {org_description}

Web Search Summary:
{web_info}

Candidate NAICS Codes (2022 version):
{retrieved_json}

Instructions:
1. Analyze the organization’s primary line of business.
2. Select ONE NAICS code ONLY from the candidate list.
3. Respond strictly in this JSON format:
{{
  "naics_code": "XXXXX",
  "description": "..."
}}
"""

    response = chat.invoke([HumanMessage(content=prompt)])
    raw_output = response.content.strip()

    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group())
        except json.JSONDecodeError:
            result = {"naics_code": None, "description": raw_output}
    else:
        result = {"naics_code": None, "description": raw_output}

    return result

# ------------------------------------------------------------
# 7. File Upload
# ------------------------------------------------------------
uploaded_file = st.file_uploader("📂 Upload Excel/CSV file with organizations", type=["xlsx","csv"])
if uploaded_file:
    if uploaded_file.name.endswith(".xlsx"):
        df_input = pd.read_excel(uploaded_file)
    else:
        df_input = pd.read_csv(uploaded_file)
    
    st.success(f"✅ Loaded {len(df_input)} organizations for enrichment.")

    # Check required columns
    required_cols = ["Org Name"]
    for col in required_cols:
        if col not in df_input.columns:
            st.error(f"❌ Missing required column: {col}")
            st.stop()

    # Optional: fill missing columns
    df_input["Description"] = df_input.get("Description", "")
    df_input["Country"] = df_input.get("Country", "")

    # ------------------------------------------------------------
    # 8. Batch Prediction
    # ------------------------------------------------------------
    if st.button("🚀 Enrich Organizations"):
        results = []
        progress_bar = st.progress(0)
        status_text = st.empty()

        for idx, row in df_input.iterrows():
            status_text.text(f"Processing {idx+1}/{len(df_input)}: {row['Org Name']}")
            res = predict_naics_agent(row["Org Name"], row.get("Description",""), row.get("Country",""))
            results.append({
                "Org Name": row["Org Name"],
                "Description": row.get("Description",""),
                "Country": row.get("Country",""),
                "NAICS Code": res.get("naics_code"),
                "NAICS Description": res.get("description")
            })
            progress_bar.progress((idx+1)/len(df_input))

        st.success("✅ Enrichment Complete!")

        df_results = pd.DataFrame(results)
        st.dataframe(df_results)

        # ------------------------------------------------------------
        # 9. Download Enriched Excel
        # ------------------------------------------------------------
        def to_excel(df):
            output = BytesIO()
            df.to_excel(output, index=False)
            output.seek(0)
            return output

        st.download_button(
            label="📥 Download Enriched Excel",
            data=to_excel(df_results),
            file_name="enriched_organizations.xlsx"
        )
