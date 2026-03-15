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
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

# ------------------------------------------------------------
# 1. Load Environment Variables
# ------------------------------------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env")
    st.stop()

# ------------------------------------------------------------
# 2. Page Setup
# ------------------------------------------------------------
st.set_page_config(
    page_title="Batch NAICS Enrichment Agent",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Batch NAICS Enrichment Agent")
st.markdown("Upload organizations and automatically classify their NAICS codes.")

# ------------------------------------------------------------
# 3. Load NAICS Dataset
# ------------------------------------------------------------
@st.cache_data
def load_naics():

    df = pd.read_excel("naics_code.xlsx")
    df.columns = df.columns.str.strip()

    return df

df_naics = load_naics()

st.success(f"✅ Loaded {len(df_naics)} NAICS records")

# ------------------------------------------------------------
# 4. Build FAISS Index
# ------------------------------------------------------------
@st.cache_resource
def build_index():

    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    descriptions = df_naics["Description"].tolist()

    embeddings = embed_model.encode(
        descriptions,
        convert_to_numpy=True
    ).astype(np.float32)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return embed_model, index

embed_model, index = build_index()

st.info("Embedding model and FAISS index ready")

# ------------------------------------------------------------
# 5. Initialize LLM + Search Tool
# ------------------------------------------------------------
search_tool = DuckDuckGoSearchRun()

chat = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=200,
    api_key=GROQ_API_KEY
)

# ------------------------------------------------------------
# 6. Get Company Summary
# ------------------------------------------------------------
def get_company_summary(org_name, country):

    query = f"{org_name} company industry business what does {org_name} do {country}"

    try:
        result = search_tool.run(query)
        return result[:1200]
    except:
        return ""

# ------------------------------------------------------------
# 7. Confidence Logic
# ------------------------------------------------------------
def calculate_confidence(distance, rank):

    if distance < 0.35 and rank == 0:
        return "HIGH"

    elif distance < 0.75 and rank <= 2:
        return "MEDIUM"

    else:
        return "LOW"

# ------------------------------------------------------------
# 8. NAICS Prediction
# ------------------------------------------------------------
def predict_naics(org_name, description="", country=""):

    summary = get_company_summary(org_name, country)

    query_text = f"""
    {org_name}
    {description}
    {summary}
    """

    query_vec = embed_model.encode(
        [query_text],
        convert_to_numpy=True
    ).astype(np.float32)

    D, I = index.search(query_vec, 5)

    retrieved = df_naics.iloc[I[0]]

    candidates = retrieved[["Naics code","Description"]].to_dict(orient="records")

    prompt = f"""
You are an expert industry classification analyst.

Company:
{org_name}

Business Summary:
{summary}

Candidate NAICS Codes:
{json.dumps(candidates, indent=2)}

Select the BEST matching NAICS code.

Return JSON only:

{{
"naics_code":"XXXXX",
"description":"NAICS description"
}}
"""

    response = chat.invoke([HumanMessage(content=prompt)])

    raw_output = response.content.strip()

    try:

        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        result = json.loads(match.group())

        code = result.get("naics_code")
        description = result.get("description")

    except:
        return None, None, "LOW"

    rank = 4

    for idx, row in retrieved.iterrows():

        if str(row["Naics code"]) == str(code):

            rank = list(retrieved.index).index(idx)

    distance = float(D[0][0])

    confidence = calculate_confidence(distance, rank)

    return code, description, confidence

# ------------------------------------------------------------
# 9. Upload Input File
# ------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Excel or CSV file",
    type=["xlsx","csv"]
)

if uploaded_file:

    if uploaded_file.name.endswith(".xlsx"):
        df_input = pd.read_excel(uploaded_file)
    else:
        df_input = pd.read_csv(uploaded_file)

    df_input.columns = df_input.columns.str.strip()

    st.success(f"Loaded {len(df_input)} organizations")

    if "Org Name" not in df_input.columns:
        st.error("❌ Input file must contain 'Org Name' column")
        st.stop()

    df_input["Description"] = df_input.get("Description","")
    df_input["Country"] = df_input.get("Country","")

# ------------------------------------------------------------
# 10. Run Batch Enrichment
# ------------------------------------------------------------
    if st.button("🚀 Enrich Organizations"):

        results = []

        progress_bar = st.progress(0)

        for i,row in df_input.iterrows():

            code, desc, conf = predict_naics(
                row["Org Name"],
                row["Description"],
                row["Country"]
            )

            results.append({

                "Org Name":row["Org Name"],
                "Description":row["Description"],
                "Country":row["Country"],
                "NAICS Code":code,
                "NAICS Description":desc,
                "Confidence":conf
            })

            progress_bar.progress((i+1)/len(df_input))

        df_results = pd.DataFrame(results)

        st.success("✅ Enrichment Completed")

        st.dataframe(df_results)

# ------------------------------------------------------------
# 11. Download Results
# ------------------------------------------------------------
        def to_excel(df):

            output = BytesIO()
            df.to_excel(output,index=False)
            output.seek(0)

            return output

        st.download_button(
            label="📥 Download Enriched Excel",
            data=to_excel(df_results),
            file_name="naics_enriched_output.xlsx"
        )