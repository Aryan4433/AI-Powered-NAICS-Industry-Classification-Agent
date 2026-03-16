# ============================================================
# 🌐 Streamlit NAICS Classification Agent 
# ============================================================

import streamlit as st
import pandas as pd
import json
import os
import re
from io import BytesIO
from dotenv import load_dotenv
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
    page_title="NAICS Classification Agent",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 NAICS Classification Agent")
st.markdown("Upload organizations and classify NAICS codes automatically.")

# ------------------------------------------------------------
# 3. Load NAICS Dataset
# ------------------------------------------------------------
@st.cache_data
def load_naics():

    df = pd.read_excel("naics_code.xlsx")
    df.columns = df.columns.str.strip()

    return df

df_naics = load_naics()

st.success(f"✅ Loaded {len(df_naics)} NAICS codes")

# ------------------------------------------------------------
# 4. Initialize LLM
# ------------------------------------------------------------
chat = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=300,
    api_key=GROQ_API_KEY
)

search_tool = DuckDuckGoSearchRun()

# ------------------------------------------------------------
# 5. Clean Company Name
# ------------------------------------------------------------
def clean_company_name(name):

    suffixes = [
        "llc","inc","ltd","corp","corporation",
        "company","co","holdings","group"
    ]

    name = name.lower()

    for s in suffixes:
        name = name.replace(s,"")

    return name.strip()

# ------------------------------------------------------------
# 6. Get Company Summary
# ------------------------------------------------------------
def get_company_summary(org_name, address, country):

    org_clean = clean_company_name(org_name)

    query = f"""
    {org_clean} company industry
    what does {org_clean} company do
    {address} {country}
    """

    try:
        result = search_tool.run(query)
        return result[:1500]

    except:
        return ""

# ------------------------------------------------------------
# 7. NAICS Candidate Filtering
# ------------------------------------------------------------
def get_candidate_naics(summary):

    summary = summary.lower()

    df_temp = df_naics.copy()

    df_temp["score"] = df_temp["Description"].str.lower().apply(
        lambda x: sum(word in x for word in summary.split())
    )

    candidates = df_temp.sort_values("score", ascending=False).head(30)

    return candidates[["Naics code","Description"]].to_dict(orient="records")

# ------------------------------------------------------------
# 8. NAICS Prediction
# ------------------------------------------------------------
def predict_naics(org_name, address, country):

    summary = get_company_summary(org_name, address, country)

    if summary == "":
        summary = org_name

    candidates = get_candidate_naics(summary)

    prompt = f"""
You are an expert industry classification analyst.

Determine the best NAICS code for the company.

Company Name:
{org_name}

Address:
{address}

Country:
{country}

Business Summary:
{summary}

Candidate NAICS Codes:
{json.dumps(candidates, indent=2)}

Select the MOST appropriate NAICS code.

Return JSON ONLY:

{{
"naics_code":"XXXXX",
"description":"NAICS description",
"confidence":"HIGH | MEDIUM | LOW"
}}
"""

    try:

        response = chat.invoke([HumanMessage(content=prompt)])

        raw_output = response.content.strip()

        match = re.search(r"\{.*\}", raw_output, re.DOTALL)

        result = json.loads(match.group())

        code = result.get("naics_code")
        description = result.get("description")
        confidence = result.get("confidence")

    except:

        code = None
        description = None
        confidence = "LOW"

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

# ------------------------------------------------------------
# Input Validation
# ------------------------------------------------------------
    required_columns = ["Org Name","Address"]

    missing_cols = [col for col in required_columns if col not in df_input.columns]

    if missing_cols:
        st.error(f"❌ Input must contain columns: {', '.join(required_columns)}")
        st.stop()

    df_input["Country"] = df_input.get("Country","")

# ------------------------------------------------------------
# 10. Run Classification
# ------------------------------------------------------------
    if st.button("🚀 Classify NAICS"):

        results = []

        progress_bar = st.progress(0)

        for i,row in df_input.iterrows():

            code, desc, conf = predict_naics(
                row["Org Name"],
                row["Address"],
                row["Country"]
            )

            results.append({

                "Org Name":row["Org Name"],
                "Address":row["Address"],
                "Country":row["Country"],
                "NAICS Code":code,
                "NAICS Description":desc,
                "Confidence":conf
            })

            progress_bar.progress((i+1)/len(df_input))

        df_results = pd.DataFrame(results)

        st.success("✅ Classification Completed")

        st.dataframe(df_results)

# ------------------------------------------------------------
# 11. Download Output
# ------------------------------------------------------------
        def to_excel(df):

            output = BytesIO()
            df.to_excel(output,index=False)
            output.seek(0)

            return output

        st.download_button(
            label="📥 Download Excel",
            data=to_excel(df_results),
            file_name="naics_classified_output.xlsx"
        )