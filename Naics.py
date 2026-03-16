# ============================================================
# 🌐 Smart NAICS Classification Agent (Interactive Version)
# ============================================================

import streamlit as st
import pandas as pd
import json
import os
import re
import time
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
    page_title="Smart NAICS Classification Agent",
    page_icon="🌐",
    layout="wide"
)

st.title("🌐 Smart NAICS Classification Agent")
st.markdown("Upload organizations and classify NAICS codes automatically.")

# ------------------------------------------------------------
# 3. Sidebar Settings
# ------------------------------------------------------------
st.sidebar.title("⚙️ Agent Settings")

show_summary = st.sidebar.checkbox("Show company summary", False)
show_candidates = st.sidebar.checkbox("Show NAICS candidates", False)

max_candidates = st.sidebar.slider(
    "Number of NAICS candidates",
    min_value=10,
    max_value=50,
    value=30
)

# ------------------------------------------------------------
# 4. Load NAICS Dataset
# ------------------------------------------------------------
@st.cache_data
def load_naics():

    df = pd.read_excel("naics_code.xlsx")
    df.columns = df.columns.str.strip()

    return df

df_naics = load_naics()

st.success(f"✅ Loaded {len(df_naics)} NAICS codes")

# ------------------------------------------------------------
# 5. Initialize LLM
# ------------------------------------------------------------
chat = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=400,
    api_key=GROQ_API_KEY
)

search_tool = DuckDuckGoSearchRun()

# ------------------------------------------------------------
# 6. Clean Company Name
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
# 7. Get Company Summary
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
# 8. Candidate NAICS Filtering
# ------------------------------------------------------------
def get_candidate_naics(summary):

    summary = summary.lower()

    df_temp = df_naics.copy()

    df_temp["score"] = df_temp["Description"].str.lower().apply(
        lambda x: sum(word in x for word in summary.split())
    )

    candidates = df_temp.sort_values(
        "score",
        ascending=False
    ).head(max_candidates)

    return candidates[["Naics code","Description"]].to_dict(orient="records")

# ------------------------------------------------------------
# 9. Predict NAICS
# ------------------------------------------------------------
def predict_naics(org_name, address, country):

    summary = get_company_summary(org_name,address,country)

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
{json.dumps(candidates,indent=2)}

Select the MOST appropriate NAICS code.

Return JSON ONLY:

{{
"naics_code":"XXXXX",
"description":"NAICS description",
"confidence":"HIGH | MEDIUM | LOW",
"reasoning":"Why this NAICS code was selected"
}}
"""

    for attempt in range(2):

        try:

            response = chat.invoke(
                [HumanMessage(content=prompt)]
            )

            raw_output = response.content.strip()

            match = re.search(r"\{.*\}",raw_output,re.DOTALL)

            result = json.loads(match.group())

            return (
                result.get("naics_code"),
                result.get("description"),
                result.get("confidence"),
                result.get("reasoning"),
                summary,
                candidates
            )

        except:
            continue

    return None,None,"LOW","LLM parsing failed",summary,candidates

# ------------------------------------------------------------
# 10. Single Company Lookup
# ------------------------------------------------------------
st.sidebar.subheader("🔎 Quick Company Lookup")

company = st.sidebar.text_input("Company Name")
address = st.sidebar.text_input("Address")
country = st.sidebar.text_input("Country")

if st.sidebar.button("Classify Company"):

    code,desc,conf,reason,summary,candidates = predict_naics(
        company,address,country
    )

    st.sidebar.write("### Result")

    st.sidebar.write("NAICS Code:",code)
    st.sidebar.write("Description:",desc)
    st.sidebar.write("Confidence:",conf)

# ------------------------------------------------------------
# 11. Upload Input File
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
# Data Preview
# ------------------------------------------------------------
    st.subheader("📄 Uploaded Data Preview")
    st.dataframe(df_input.head(10))

# ------------------------------------------------------------
# Input Validation
# ------------------------------------------------------------
    required_columns = ["Org Name","Address"]

    missing_cols = [
        col for col in required_columns
        if col not in df_input.columns
    ]

    if missing_cols:

        st.error(
            f"❌ Input must contain columns: {', '.join(required_columns)}"
        )

        st.stop()

    df_input["Country"] = df_input.get("Country","")

# ------------------------------------------------------------
# 12. Run Classification
# ------------------------------------------------------------
    if st.button("🚀 Classify NAICS"):

        start_time = time.time()

        results = []

        progress_bar = st.progress(0)

        status = st.empty()

        for i,row in df_input.iterrows():

            status.text(f"Processing: {row['Org Name']}")

            code,desc,conf,reason,summary,candidates = predict_naics(
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
                "Confidence":conf,
                "Reasoning":reason
            })

            progress_bar.progress((i+1)/len(df_input))

        df_results = pd.DataFrame(results)

        end_time = time.time()

        st.success("✅ Classification Completed")

# ------------------------------------------------------------
# Confidence Chart
# ------------------------------------------------------------
        st.subheader("📊 Confidence Distribution")

        st.bar_chart(df_results["Confidence"].value_counts())

# ------------------------------------------------------------
# Editable Results
# ------------------------------------------------------------
        st.subheader("✏️ Edit Results")

        edited_df = st.data_editor(df_results)

# ------------------------------------------------------------
# Show Results
# ------------------------------------------------------------
        st.subheader("📋 Classification Results")

        st.dataframe(edited_df)

        st.info(
            f"⏱ Processing Time: {round(end_time-start_time,2)} seconds"
        )

# ------------------------------------------------------------
# 13. Download Output
# ------------------------------------------------------------
        def to_excel(df):

            output = BytesIO()

            df.to_excel(output,index=False)

            output.seek(0)

            return output

        st.download_button(
            label="📥 Download Excel",
            data=to_excel(edited_df),
            file_name="naics_classified_output.xlsx"
        )