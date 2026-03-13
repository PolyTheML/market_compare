import os
import re
import json
import time
import tempfile
from typing import List, Literal
import streamlit as st
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from pydantic import BaseModel
# from zhipuai import ZhipuAI  # Import only when needed to avoid initialization issues
# from openai import OpenAI # Required for type hinting if strictly following original logic, though ZhipuAI wraps it.

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(page_title="Insurer Report RAG Analyzer", layout="wide")

# ---------------------------------------------------------
# Pydantic Schemas for Structured Outputs
# ---------------------------------------------------------
class SolvencyRatioSchema(BaseModel):
    capital_ratio: int
    regulatory_framework: Literal["Solvency II", "SST"]

class DiscountRatePerDuration(BaseModel):
    duration_year: int
    discount_rate: float

class DiscountRatesSchema(BaseModel):
    discount_rates_per_duration: List[DiscountRatePerDuration]

class CyberRiskStrategiesSchema(BaseModel):
    strategies: List[str]

# ---------------------------------------------------------
# Core Logic Functions
# ---------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalizes whitespace and removes empty lines."""
    cleaned = re.sub(r"[ \t]+", " ", text)
    cleaned = "\n".join(line.strip() for line in cleaned.splitlines())
    cleaned = re.sub(r"\n\s*\n+", "\n\n", cleaned)
    return cleaned

def create_chunks(text: str, chunk_size=2000, overlap=300) -> list[str]:
    """Splits text into overlapping chunks."""
    step = chunk_size - overlap
    if step <= 0: raise ValueError("chunk_size must be > overlap")
    return [text[i : i + chunk_size] for i in range(0, len(text), step)]

def get_embeddings(client, text: str, model: str) -> list[float]:
    """Generates embeddings using ZhipuAI."""
    resp = client.embeddings.create(
        model=model,
        input=text,
        encoding_format="float"
    )
    return resp.data[0].embedding

def process_uploaded_files(uploaded_files, client, embeddings_model, progress_bar):
    """Processes uploaded PDFs into chunks and embeddings."""
    embeddings_data = []
    
    total_files = len(uploaded_files)
    
    for idx, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((idx + 0.5) / total_files, text=f"Processing {uploaded_file.name}...")
        
        # Read PDF using PyMuPDF from bytes
        # To use fitz.open with a stream, we read the file into memory
        pdf_bytes = uploaded_file.read()
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            full_text = "".join(page.get_text() for page in doc)
            
        cleaned = clean_text(full_text)
        chunks = create_chunks(cleaned)
        
        for chunk in chunks:
            emb = get_embeddings(client, chunk, embeddings_model)
            embeddings_data.append({
                "company": uploaded_file.name.replace(".pdf", ""), # Use filename as company name
                "chunk": chunk,
                "embedding": emb
            })
            
    progress_bar.empty()
    return embeddings_data

def retrieve_top_matching_chunks(document_embeddings: list[dict], prompt: str, embeddings_model: str, top_n=7, threshold=0.5) -> str:
    """Retrieves relevant chunks based on cosine similarity."""
    prompt_emb = get_embeddings(client, prompt, embeddings_model)
    if not prompt_emb: return "No context found."

    prompt_vec = np.array(prompt_emb).reshape(1, -1)
    df = pd.DataFrame(document_embeddings)
    
    # Calculate Cosine Similarity
    df['similarity'] = df['embedding'].apply(
        lambda emb: np.dot(prompt_vec, np.array(emb).reshape(1, -1).T) / 
                    (np.linalg.norm(prompt_vec) * np.linalg.norm(np.array(emb).reshape(1, -1)))
    )[0,0] # Access scalar value

    df_filtered = df[df['similarity'] >= threshold]
    if df_filtered.empty: return "No relevant content found in uploaded files."

    top = df_filtered.nlargest(top_n, 'similarity')
    
    retrieved_chunks = [
        f"=== Chunk {i} ({row['company']}, Similarity: {row['similarity']:.2f}) ===\n{row['chunk']}"
        for i, (_, row) in enumerate(top.iterrows(), start=1)
    ]
    return "\n\n".join(retrieved_chunks)

def query_structured_output(client, retrieved_chunks, prompt, schema, system_prompt, llm_model):
    """Runs the structured output generation."""
    augmented_query = (
        f"Prompt:\n{prompt}\n\n"
        f"Text Chunks:\n{retrieved_chunks}\n\n"
        "Extract the required fields according to the schema below.\n"
        f"{json.dumps(schema.model_json_schema(), indent=2)}\n\n"
        "If any field cannot be found, use 'NA'."
    )

    try:
        # Using ZhipuAI client which follows OpenAI SDK structure for responses.parse
        response = client.responses.parse(
            model=llm_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": augmented_query}
            ],
            temperature=0.1,
            text_format=schema
        )
        return response.output_parsed
    except Exception as e:
        return {"error": str(e)}

# ---------------------------------------------------------
# Prompts Configuration
# ---------------------------------------------------------

# Define an enhanced system prompt for RAG + Structured Outputs
SYSTEM_PROMPT = """
You are an AI assistant designed for actuarial and financial analysis of
European insurance company annual reports.

Your primary objective is to extract structured financial and risk metrics
from long-form regulatory and financial documents to enable market
comparison across insurers.

The system operates using Retrieval Augmented Generation (RAG). Always rely
on retrieved document context and avoid making assumptions beyond the text.

----------------------------------------------------
TASK OBJECTIVE
----------------------------------------------------

Extract key financial, regulatory, and risk management information from
insurance company annual reports and convert them into structured outputs
for actuarial analysis and cross-company comparison.

Focus on identifying:

• Solvency metrics
• Discount rate assumptions
• Risk management strategies
• Cyber risk disclosures
• Capital and reserve information
• Premium and claims indicators
• Key actuarial assumptions

----------------------------------------------------
EXTRACTION GUIDELINES
----------------------------------------------------

1. Only extract information that appears explicitly in the retrieved context.

2. Preserve numerical precision exactly as written in the document.

3. Capture the surrounding sentence or paragraph that supports the extracted
information to ensure traceability.

4. Standardize numerical formats where possible:
   - Percentages (e.g., 185%)
   - Currency values (e.g., €3.2 billion)
   - Interest/discount rates (e.g., 3.75%)

5. If information is not available in the retrieved context, return
"Not Found" rather than guessing.

----------------------------------------------------
KEY METRICS TO EXTRACT
----------------------------------------------------

Extract the following fields when available:

Company Information
- company_name
- reporting_year

Regulatory Metrics
- solvency_ii_ratio
- minimum_capital_requirement
- solvency_capital_requirement

Financial Indicators
- gross_written_premium
- total_claims
- technical_provisions
- combined_ratio

Actuarial Assumptions
- discount_rate
- inflation_assumption
- reserving_method (if mentioned)

Risk Disclosures
- cyber_risk_strategy
- climate_risk_strategy
- emerging_risks

----------------------------------------------------
OUTPUT FORMAT
----------------------------------------------------

Return all extracted information in structured JSON format following this schema:

{
  "company_name": "",
  "reporting_year": "",
  "solvency_metrics": {
      "solvency_ii_ratio": "",
      "scr": "",
      "mcr": ""
  },
  "financial_metrics": {
      "gross_written_premium": "",
      "total_claims": "",
      "technical_provisions": "",
      "combined_ratio": ""
  },
  "actuarial_assumptions": {
      "discount_rate": "",
      "inflation_assumption": "",
      "reserving_method": ""
  },
  "risk_management": {
      "cyber_risk_strategy": "",
      "climate_risk_strategy": "",
      "emerging_risks": ""
  },
  "source_text": []
}

----------------------------------------------------
COMPARATIVE ANALYSIS SUPPORT
----------------------------------------------------

Structure outputs so they can be easily compared across multiple insurers.
Ensure consistent naming conventions and formatting.

Do not include commentary outside the JSON structure.
"""



# Prompt for extracting solvency capital ratios specifically for the year 2024
PROMPT_SOLVENCY = """
You are extracting regulatory solvency information from an insurance
company's annual report using Retrieval Augmented Generation (RAG).

TASK
Identify the group's solvency capital ratio reported for the year 2024.

EXTRACTION RULES
- Extract only values explicitly stated in the retrieved document context.
- Do not infer or estimate values.
- If multiple solvency ratios are mentioned, prioritize:
  1) Group-level ratio
  2) Value reported as of December 31, 2024
- Preserve the numerical format exactly as written (e.g., 185%).

ADDITIONAL CONTEXT
Also identify the regulatory framework used by the insurer:
- Solvency II
- Swiss Solvency Test (SST)
- Other framework if explicitly mentioned

OUTPUT FORMAT (JSON)

{
  "reporting_year": "2024",
  "solvency_ratio_percent": "",
  "regulatory_framework": "",
  "source_text": ""
}

If the solvency ratio is not explicitly found in the retrieved context,
return "Not Found" for the missing fields.
"""

# Prompt for retrieving discount rate data specifically for the year 2024
PROMPT_DISCOUNT = """
You are extracting actuarial discount rate assumptions from an insurance
company's annual report using Retrieval Augmented Generation (RAG).

TASK
Extract the discount rates used for insurance or financial contract
liabilities for the reporting year 2024.

EXTRACTION RULES
- Only extract rates explicitly stated in the retrieved context.
- The reporting date must correspond to December 31, 2024.
- Extract rates only for EUR-denominated liabilities.
- Preserve the percentage format exactly as written.

DURATION EXTRACTION
Identify discount rates for each maturity duration when available.
Examples include:
1 year, 5 years, 10 years, 20 years, 30 years, 40 years.

If the document provides a yield curve table, extract all listed durations.

ASSUMPTION RULE
If the valuation approach is not explicitly stated, assume:
- Non-VFA
- Unit-linked contracts
- Liquid products

OUTPUT FORMAT (JSON)

{
  "reporting_year": "2024",
  "currency": "EUR",
  "discount_curve": [
      {
        "duration_years": "",
        "discount_rate_percent": ""
      }
  ],
  "valuation_approach": "",
  "source_text": ""
}

If discount rates cannot be found in the retrieved context,
return an empty list for "discount_curve".
"""

# Prompt for identifying cybersecurity risk management strategies
PROMPT_CYBER = """
You are extracting cybersecurity risk management disclosures from
an insurance company's annual report using Retrieval Augmented Generation (RAG).

TASK
Identify the insurer's documented approach to cyber-risk assessment,
governance, and mitigation.

EXTRACTION RULES
- Extract only information explicitly stated in the retrieved context.
- Do not infer strategies not mentioned in the report.
- Focus on formal governance structures, policies, and operational controls.

KEY ELEMENTS TO IDENTIFY

1. Governance
   - Responsible committee or board oversight
   - Chief Information Security Officer (CISO) or equivalent

2. Risk Assessment
   - Cyber risk evaluation methods
   - Scenario testing or stress testing

3. Mitigation Controls
   - Security frameworks (e.g., ISO 27001, NIST)
   - Monitoring tools
   - Incident response procedures

4. Monitoring and Review
   - Frequency of cyber-risk reviews
   - Reporting mechanisms

5. Quantitative Indicators (if available)
   - cyber incident metrics
   - investment in cybersecurity
   - staffing levels

OUTPUT FORMAT (JSON)

{
  "cyber_risk_governance": [],
  "risk_assessment_methods": [],
  "mitigation_controls": [],
  "monitoring_and_review": [],
  "quantitative_metrics": [],
  "source_text": []
}

Each item in the lists should contain a short bullet extracted from the
report describing the policy, process, or control.

If no cyber-risk information is found in the retrieved context,
return empty lists.
"""

QUERY_CONFIG = {
    "Solvency Ratio": {
        "prompt": PROMPT_SOLVENCY,
        "schema": SolvencyRatioSchema
    },
    "Discount Rates": {
        "prompt": PROMPT_DISCOUNT,
        "schema": DiscountRatesSchema
    },
    "Cyber Risk Strategies": {
        "prompt": PROMPT_CYBER,
        "schema": CyberRiskStrategiesSchema
    }
}

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.title("📊 Insurance Annual Report Analyzer (RAG + Structured Outputs)")
st.markdown("""
Upload one or more insurance annual reports (PDF) to extract key financial metrics 
using Retrieval Augmented Generation (RAG) and Structured Outputs.
""")

# Sidebar Configuration
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Zhipu AI API Key", type="password")
    
    st.subheader("Model Selection")
    # Common Zhipu AI models
    llm_options = ["glm-4", "glm-4-flash", "glm-3-turbo"]
    emb_options = ["embedding-2", "embedding-3"]
    
    selected_llm = st.selectbox("LLM Model", llm_options, index=0)
    selected_emb = st.selectbox("Embedding Model", emb_options, index=0)
    
    st.markdown("---")
    st.info("Ensure your API key has access to the selected models.")

# File Uploader
uploaded_files = st.file_uploader(
    "Upload Annual Reports (PDF)", 
    type="pdf", 
    accept_multiple_files=True
)

# Main Logic
if uploaded_files and api_key:
    if st.button("Process Reports", type="primary"):
        try:
            # Initialize Client
            from zhipuai import ZhipuAI
            client = ZhipuAI(api_key=api_key)
            
            # Stage 1: Ingestion & Embedding
            with st.spinner("Reading files and generating embeddings..."):
                progress_bar = st.progress(0)
                embeddings_data = process_uploaded_files(
                    uploaded_files, client, selected_emb, progress_bar
                )
            
            if not embeddings_data:
                st.error("No data could be extracted from files.")
            else:
                st.success(f"Processed {len(uploaded_files)} files into {len(embeddings_data)} chunks.")
                
                # Store in session state to persist during interaction
                st.session_state['embeddings_data'] = embeddings_data
                st.session_state['client'] = client
                
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")

# Check if data is ready in session state
if 'embeddings_data' in st.session_state and 'client' in st.session_state:
    st.markdown("---")
    st.header("Extraction Results")
    
    embeddings_data = st.session_state['embeddings_data']
    client = st.session_state['client']
    
    # Create tabs for each query type
    tabs = st.tabs(list(QUERY_CONFIG.keys()))
    
    results_store = {}
    
    for tab, (query_name, config) in zip(tabs, QUERY_CONFIG.items()):
        with tab:
            st.markdown(f"### {query_name} Extraction")
            
            if st.button(f"Extract {query_name}", key=f"btn_{query_name}"):
                with st.spinner("Retrieving relevant context and querying LLM..."):
                    # Stage 2: Retrieval
                    # Note: We use the prompt to find relevant chunks
                    retrieved_context = retrieve_top_matching_chunks(
                        embeddings_data, 
                        config['prompt'], 
                        selected_emb
                    )
                    
                    if "No relevant content" in retrieved_context:
                        st.warning("No relevant context found in documents for this query.")
                    else:
                        # Stage 3: Structured Output Generation
                        structured_result = query_structured_output(
                            client, 
                            retrieved_context, 
                            config['prompt'], 
                            config['schema'], 
                            SYSTEM_PROMPT, 
                            selected_llm
                        )
                        
                        results_store[query_name] = structured_result
                        
                        # Display Results
                        if isinstance(structured_result, dict) and "error" in structured_result:
                            st.error(f"Error: {structured_result['error']}")
                        else:
                            st.json(structured_result.model_dump_json(indent=2))
                            
                        # Expander to show retrieved context
                        with st.expander("View Retrieved Context"):
                            st.text_area("Source Chunks", retrieved_context, height=300)

else:
    if not uploaded_files:
        st.info("Please upload PDF files to begin.")
    if not api_key:
        st.info("Please enter your Zhipu AI API Key in the sidebar.")
