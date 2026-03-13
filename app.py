import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Import necessary libraries for the analysis
import os
import re
import requests
import fitz  # PyMuPDF
import openai
from pydantic import BaseModel
from typing import List, Literal
from sklearn.metrics.pairwise import cosine_similarity

# Configuration
LARGE_LANGUAGE_MODEL = "glm-4"
EMBEDDINGS_MODEL = "embedding-2"
CHUNK_SIZE = 2000
OVERLAP = 300
TOP_N = 7
THRESHOLD = 0.5
RANDOM_SEED = 42

# Set up Zhipu AI client
client = openai.OpenAI(
    api_key=st.secrets.get("ZHIPU_API_KEY", os.environ.get("ZHIPU_API_KEY")),
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

# Pydantic schemas
class SolvencyRatioSchema(BaseModel):
    company: str
    solvency_ratio: float
    regulatory_framework: str
    year: int

class DiscountRatePerDuration(BaseModel):
    duration: str
    discount_rate: float
    currency: str

class DiscountRatesSchema(BaseModel):
    company: str
    discount_rates: List[DiscountRatePerDuration]
    year: int

class CyberRiskStrategiesSchema(BaseModel):
    company: str
    cyber_risk_strategies: List[str]
    year: int

# Utility functions
def clean_text(text: str) -> str:
    """Clean and normalize text."""
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def create_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def generate_text_embedding(chunk: str, model: str = EMBEDDINGS_MODEL) -> list[float]:
    """Generate embeddings for text chunks."""
    try:
        response = client.embeddings.create(
            input=chunk,
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return []

def load_pdfs_from_uploads(uploaded_files) -> list[dict]:
    """Load PDFs from uploaded files."""
    documents = []
    for uploaded_file in uploaded_files:
        try:
            pdf_data = uploaded_file.read()
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append({
                "company": uploaded_file.name.replace('.pdf', ''),
                "text": clean_text(text)
            })
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    return documents

def load_default_pdfs() -> list[dict]:
    """Load default PDFs from URLs."""
    annual_report_urls = {
        "AXA": "https://www-axa-com.cdn.axa-contento-118412.eu/www-axa-com/fd85b507-f97f-4ac5-861b-6b2b90e1c601_AXA_URD2024_EN.pdf",
        "Generali": "https://www.generali.com/doc/jcr:259c5d6e-46f7-4a43-9512-58e5dcbd2a56/lang:en/Annual%20Integrated%20Report%20and%20Consolidated%20Financial%20Statements%202024_Generali%20Group_final_interactive.pdf",
        "Zurich": "https://edge.sitecorecloud.io/zurichinsur6934-zwpcorp-prod-ae5e/media/project/zurich/dotcom/investor-relations/docs/financial-reports/2024/annual-report-2024-en.pdf"
    }

    documents = []
    for company, url in annual_report_urls.items():
        try:
            response = requests.get(url)
            response.raise_for_status()
            doc = fitz.open(stream=response.content, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            documents.append({
                "company": company,
                "text": clean_text(text)
            })
            st.success(f"Loaded {company} report successfully")
        except Exception as e:
            st.error(f"Failed to load {company}: {e}")
    return documents

def retrieve_top_matching_chunks(query_embedding: list[float], all_chunks: list[dict], top_n: int = TOP_N, threshold: float = THRESHOLD) -> list[dict]:
    """Retrieve top matching chunks using cosine similarity."""
    if not all_chunks:
        return []

    chunk_embeddings = np.array([chunk['embedding'] for chunk in all_chunks])
    query_embedding = np.array(query_embedding).reshape(1, -1)

    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]

    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_chunks = []

    for idx in top_indices:
        if similarities[idx] >= threshold:
            top_chunks.append({
                'chunk': all_chunks[idx]['chunk'],
                'similarity': similarities[idx],
                'company': all_chunks[idx]['company']
            })

    return top_chunks

def query_structured_output(query: str, context_chunks: list[str], response_schema) -> dict:
    """Query the LLM with structured output."""
    context = "\n\n".join(context_chunks)

    prompt = f"""
    Based on the following context from insurance company annual reports, please extract and structure the requested information.

    Context:
    {context}

    Query: {query}

    Please provide a structured response following the specified format. If information is not available, indicate this clearly.
    """

    try:
        response = client.beta.chat.completions.parse(
            model=LARGE_LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert actuarial analyst extracting structured information from insurance reports."},
                {"role": "user", "content": prompt}
            ],
            response_format=response_schema
        )
        return response.choices[0].message.parsed
    except Exception as e:
        st.error(f"Error in LLM query: {e}")
        return None

# Streamlit App
st.title("🏢 GenAI-Driven Market Comparison Tool")
st.markdown("Analyze insurance company reports using AI-powered extraction and comparison")

# Sidebar for configuration
st.sidebar.header("Configuration")
use_default = st.sidebar.checkbox("Use Default Reports (AXA, Generali, Zurich)", value=True)
uploaded_files = st.sidebar.file_uploader("Or Upload Your Own PDFs", accept_multiple_files=True, type="pdf")

# Analysis options
analysis_type = st.sidebar.selectbox(
    "Select Analysis Type",
    ["Solvency Ratios", "Discount Rates", "Cyber Risk Strategies"]
)

if st.button("🚀 Run Analysis"):
    with st.spinner("Loading documents..."):
        if use_default:
            documents = load_default_pdfs()
        elif uploaded_files:
            documents = load_pdfs_from_uploads(uploaded_files)
        else:
            st.error("Please select default reports or upload PDFs")
            st.stop()

    if not documents:
        st.error("No documents loaded")
        st.stop()

    # Process documents
    with st.spinner("Processing documents..."):
        all_chunks = []
        for doc in documents:
            chunks = create_chunks(doc['text'])
            for chunk in chunks:
                embedding = generate_text_embedding(chunk)
                if embedding:
                    all_chunks.append({
                        'chunk': chunk,
                        'embedding': embedding,
                        'company': doc['company']
                    })

    st.success(f"Processed {len(all_chunks)} text chunks from {len(documents)} documents")

    # Run analysis based on type
    with st.spinner(f"Running {analysis_type} analysis..."):
        if analysis_type == "Solvency Ratios":
            query = "Extract the solvency ratio, regulatory framework, and year for each company."
            schema = SolvencyRatioSchema
            results = []

            for doc in documents:
                company_chunks = [chunk for chunk in all_chunks if chunk['company'] == doc['company']]
                if company_chunks:
                    context_chunks = [chunk['chunk'] for chunk in company_chunks[:5]]  # Use top chunks
                    result = query_structured_output(query, context_chunks, schema)
                    if result:
                        results.append(result.dict() if hasattr(result, 'dict') else result)

        elif analysis_type == "Discount Rates":
            query = "Extract discount rates by duration, including currency and year."
            schema = DiscountRatesSchema
            results = []

            for doc in documents:
                company_chunks = [chunk for chunk in all_chunks if chunk['company'] == doc['company']]
                if company_chunks:
                    context_chunks = [chunk['chunk'] for chunk in company_chunks[:5]]
                    result = query_structured_output(query, context_chunks, schema)
                    if result:
                        results.append(result.dict() if hasattr(result, 'dict') else result)

        elif analysis_type == "Cyber Risk Strategies":
            query = "Extract cyber risk mitigation strategies and approaches."
            schema = CyberRiskStrategiesSchema
            results = []

            for doc in documents:
                company_chunks = [chunk for chunk in all_chunks if chunk['company'] == doc['company']]
                if company_chunks:
                    context_chunks = [chunk['chunk'] for chunk in company_chunks[:5]]
                    result = query_structured_output(query, context_chunks, schema)
                    if result:
                        results.append(result.dict() if hasattr(result, 'dict') else result)

    # Display results
    if results:
        st.header("📊 Analysis Results")

        # Convert to DataFrame for display
        df = pd.DataFrame(results)
        st.dataframe(df)

        # Download results
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"{analysis_type.lower().replace(' ', '_')}_analysis.csv",
            mime="text/csv"
        )

        # Summary statistics
        st.header("📈 Summary")
        st.metric("Companies Analyzed", len(results))
        st.metric("Total Data Points", len(df))

    else:
        st.warning("No results found. Try adjusting the query or check if the documents contain the requested information.")

st.markdown("---")
st.markdown("Built with Streamlit and Zhipu AI GLM-4")

# Function to extract text from PDF
def extract_text_from_pdf(file_path: str) -> str:
    try:
        doc = fitz.open(file_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
        return ""

# Function to chunk text
def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 300) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks

# Function to find relevant chunks
def find_relevant_chunks(query: str, chunks: List[str], top_n: int = 7) -> List[str]:
    query_embedding = generate_text_embedding(query)
    if not query_embedding:
        return []

    chunk_embeddings = [generate_text_embedding(chunk) for chunk in chunks]
    chunk_embeddings = [emb for emb in chunk_embeddings if emb]

    if not chunk_embeddings:
        return []

    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]

    return [chunks[i] for i in top_indices]

# Function to extract solvency ratios
def extract_solvency_ratios(company: str, relevant_chunks: List[str]) -> List[SolvencyRatioSchema]:
    context = "\n".join(relevant_chunks)

    prompt = f"""
    Extract solvency capital ratios from the following text for {company}.
    Return the data in a structured format with company name, year, ratio value, and regulatory framework (e.g., Solvency II, SST).
    Only extract actual numerical ratios mentioned in the text.
    """

    try:
        response = client.beta.chat.completions.parse(
            model=LARGE_LANGUAGE_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert in extracting financial data from insurance reports."},
                {"role": "user", "content": f"{prompt}\n\nContext:\n{context}"}
            ],
            response_format=SolvencyRatioSchema
        )
        return [response.choices[0].message.parsed]
    except Exception as e:
        st.error(f"Error extracting solvency ratios: {e}")
        return []

# Main analysis function
def run_market_comparison():
    st.header("Market Comparison Analysis")

    # Predefined company reports
    companies = ["AXA", "Generali", "Zurich"]

    all_ratios = []

    for company in companies:
        st.subheader(f"Processing {company}")

        # Download and process PDF
        pdf_path = f"./annual_reports/{company}.pdf"

        if not os.path.exists(pdf_path):
            st.warning(f"PDF for {company} not found. Please ensure PDFs are available.")
            continue

        # Extract text
        text = extract_text_from_pdf(pdf_path)
        if not text:
            continue

        # Chunk text
        chunks = chunk_text(text)

        # Find relevant chunks for solvency ratios
        query = "solvency capital ratio SCR SST Solvency II"
        relevant_chunks = find_relevant_chunks(query, chunks)

        # Extract ratios
        ratios = extract_solvency_ratios(company, relevant_chunks)
        all_ratios.extend(ratios)

        st.write(f"Found {len(ratios)} solvency ratios for {company}")

    # Display results
    if all_ratios:
        st.header("Comparison Results")
        df = pd.DataFrame([ratio.dict() for ratio in all_ratios])
        st.dataframe(df)

        # Simple visualization
        if 'ratio' in df.columns and 'company' in df.columns:
            st.bar_chart(df.set_index('company')['ratio'])
    else:
        st.warning("No solvency ratios were extracted. This could be due to API limits or missing data.")

# File uploader for custom PDFs
st.header("Upload Custom Report")
uploaded_file = st.file_uploader("Upload a PDF report for analysis", type="pdf")

if uploaded_file is not None:
    st.success("File uploaded successfully!")
    # Save uploaded file temporarily
    with open("temp_upload.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Extract and display some text
    text = extract_text_from_pdf("temp_upload.pdf")
    if text:
        st.subheader("Extracted Text Preview")
        st.text_area("Text content", text[:1000] + "..." if len(text) > 1000 else text, height=200)

# Button to run analysis
if st.button("Run Market Comparison Analysis"):
    with st.spinner("Running analysis... This may take a few minutes."):
        run_market_comparison()

st.header("About")
st.markdown("""
This app uses Retrieval-Augmented Generation (RAG) with Zhipu AI's GLM-4 model to:
- Extract solvency ratios from insurance annual reports
- Compare data across different companies
- Provide structured insights for actuarial analysis

**Note**: This is a demonstration app. Full analysis requires API access and may incur costs.
""")

# Add more sections as needed