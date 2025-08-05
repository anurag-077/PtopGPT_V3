import os
import pandas as pd
import streamlit as st
import logging
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
import tensorflow as tf
import joblib
from fuzzywuzzy import process
from time import sleep
import tiktoken

# Suppress warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Set Pandas future behavior
pd.set_option('future.no_silent_downcasting', True)

# Load environment variables
load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY not found in .env file. Please set it correctly.")
    st.stop()
logger.info("OPENAI_API_KEY loaded: %s", "yes" if os.environ.get("OPENAI_API_KEY") else "no")

# Category mapping
CATEGORY_MAPPING = {
    "demand": [
        "total sold - igr", "flat_sold - igr", "office_sold - igr", "others_sold - igr", "shop_sold - igr",
        "commercial_sold - igr", "other_sold - igr", "residential_sold - igr",
        "1.5bhk_sold - igr", "1bhk_sold - igr", "2.25bhk_sold - igr", "2.5bhk_sold - igr",
        "2.75bhk_sold - igr", "2bhk_sold - igr", "3bhk_sold - igr", "<1bhk_sold - igr",
        ">3bhk_sold - igr", "total carpet area consumed (sqft) - igr",
        "flat_carpet_area_consumed(in sqft) - igr", "office_carpet_area_consumed(in sqft) - igr",
        "others_carpet_area_consumed(in sqft) - igr", "shop_carpet_area_consumed(in sqft) - igr",
        "1.5bhk_carpet_area_consumed(in sqft) - igr", "1bhk_carpet_area_consumed(in sqft) - igr",
        "2.25bhk_carpet_area_consumed(in sqft) - igr", "2.5bhk_carpet_area_consumed(in sqft) - igr",
        "2.75bhk_carpet_area_consumed(in sqft) - igr", "2bhk_carpet_area_consumed(in sqft) - igr",
        "3bhk_carpet_area_consumed(in sqft) - igr", "<1bhk_carpet_area_consumed(in sqft) - igr",
        ">3bhk_carpet_area_consumed(in sqft) - igr", "residential_carpet_area_consumed(in sqft) - igr",
        "commercial_carpet_area_consumed(in sqft) - igr", "other_carpet_area_consumed(in sqft) - igr",
        "flat - avg carpet area (in sqft)", "others - avg carpet area (in sqft)",
        "office - avg carpet area (in sqft)", "shop - avg carpet area (in sqft)",
        "<1bhk - avg carpet area (in sqft)", "1bhk - avg carpet area (in sqft)",
        "1.5bhk - avg carpet area (in sqft)", "2bhk - avg carpet area (in sqft)",
        "2.25bhk - avg carpet area (in sqft)", "2.5bhk - avg carpet area (in sqft)",
        "2.75bhk - avg carpet area (in sqft)", "3bhk - avg carpet area (in sqft)",
        ">3bhk - avg carpet area (in sqft)",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects"
    ],
    "supply": [
        "total units", "flat total", "shop total", "office total", "others total",
        "<1bhk total", "1bhk total", "1.5bhk total", "2bhk total", "2.25bhk total",
        "2.5bhk total", "2.75bhk total", "3bhk total", ">3bhk total",
        "total carpet area supplied (sqft)", "flat_carpet_area_supplied_rera_sqft",
        "shop_carpet_area_supplied_rera_sqft", "others_carpet_area_supplied_rera_sqft",
        "office_carpet_area_supplied_rera_sqft", "undefined flats_carpet_area_supplied_rera_sqft",
        "<1bhk_carpet_area_supplied_rera_sqft", "1bhk_carpet_area_supplied_rera_sqft",
        "1.5bhk_carpet_area_supplied_rera_sqft", "2bhk_carpet_area_supplied_rera_sqft",
        "2.25bhk_carpet_area_supplied_rera_sqft", "2.5bhk_carpet_area_supplied_rera_sqft",
        "2.75bhk_carpet_area_supplied_rera_sqft", "3bhk_carpet_area_supplied_rera_sqft",
        ">3bhk_carpet_area_supplied_rera_sqft",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects"
    ],
    "price": [
        "total_sales - igr", "flat- total agreement price", "office- total agreement price",
        "others- total agreement price", "shop- total agreement price",
        "1.5bhk- total agreement price", "1bhk- total agreement price",
        "2.25bhk- total agreement price", "2.5bhk- total agreement price",
        "2.75bhk- total agreement price", "2bhk- total agreement price",
        "3bhk- total agreement price", "<1bhk- total agreement price",
        ">3bhk- total agreement price", "commercial- total agreement price",
        "other- total agreement price", "residential- total agreement price",
        "flat - 50th percentile rate", "flat - 75th percentile rate", "flat - 90th percentile rate",
        "office - 50th percentile rate", "others - 50th percentile rate", "shop - 50th percentile rate",
        "office - 75th percentile rate", "others - 75th percentile rate", "shop - 75th percentile rate",
        "office - 90th percentile rate", "others - 90th percentile rate", "shop - 90th percentile rate",
        "commercial- avg agreement price", "other- avg agreement price", "residential- avg agreement price",
        "flat- avg agreement price", "office- avg agreement price", "others- avg agreement price",
        "shop- avg agreement price", "1.5bhk- avg agreement price", "1bhk- avg agreement price",
        "2.25bhk- avg agreement price", "2.5bhk- avg agreement price", "2.75bhk- avg agreement price",
        "2bhk- avg agreement price", "3bhk- avg agreement price", "<1bhk- avg agreement price",
        ">3bhk- avg agreement price", "flat - weighted average rate", "office - weighted average rate",
        "others - weighted average rate", "shop - weighted average rate",
        "flat - most prevailing rate - range", "office - most prevailing rate - range",
        "others - most prevailing rate - range", "shop - most prevailing rate - range",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects"
    ],
    "demography": [
        "flat - pincode wise unit sold", "others - pincode wise unit sold",
        "office - pincode wise unit sold", "shop - pincode wise unit sold",
        "<1bhk - pincode wise unit sold", "1bhk - pincode wise unit sold",
        "1.5bhk - pincode wise unit sold", "2bhk - pincode wise unit sold",
        "2.25bhk - pincode wise unit sold", "2.5bhk - pincode wise unit sold",
        "2.75bhk - pincode wise unit sold", "3bhk - pincode wise unit sold",
        ">3bhk - pincode wise unit sold", "flat - age range wise unit sold",
        "others - age range wise unit sold", "office - age range wise unit sold",
        "shop - age range wise unit sold", "<1bhk - age range wise unit sold",
        "1bhk - age range wise unit sold", "1.5bhk - age range wise unit sold",
        "2bhk - age range wise unit sold", "2.25bhk - age range wise unit sold",
        "2.5bhk - age range wise unit sold", "2.75bhk - age range wise unit sold",
        "3bhk - age range wise unit sold", ">3bhk - age range wise unit sold",
        "flat - pincode wise total sales", "others - pincode wise total sales",
        "office - pincode wise total sales", "shop - pincode wise total sales",
        "<1bhk - pincode wise total sales", "1bhk - pincode wise total sales",
        "1.5bhk - pincode wise total sales", "2bhk - pincode wise total sales",
        "2.25bhk - pincode wise total sales", "2.5bhk - pincode wise total sales",
        "2.75bhk - pincode wise total sales", "3bhk - pincode wise total sales",
        ">3bhk - pincode wise total sales", "flat - pincode wise carpet area consumed in sqft",
        "others - pincode wise carpet area consumed in sqft",
        "office - pincode wise carpet area consumed in sqft",
        "shop - pincode wise carpet area consumed in sqft",
        "<1bhk - pincode wise carpet area consumed in sqft",
        "1bhk - pincode wise carpet area consumed in sqft",
        "1.5bhk - pincode wise carpet area consumed in sqft",
        "2bhk - pincode wise carpet area consumed in sqft",
        "2.25bhk - pincode wise carpet area consumed in sqft",
        "2.5bhk - pincode wise carpet area consumed in sqft",
        "2.75bhk - pincode wise carpet area consumed in sqft",
        "3bhk - pincode wise carpet area consumed in sqft",
        ">3bhk - pincode wise carpet area consumed in sqft",
        "flat - age range wise carpet area consumed in sqft",
        "others - age range wise carpet area consumed in sqft",
        "office - age range wise carpet area consumed in sqft",
        "shop - age range wise carpet area consumed in sqft",
        "top_buyer_pincode", "top10_buyer_in_locality",
        "loc_lat", "loc_lng", "residential share (in %)", "commercial share (in %)",
        "others share (in %)", "residential+commercial share (in %)",
        "total projects commenced", "total no.of rera entry",
        "total no. of blocks (based of projects)", "no. of residential projects",
        "no. of commercial projects", "no. of others projects", "residential+commercial",
        "no. of indusrial projects"
    ]
}

# File paths
try:
    excel_path = os.path.join(os.path.dirname(__file__), "SampleR.xlsx")
    pickle_path = os.path.join(os.path.dirname(__file__), "SampleR.pkl")
except Exception as e:
    logger.error(f"Failed to locate data files: {e}")
    st.error(f"Failed to locate data files: {e}. Ensure SampleR.xlsx is in the correct directory.")
    st.stop()

# Load and clean data
def load_and_clean_data(excel_path, pickle_path, villages=None, years=None, category=None):
    try:
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
            logger.info(f"Pickle file loaded. Shape: {df.shape}")
        else:
            df = pd.read_excel(excel_path)
            joblib.dump(df, pickle_path, compress=3)
            logger.info(f"Excel file loaded and saved as pickle. Shape: {df.shape}")
        
        df["final location"] = df["final location"].str.strip().str.lower()
        available_villages = df["final location"].unique()
        logger.info(f"Available villages: {available_villages}")
        
        if villages:
            df = df[df["final location"].isin([v.lower() for v in villages])]
            if df.empty:
                logger.error(f"No data for villages {villages}")
                st.error(f"No data for villages {villages}. Check village names in SampleR.xlsx.")
                st.stop()
            logger.info(f"Filtered data for villages {villages}. Shape: {df.shape}")
        
        years = [y for y in (years or [2020, 2021, 2022, 2023, 2024]) if 2020 <= y <= 2024]
        if years:
            df = df[df["year"].isin(years)]
            logger.info(f"Filtered data for years {years}. Shape: {df.shape}")
        
        df = df.sort_values(by=["final location", "year"])
        
        if category and category != "general":
            relevant_columns = ["final location", "year"]
            category_keywords = CATEGORY_MAPPING.get(category.lower(), [])
            for col in df.columns:
                if any(keyword in col.lower() for keyword in category_keywords):
                    relevant_columns.append(col)
            relevant_columns = list(dict.fromkeys(relevant_columns))
            df = df[[col for col in relevant_columns if col in df.columns]]
            logger.info(f"Filtered columns for category '{category}'. Shape: {df.shape}")
        
        defaults = {
            "year": 2020,
            "total sold - igr": 0,
            "1bhk_sold - igr": 0,
            "flat total": 0,
            "shop total": 0,
            "office total": 0,
            "others total": 0,
            "1bhk total": 0,
            "<1bhk total": 0
        }
        
        df = df.infer_objects(copy=False).fillna({col: defaults.get(col, 0) for col in df.columns})
        logger.info(f"Final data shape: {df.shape}, columns: {df.columns.tolist()}")
        return df, defaults
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        st.error(f"Failed to load data: {e}. Check SampleR.xlsx.")
        st.stop()

# Get village names
@st.cache_data
def get_village_names():
    try:
        if Path(pickle_path).exists():
            df = joblib.load(pickle_path)
        else:
            df = pd.read_excel(excel_path)
            joblib.dump(df, pickle_path, compress=3)
        villages = sorted(df["final location"].str.strip().str.lower().unique())
        if not villages:
            logger.error("No villages found in SampleR.xlsx")
            st.error("No villages found in SampleR.xlsx.")
            st.stop()
        logger.info(f"Available villages: {villages}")
        return villages
    except Exception as e:
        logger.error(f"Failed to load village names: {e}")
        st.error(f"Failed to load village names: {e}.")
        st.stop()

# Format data for a single column
def format_column_data(df, column, village1, village2, defaults, years=[2020, 2021, 2022, 2023, 2024]):
    lines = []
    for village in [village1.lower(), village2.lower()]:
        village_df = df[df["final location"] == village]
        for year in years:
            year_df = village_df[village_df["year"] == year]
            value = year_df[column].iloc[0] if not year_df.empty and column in year_df.columns else defaults.get(column, 'N/A')
            lines.append(f"{village} {year} {column}: {value}")
    return "\n".join(lines)

# Create documents organized by column
def create_documents(df, village1, village2, defaults, include_columns=None):
    documents = []
    include_columns = include_columns if include_columns else [col for col in df.columns if col not in ['final location', 'year']]
    years = [2020, 2021, 2022, 2023, 2024]
    
    for column in include_columns:
        if column in df.columns:
            content = format_column_data(df, column, village1, village2, defaults, years)
            documents.append(
                Document(
                    page_content=content,
                    metadata={'column': column, 'village1': village1.lower(), 'village2': village2.lower()}
                )
            )
    
    logger.info(f"Created {len(documents)} documents for villages {village1} and {village2}")
    return documents

# Create vector store
@st.cache_resource
def get_vector_store(village1, village2, category):
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("HuggingFace embeddings initialized")

        vector_store_path = f"faiss_index_{village1}_{village2}_{category}"
        df, defaults = load_and_clean_data(excel_path, pickle_path, villages=[village1, village2], years=[2020, 2021, 2022, 2023, 2024], category=category)
        
        if df.empty:
            logger.error(f"No data for {village1} or {village2} in 2020-2024.")
            st.error(f"No data for {village1} or {village2} in 2020-2024. Check village names.")
            st.stop()
        
        available_villages = df["final location"].unique()
        selected_villages = []
        for village in [village1.lower(), village2.lower()]:
            if village not in available_villages:
                matches = process.extract(village, available_villages, limit=1)
                if matches and matches[0][1] >= 70:
                    best_match = matches[0][0]
                    logger.warning(f"Village '{village}' not found. Using: '{best_match}'")
                    st.warning(f"Village '{village}' not found. Using: '{best_match}'")
                    selected_villages.append(best_match)
                else:
                    suggestions = [match[0] for match in process.extract(village, available_villages, limit=3)]
                    logger.error(f"Village '{village}' not found. Suggestions: {suggestions}")
                    st.error(f"Village '{village}' not found. Suggestions: {', '.join(suggestions) or 'None'}")
                    st.stop()
            else:
                selected_villages.append(village)
        
        df = df[df["final location"].isin(selected_villages)]
        include_columns = CATEGORY_MAPPING.get(category.lower(), [])
        include_columns = [col for col in include_columns if col in df.columns and col not in ['final location', 'year']]
        
        if not include_columns:
            logger.error("No valid columns for category.")
            st.error(f"No valid columns for category '{category}'. Check SampleR.xlsx.")
            st.stop()
        
        if os.path.exists(vector_store_path):
            os.system(f"rm -rf {vector_store_path}")
            logger.info(f"Cleared FAISS index at {vector_store_path}")
        
        documents = create_documents(df, village1, village2, defaults, include_columns)
        if not documents:
            logger.error("No documents created for FAISS index")
            st.error("Failed to create documents. Check SampleR.xlsx.")
            st.stop()
        
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(vector_store_path)
        logger.info(f"FAISS index created with {len(documents)} documents")
        return vector_store, defaults
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        st.error(f"Failed to create vector store: {str(e)}.")
        st.stop()

# Token counting
def count_tokens(text, model="gpt-4o-mini"):
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception as e:
        logger.error(f"Error counting tokens: {e}")
        return 0

# LLM setup
try:
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=8000,
        max_retries=3,
        api_key=os.environ.get("OPENAI_API_KEY")
    )
    logger.info("OpenAI gpt-4o-mini LLM initialized")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI: {e}")
    st.error(f"Failed to initialize OpenAI: {e}. Check API key.")
    st.stop()

# Prompt template
prompt_template = """You are a real estate analyst. The user has requested: "{question}" for {village1} and {village2} (2020–2024). The data from SampleR.xlsx contains metrics for 
the '{category}' category, prefixed by village and year (e.g., 'Aundh 2020 total units'). Use the data to compare {village1} and {village2}. If direct metrics are missing, infer trends or note limitations. Provide a concise text-based analysis.

Data:
{context}

Provide the analysis focusing on trends and insights.
"""
# Streamlit UI
st.title("Real Estate Village Comparison (2020–2024)")
st.markdown("Select two villages and multiple categories to compare real estate data.")

villages = get_village_names()

col1, col2 = st.columns(2)
with col1:
    village1 = st.selectbox("Village 1", options=[""] + villages, index=0, placeholder="Select Village 1")
with col2:
    village2 = st.selectbox("Village 2", options=[""] + villages, index=0, placeholder="Select Village 2")

# Changed to multiselect for multiple categories
categories = st.multiselect("Categories", options=["Demand", "Supply", "Price", "Demography"], default=["Demand"])
query = st.text_area("Query", placeholder="e.g., give me analysis of both for selected categories")

if st.button("Compare Villages"):
    if not village1 or village1 == "":
        st.error("Please select Village 1.")
        st.stop()
    if not village2 or village2 == "":
        st.error("Please select Village 2.")
        st.stop()
    if village1.lower() == village2.lower():
        st.error("Villages must be different.")
        st.stop()
    if not categories:
        st.error("Please select at least one category.")
        st.stop()
    
    try:
        if not query or query.strip() == "":
            query = f"Compare {', '.join(categories).lower()} metrics for {village1} and {village2}"
            st.info(f"No query provided. Using default: '{query}'")
        
        logger.info(f"Query: '{query}'")
        
        # Load and clean data for all categories together
        all_categories = [cat.lower() for cat in categories]
        combined_include_columns = set()
        for category in all_categories:
            combined_include_columns.update(CATEGORY_MAPPING.get(category, []))
        combined_include_columns = list(combined_include_columns)
        
        df, defaults = load_and_clean_data(excel_path, pickle_path, villages=[village1, village2], years=[2020, 2021, 2022, 2023, 2024], category=None)
        if df.empty:
            logger.error(f"No data for {village1} or {village2} in 2020-2024.")
            st.error(f"No data for {village1} or {village2} in 2020-2024. Check village names.")
            st.stop()
        
        # Filter columns based on combined categories
        relevant_columns = ["final location", "year"]
        for col in df.columns:
            if any(keyword in col.lower() for keyword in combined_include_columns):
                relevant_columns.append(col)
        relevant_columns = list(dict.fromkeys(relevant_columns))
        df = df[[col for col in relevant_columns if col in df.columns]]
        logger.info(f"Filtered data for combined categories. Shape: {df.shape}")
        
        # Create a single vector store
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("HuggingFace embeddings initialized")
        
        vector_store_path = f"faiss_index_{village1}_{village2}_combined"
        if os.path.exists(vector_store_path):
            os.system(f"rm -rf {vector_store_path}")
            logger.info(f"Cleared FAISS index at {vector_store_path}")
        
        documents = create_documents(df, village1, village2, defaults, combined_include_columns)
        if not documents:
            logger.error("No documents created for FAISS index")
            st.error("Failed to create documents. Check SampleR.xlsx.")
            st.stop()
        
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(vector_store_path)
        logger.info(f"FAISS index created with {len(documents)} documents for combined categories")
        
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": len(combined_include_columns)})
        
        with st.spinner("Generating comparison..."):
            try:
                docs = retriever.invoke(query.strip())
                if not docs:
                    logger.error("No documents retrieved from FAISS")
                    st.error("No relevant data found. Refine query or check village names.")
                    st.stop()
                
                context = "\n\n".join([doc.page_content.strip() for doc in docs])
                logger.info(f"Retrieved {len(docs)} documents")
                
                formatted_prompt = prompt_template.format(
                    question=query.strip(),
                    village1=village1,
                    village2=village2,
                    category=", ".join(categories).lower(),
                    context=context
                )
                
                input_tokens = count_tokens(formatted_prompt)
                logger.info(f"Input tokens: {input_tokens}")
                
                # Stream the LLM response
                st.markdown("### Analysis")
                
                # Generator function to yield chunks
                def generate_response():
                    stream = llm.stream(formatted_prompt)
                    for chunk in stream:
                        chunk_text = chunk.content if hasattr(chunk, 'content') else str(chunk)
                        yield chunk_text
                
                # Use st.write_stream with the generator
                st.write_stream(generate_response)
                
                # Calculate output tokens after streaming (approximate)
                response_text = "".join(generate_response())  # Re-run generator to get full text
                output_tokens = count_tokens(response_text)
                st.info(f"Tokens used - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")
                
                with st.expander("Show Sources"):
                    for doc in docs:
                        st.text(doc.page_content)
            except Exception as e:
                logger.error(f"Error processing comparison: {e}")
                st.error(f"Could not process request: {str(e)}.")
                st.stop()
    except Exception as e:
        logger.error(f"Error in comparison setup: {e}")
        st.error(f"Failed to set up comparison: {str(e)}.")
        st.stop()