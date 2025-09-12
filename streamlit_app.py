import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import re
import time # Import time for potential rate limiting
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import InferenceClient

# --- FOCUSED AI PROMPT FUNCTIONS (NEW STRATEGY) ---

def get_mapping_prompt(content, column_name, options_list):
    """Creates a focused prompt for mapping content to a list of options."""
    return f"""
    Analyze the following 'Page Content'. Your task is to select the most relevant term(s) from the provided 'Options List' that accurately describe the content.

    **Instructions**:
    - If the column is 'Functional Area', select only the single best option.
    - For other columns, you can select multiple options if they are all relevant.
    - Your response MUST ONLY contain terms from the 'Options List'.
    - Separate multiple terms with a comma.
    - If no terms from the list are relevant, respond with an empty string.

    **Page Content**:
    ---
    {content[:4000]}
    ---

    **Column to Map**: {column_name}

    **Options List**: {options_list}

    **Your Response**:
    """

def get_keywords_prompt(content):
    """Creates a focused prompt for generating keywords."""
    return f"""
    Perform a deep analysis of the following 'Page Content'. Your task is to generate a list of exactly 20 comma-separated, unique technical keywords that are central to the document.

    **Critical Formatting Rule**:
    - Your response MUST ONLY be the comma-separated list of keywords. Do not add labels or explanations.

    **Exclusion Rules (Do Not Include)**:
    - Generic Terms: "documentation", "overview", "guide", "prerequisites", "steps", "introduction".
    - Broad Terms: "ports", "load balancers", "proxy servers", "customer-managed", "Alation Cloud Service", "data catalog".
    - UI References: "toggle", "button", "click", "Preview", "Import", "Run".
    - Placeholders: "table name", "S3 bucket name", "SQL template".
    - SQL Keywords or Release Status Terms.

    **Page Content**:
    ---
    {content[:4000]}
    ---

    **Your Response (comma-separated keywords only)**:
    """

# --- AI API CALLER (REVISED) ---

def call_ai_provider(prompt, api_key, provider, hf_model_id=None):
    """A single function to handle calls to any selected AI provider."""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            return response.text.strip()
        elif provider == "OpenAI (GPT-4)":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}]
            )
            return response.choices[0].message.content.strip()
        elif provider == "Hugging Face":
            client = InferenceClient(token=api_key)
            response = client.text_generation(prompt, model=hf_model_id, max_new_tokens=256)
            return response.strip()
    except Exception as e:
        st.warning(f"AI API call failed: {e}")
        return "" # Return empty string on failure

# --- DATA ENRICHMENT ORCHESTRATOR (REVISED) ---

def enrich_data_with_ai(dataframe, user_roles, topics, functional_areas, api_key, provider, hf_model_id=None):
    df_to_process = dataframe.copy()
    if 'Functional Area' not in df_to_process.columns:
        df_to_process['Functional Area'] = ''
    if 'Keywords' not in df_to_process.columns:
        df_to_process['Keywords'] = ''
        
    total_rows = len(df_to_process)
    pb = st.progress(0, f"Starting AI enrichment for {total_rows} rows...")

    for index, row in df_to_process.iterrows():
        # Update progress bar
        pb.progress((index + 1) / total_rows, f"Processing row {index + 1}/{total_rows}...")
        content = row['Page Content']

        # Task 1: Fill User Role if blank
        if pd.isna(row['User Role']) or row['User Role'] == '':
            prompt = get_mapping_prompt(content, 'User Role', user_roles)
            ai_response = call_ai_provider(prompt, api_key, provider, hf_model_id)
            df_to_process.loc[index, 'User Role'] = ai_response
            time.sleep(1) # Add a small delay to avoid hitting rate limits

        # Task 2: Fill Topics if blank
        if pd.isna(row['Topics']) or row['Topics'] == '':
            prompt = get_mapping_prompt(content, 'Topics', topics)
            ai_response = call_ai_provider(prompt, api_key, provider, hf_model_id)
            df_to_process.loc[index, 'Topics'] = ai_response
            time.sleep(1)

        # Task 3: Always map Functional Area
        prompt = get_mapping_prompt(content, 'Functional Area', functional_areas)
        ai_response = call_ai_provider(prompt, api_key, provider, hf_model_id)
        df_to_process.loc[index, 'Functional Area'] = ai_response
        time.sleep(1)
        
        # Task 4: Always generate Keywords
        prompt = get_keywords_prompt(content)
        ai_response = call_ai_provider(prompt, api_key, provider, hf_model_id)
        # Enclose in quotes for CSV compatibility
        df_to_process.loc[index, 'Keywords'] = f'"{ai_response}"'
        time.sleep(1)

    return df_to_process

# --- Utility and Scraping Functions (No changes needed) ---
@st.cache_data
def analyze_page_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text().strip() if soup.find('title') else 'No Title Found'
        return soup, title
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
        return None, "Fetch Error"

def get_deployment_type_from_scraping(soup):
    if not soup: return ""
    if soup.find('p', class_='cloud-label') and soup.find('p', class_='on-prem-label'):
        return "Alation Cloud Service, Customer Managed"
    if soup.find('p', class_='cloud-label'): return "Alation Cloud Service"
    if soup.find('p', class_='on-prem-label'): return "Customer Managed"
    return ""

def extract_main_content(soup):
    if not soup: return "Content Not Available"
    main_content = soup.find('article') or soup.find('main') or soup.body
    if main_content:
        for element in main_content.find_all(['nav', 'header', 'footer', 'aside']):
            element.decompose()
        return main_content.get_text(separator=' ', strip=True)
    return "Main Content Not Found"

def find_items_in_text(text, items):
    if not isinstance(text, str): return ""
    found_items = sorted([item for item in items if re.search(r'\b' + re.escape(item) + r'\b', text, re.IGNORECASE)])
    return ", ".join(found_items)

# --- Streamlit UI (No major changes, just ensuring it calls the new orchestrator) ---
st.set_page_config(layout="wide")
st.title("📄 AI-Powered Content Mapper (v2 - Robust)")
st.markdown("A five-step tool to scrape, map, and enrich web content using focused AI tasks.")

# Initialize session state 
if 'df1' not in st.session_state: st.session_state.df1 = pd.DataFrame()
if 'df2' not in st.session_state: st.session_state.df2 = pd.DataFrame()
if 'df3' not in st.session_state: st.session_state.df3 = pd.DataFrame()
if 'df_final' not in st.session_state: st.session_state.df_final = pd.DataFrame()
if 'user_roles' not in st.session_state: st.session_state.user_roles = None
if 'topics' not in st.session_state: st.session_state.topics = None
if 'functional_areas' not in st.session_state: st.session_state.functional_areas = None

# Step 1: Scrape URLs
with st.expander("Step 1: Scrape URLs and Content", expanded=True):
    urls_file = st.file_uploader("Upload URLs File (.txt)", key="step1")
    if st.button("🚀 Scrape URLs", type="primary"):
        if urls_file:
            urls = [line.strip() for line in io.StringIO(urls_file.getvalue().decode("utf-8")) if line.strip()]
            results, pb = [], st.progress(0, "Starting...")
            for i, url in enumerate(urls):
                pb.progress((i + 1) / len(urls), f"Processing URL {i+1}/{len(urls)}...")
                soup, title = analyze_page_content(url)
                data = {'Page Title': title, 'Page URL': url}
                if soup:
                    data.update({'Deployment Type': get_deployment_type_from_scraping(soup), 'Page Content': extract_main_content(soup)})
                else:
                    data.update({'Deployment Type': 'Fetch Error', 'Page Content': 'Fetch Error'})
                results.append(data)
            st.session_state.df1 = pd.DataFrame(results)
            st.session_state.df2, st.session_state.df3, st.session_state.df_final = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            st.session_state.user_roles, st.session_state.topics, st.session_state.functional_areas = None, None, None
            st.success("✅ Step 1 complete!")
        else:
            st.warning("⚠️ Please upload a URLs file.")

# Step 2: User Roles
with st.expander("Step 2: Upload and Map User Roles", expanded=True):
    is_disabled = st.session_state.df1.empty
    if is_disabled: st.info("Complete Step 1 to begin.")
    roles_file = st.file_uploader("Upload User Roles File (.txt)", key="step2", disabled=is_disabled)
    if st.button("🗺️ Map User Roles", disabled=is_disabled):
        if roles_file:
            st.session_state.user_roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]
            if st.session_state.user_roles:
                df = st.session_state.df1.copy()
                df['User Role'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, st.session_state.user_roles))
                st.session_state.df2 = df
                st.success("✅ Step 2 complete!")
            else: st.warning("⚠️ Roles file is empty.")
        else: st.warning("⚠️ Please upload a roles file.")

# Step 3: Topics
with st.expander("Step 3: Upload and Map Topics", expanded=True):
    is_disabled = st.session_state.df2.empty
    if is_disabled: st.info("Complete Step 2 to proceed.")
    topics_file = st.file_uploader("Upload Topics File (.txt)", key="step3", disabled=is_disabled)
    if st.button("🏷️ Map Topics", disabled=is_disabled):
        if topics_file:
            st.session_state.topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
            if st.session_state.topics:
                df = st.session_state.df2.copy()
                df['Topics'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, st.session_state.topics))
                st.session_state.df3 = df
                st.success("✅ Step 3 complete!")
            else: st.warning("⚠️ Topics file is empty.")
        else: st.warning("⚠️ Please upload a topics file.")

# Step 4: Functional Areas
with st.expander("Step 4: Upload Functional Areas", expanded=True):
    is_disabled = st.session_state.df3.empty
    if is_disabled: st.info("Complete Step 3 to proceed.")
    areas_file = st.file_uploader("Upload Functional Areas File (.txt)", key="step4", disabled=is_disabled)
    if areas_file is not None and not is_disabled:
        st.session_state.functional_areas = [line.strip() for line in io.StringIO(areas_file.getvalue().decode("utf-8")) if line.strip()]
        if st.session_state.functional_areas:
            st.success(f"✅ Step 4 complete! Loaded {len(st.session_state.functional_areas)} functional areas.")
        else:
            st.warning("⚠️ Functional areas file is empty.")

# Step 5: AI Enrichment
with st.expander("Step 5: Enrich Data with AI", expanded=True):
    all_data_loaded = st.session_state.user_roles and st.session_state.topics and st.session_state.functional_areas
    is_disabled = st.session_state.df3.empty or not all_data_loaded
    if is_disabled: st.info("Complete Steps 1-4 to enable AI enrichment.")

    ai_provider = st.selectbox("Choose AI Provider", ["Google Gemini", "OpenAI (GPT-4)", "Hugging Face"], disabled=is_disabled)
    api_key_label = "API Key" if ai_provider != "Hugging Face" else "Hugging Face User Access Token"
    api_key = st.text_input(f"Enter your {api_key_label}", type="password", disabled=is_disabled)
    hf_model_id = None
    if ai_provider == "Hugging Face":
        hf_model_id = st.text_input("Enter Hugging Face Model ID", help="e.g., mistralai/Mistral-7B-Instruct-v0.2", disabled=is_disabled)

    if st.button("🤖 Fill Blanks with AI", disabled=is_disabled):
        if not api_key: st.warning(f"Please enter your {api_key_label}.")
        elif ai_provider == "Hugging Face" and not hf_model_id: st.warning("Please enter a Hugging Face Model ID.")
        else:
            with st.spinner("AI is processing... This may take several minutes."):
                st.session_state.df_final = enrich_data_with_ai(
                    st.session_state.df3,
                    st.session_state.user_roles,
                    st.session_state.topics,
                    st.session_state.functional_areas,
                    api_key, 
                    ai_provider, 
                    hf_model_id
                )
            st.success("✅ AI enrichment complete! The final report is ready.")

# Results Display
st.markdown("---")
st.subheader("📊 Results")
df_to_show = pd.DataFrame()
if not st.session_state.df_final.empty: df_to_show = st.session_state.df_final
elif not st.session_state.df3.empty: df_to_show = st.session_state.df3
elif not st.session_state.df2.empty: df_to_show = st.session_state.df2
elif not st.session_state.df1.empty: df_to_show = st.session_state.df1

if not df_to_show.empty:
    final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords']
    display_columns = [col for col in final_columns if col in df_to_show.columns]
    st.dataframe(df_to_show[display_columns])
    csv_data = df_to_show[display_columns].to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Download Report (CSV)", csv_data, "enriched_report.csv", "text/csv")
else:
    st.write("Upload a file in Step 1 to begin.")
