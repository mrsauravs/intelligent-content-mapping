import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import re
import json
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import InferenceClient

# --- DYNAMIC MASTER PROMPT FUNCTION ---
# This function now builds the prompt dynamically using the uploaded lists.
def get_master_prompt(user_roles, topics, functional_areas, page_content):
    return f"""
You are an expert content analyst and data enrichment specialist.
Your task is to analyze the provided page content and CSV data to fill in the blank fields.
Accuracy and relevance are critical.

**Your Inputs**:

1.  A single row of CSV data with some potentially blank columns.
2.  The full text content from the page URL, provided below.
3.  Lists of approved terms for 'User Role', 'Topics', and 'Functional Area'.

**Your Goal**:

- Analyze the page content and complete the CSV data for the given row.
- Follow these steps precisely:

    - Step 1: Analyze Page Content
        - Use the 'Page Content' provided below as the source of truth for your analysis. Do NOT re-scrape the URL.

    - Step 2: Fill Missing Columns
        - For any blank columns (`Deployment Type`, `User Role`, `Topics`), fill them based on your analysis of the 'Page Content'.
        - Do not modify columns that already have a value.

    - Step 3: Map Contextual Metadata
        - For the `Functional Area` column, select the SINGLE most relevant term from the provided 'Functional Area List'.
        - For the `User Role` column (if blank), select one or more relevant terms from the 'User Role List'.
        - For the `Topics` column (if blank), select one or more relevant terms from the 'Topics List'.

    - Step 4: Generate Keywords
        - Generate a list of exactly 20 comma-separated, unique technical keywords from the 'Page Content'.
        - The entire list must be enclosed in a single pair of double quotes (e.g., "keyword1, keyword2, ...").
        - Exclude generic terms like "documentation", "overview", "guide", etc.

**Constraint Lists**:

- User Role List: {user_roles}
- Topics List: {topics}
- Functional Area List: {functional_areas}

**Page Content to Analyze**:
---
{page_content}
---

**Final Output Instructions**:
- Your final output must be only the completed single row of CSV data.
- Do not provide explanations. Enclose the output in a markdown code block (```csv ... ```).
- The header must be: `Page Title,Page URL,Deployment Type,User Role,Functional Area,Topics,Keywords`
"""

# --- Utility and Scraping Functions (No changes needed here) ---
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

# --- AI Enrichment Functions (Modified to accept lists) ---
def process_ai_response(response_text, url):
    csv_match = re.search(r'```csv\n(.*?)\n```', response_text, re.DOTALL)
    if csv_match:
        csv_data = csv_match.group(1)
        try:
            enriched_df = pd.read_csv(io.StringIO(csv_data))
            if not enriched_df.empty: return enriched_df.iloc[0]
        except Exception as e:
            st.warning(f"Could not parse CSV from AI response for {url}: {e}")
    else:
        st.warning(f"Could not find CSV in AI response for {url}")
    return None

def call_gemini_api(api_key, prompt):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    response = model.generate_content(prompt)
    return response.text

def call_openai_api(api_key, prompt):
    client = OpenAI(api_key=api_key)
    system_prompt, user_data = prompt.split("Here is the single row of CSV data to process:")
    response = client.chat.completions.create(model="gpt-4-turbo", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": "Here is the single row of CSV data to process:" + user_data}])
    return response.choices[0].message.content

def call_huggingface_api(api_token, model_id, prompt):
    client = InferenceClient(token=api_token)
    response = client.text_generation(prompt, model=model_id, max_new_tokens=512)
    return response

# MODIFIED: The function now requires the lists of terms to build the prompt.
def enrich_data_with_ai(dataframe, user_roles, topics, functional_areas, api_key, provider, hf_model_id=None):
    df_to_process = dataframe.copy()
    total_rows = len(df_to_process)
    pb = st.progress(0, f"Starting AI enrichment for {total_rows} rows...")

    for index, row in df_to_process.iterrows():
        pb.progress((index + 1) / total_rows, f"Processing row {index + 1}/{total_rows}...")
        try:
            # MODIFIED: Dynamically generate the prompt with context.
            full_prompt = get_master_prompt(user_roles, topics, functional_areas, row['Page Content'])
            
            # Prepare the single row of CSV data for the prompt
            header = "Page Title,Page URL,Deployment Type,User Role,Topics"
            row_as_csv_string = row[header.split(',')].to_frame().T.to_csv(header=True, index=False)
            final_prompt_to_send = full_prompt + f"\n\nHere is the single row of CSV data to process:\n```csv\n{row_as_csv_string}```"
            
            response_text = ""
            if provider == "Google Gemini": response_text = call_gemini_api(api_key, final_prompt_to_send)
            elif provider == "OpenAI (GPT-4)": response_text = call_openai_api(api_key, final_prompt_to_send)
            elif provider == "Hugging Face": response_text = call_huggingface_api(api_key, hf_model_id, final_prompt_to_send)
            
            enriched_row = process_ai_response(response_text, row['Page URL'])
            if enriched_row is not None:
                for col in enriched_row.index:
                    df_to_process.loc[index, col] = enriched_row[col]
        except Exception as e:
            st.error(f"An error occurred while processing {row['Page URL']}: {e}")
            continue
    return df_to_process

# --- Streamlit UI ---
st.set_page_config(layout="wide")
st.title("📄 AI-Powered Content Mapper")
st.markdown("A five-step tool to scrape, map, and enrich web content.")

# MODIFIED: Initialize session state for dataframes AND the uploaded lists.
if 'df1' not in st.session_state: st.session_state.df1 = pd.DataFrame()
if 'df2' not in st.session_state: st.session_state.df2 = pd.DataFrame()
if 'df3' not in st.session_state: st.session_state.df3 = pd.DataFrame()
if 'df_final' not in st.session_state: st.session_state.df_final = pd.DataFrame()
if 'user_roles' not in st.session_state: st.session_state.user_roles = None
if 'topics' not in st.session_state: st.session_state.topics = None
if 'functional_areas' not in st.session_state: st.session_state.functional_areas = None


# --- UI Step 1: Scrape URLs ---
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
            # Reset all subsequent steps and stored lists
            st.session_state.df2, st.session_state.df3, st.session_state.df_final = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            st.session_state.user_roles, st.session_state.topics, st.session_state.functional_areas = None, None, None
            st.success("✅ Step 1 complete!")
        else:
            st.warning("⚠️ Please upload a URLs file.")

# --- UI Step 2: Upload and Map User Roles ---
with st.expander("Step 2: Upload and Map User Roles", expanded=True):
    is_disabled = st.session_state.df1.empty
    if is_disabled: st.info("Complete Step 1 to begin.")
    
    roles_file = st.file_uploader("Upload User Roles File (.txt)", key="step2", disabled=is_disabled)
    if st.button("🗺️ Map User Roles", disabled=is_disabled):
        if roles_file:
            # MODIFIED: Store the list of roles in session_state
            st.session_state.user_roles = [line.strip() for line in io.StringIO(roles_file.getvalue().decode("utf-8")) if line.strip()]
            if st.session_state.user_roles:
                df = st.session_state.df1.copy()
                df['User Role'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, st.session_state.user_roles))
                st.session_state.df2 = df
                st.success("✅ Step 2 complete!")
            else: st.warning("⚠️ Roles file is empty.")
        else: st.warning("⚠️ Please upload a roles file.")

# --- UI Step 3: Upload and Map Topics ---
with st.expander("Step 3: Upload and Map Topics", expanded=True):
    is_disabled = st.session_state.df2.empty
    if is_disabled: st.info("Complete Step 2 to proceed.")

    topics_file = st.file_uploader("Upload Topics File (.txt)", key="step3", disabled=is_disabled)
    if st.button("🏷️ Map Topics", disabled=is_disabled):
        if topics_file:
            # MODIFIED: Store the list of topics in session_state
            st.session_state.topics = [line.strip() for line in io.StringIO(topics_file.getvalue().decode("utf-8")) if line.strip()]
            if st.session_state.topics:
                df = st.session_state.df2.copy()
                df['Topics'] = df['Page Content'].apply(lambda txt: find_items_in_text(txt, st.session_state.topics))
                st.session_state.df3 = df
                st.success("✅ Step 3 complete!")
            else: st.warning("⚠️ Topics file is empty.")
        else: st.warning("⚠️ Please upload a topics file.")

# --- NEW Step 4: Upload Functional Areas ---
with st.expander("Step 4: Upload Functional Areas", expanded=True):
    is_disabled = st.session_state.df3.empty
    if is_disabled: st.info("Complete Step 3 to proceed.")
    
    areas_file = st.file_uploader("Upload Functional Areas File (.txt)", key="step4", disabled=is_disabled)
    if areas_file is not None and not is_disabled:
        # MODIFIED: Store functional areas list in session_state as soon as it's uploaded
        st.session_state.functional_areas = [line.strip() for line in io.StringIO(areas_file.getvalue().decode("utf-8")) if line.strip()]
        if st.session_state.functional_areas:
            st.success(f"✅ Step 4 complete! Loaded {len(st.session_state.functional_areas)} functional areas.")
        else:
            st.warning("⚠️ Functional areas file is empty.")

# --- UI Step 5: Enrich Data with AI ---
with st.expander("Step 5: Enrich Data with AI", expanded=True):
    # MODIFIED: The AI step is disabled until all three lists are loaded.
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
                # MODIFIED: Pass the stored lists to the AI function.
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

st.markdown("---")
st.subheader("📊 Results")

# Determine which dataframe to show
df_to_show = pd.DataFrame()
if not st.session_state.df_final.empty: df_to_show = st.session_state.df_final
elif not st.session_state.df3.empty: df_to_show = st.session_state.df3
elif not st.session_state.df2.empty: df_to_show = st.session_state.df2
elif not st.session_state.df1.empty: df_to_show = st.session_state.df1

# Display results
if not df_to_show.empty:
    final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords']
    display_columns = [col for col in final_columns if col in df_to_show.columns]
    
    st.dataframe(df_to_show[display_columns])
    csv_data = df_to_show[display_columns].to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Download Report (CSV)", csv_data, "enriched_report.csv", "text/csv")
else:
    st.write("Upload a file in Step 1 to begin.")
