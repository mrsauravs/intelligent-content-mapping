import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import re
import time
import google.generativeai as genai
from openai import OpenAI
from huggingface_hub import InferenceClient

# A set of common English stop words to filter from the final keyword list
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "with", "by", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "can", "should", "would", "could", "its", "it", "i", "me", "my", "we", "our", "you", "your"
])


# --- FOCUSED AI PROMPT FUNCTIONS ---

def get_deployment_type_prompt(content):
    """Creates a focused prompt for determining the deployment type."""
    return f"""
    Analyze the following 'Page Content'. Your task is to determine the correct deployment type.

    **Instructions**:
    - The deployment type must be one of these three options: "Alation Cloud Service", "Customer Managed", or "Alation Cloud Service, Customer Managed".
    - Your response MUST ONLY be the single most appropriate option from that list.
    - If you cannot determine the type, respond with an empty string.

    **Page Content**:
    ---
    {content[:4000]}
    ---

    **Your Response (choose one from the list)**:
    """

def get_mapping_prompt(content, column_name, options_list, url=None):
    """Creates a focused prompt for mapping content to a list of options."""
    
    additional_instructions = ""
    if column_name == 'User Role':
        additional_instructions = """- If your analysis suggests both "Steward" and "Composer" are relevant, you MUST also include "Server Admin" and "Catalog Admin" in your response."""
    elif column_name == 'Topics':
        auth_instruction = """- If the content discusses user authentication (e.g., SAML, SSO, SCIM, LDAP, login procedures), you MUST include "User Accounts" as one of the topics."""
        url_instruction = ""
        if url and "installconfig/Update/" in url:
            url_instruction = """\n- The URL for this page contains 'installconfig/Update/'. Therefore, you MUST include "Customer Managed Server Update" as one of the topics in your response."""
        additional_instructions = auth_instruction + url_instruction

    return f"""
    Analyze the following 'Page Content'. Your task is to select the most relevant term(s) from the provided 'Options List' that accurately describe the content.

    **Instructions**:
    - If the column is 'Functional Area', select only the single best option.
    - For other columns, you can select multiple options if they are all relevant.
    {additional_instructions}
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

def get_prose_keywords_prompt(content):
    """Creates a refined prompt for extracting keywords from prose."""
    return f"""
    Analyze the following prose from a technical document. Your task is to extract up to 5 specific, technical nouns or noun phrases representing features, components, or technologies.

    **Critical Rules**:
    - Your response MUST ONLY be a comma-separated list.
    - Keywords should ideally be 1-3 words long. Avoid long descriptive phrases.
    - Do not include common English stop words (e.g., 'the', 'is', 'for', 'does').
    - Do not include vague internal identifiers (e.g., 'pg-1', 'server-01').

    **Prose Content**:
    ---
    {content[:3000]}
    ---

    **Your Response (up to 5 keywords)**:
    """

def get_titles_keywords_prompt(titles_list):
    """Creates a refined prompt for extracting keywords from section titles."""
    titles_str = ", ".join(titles_list)
    return f"""
    Analyze the following list of section titles. Your task is to extract up to 5 specific, technical nouns or noun phrases representing features, components, or technologies.

    **Critical Rules**:
    - Your response MUST ONLY be a comma-separated list.
    - Keywords should ideally be 1-3 words long. Avoid long descriptive phrases.
    - Do not include common English stop words (e.g., 'the', 'is', 'for', 'does').
    - Do not include vague internal identifiers (e.g., 'pg-1', 'server-01').

    **Section Titles**:
    ---
    {titles_str}
    ---

    **Your Response (up to 5 keywords)**:
    """

def get_code_keywords_prompt(code_content):
    """Creates a refined prompt for extracting keywords from code blocks."""
    return f"""
    Analyze the following code snippets from a technical document. Your task is to extract up to 5 of the most relevant technical keywords. Keywords can include important filenames, commands, or technologies.

    **Critical Rules**:
    - Your response MUST ONLY be a comma-separated list.
    - Keywords should ideally be 1-3 words long.

    **Code Content**:
    ---
    {code_content[:3000]}
    ---

    **Your Response (up to 5 keywords)**:
    """

# --- CENTRALIZED AI API CALLER WITH SANITIZATION ---

def call_ai_provider(prompt, api_key, provider, hf_model_id=None):
    response_text = ""
    try:
        if provider == "Google Gemini":
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash-latest')
            response = model.generate_content(prompt)
            response_text = response.text
        elif provider == "OpenAI (GPT-4)":
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": prompt}])
            response_text = response.choices[0].message.content
        elif provider == "Hugging Face":
            client = InferenceClient(token=api_key)
            response = client.text_generation(prompt, model=hf_model_id, max_new_tokens=128)
            response_text = response
    except Exception as e:
        st.warning(f"AI API call failed: {e}")
        return ""
    sanitized_text = response_text.strip().replace('"', '').replace("'", "")
    return sanitized_text

# --- DATA ENRICHMENT ORCHESTRATOR ---

def enrich_data_with_ai(dataframe, user_roles, topics, functional_areas, api_key, provider, hf_model_id=None):
    df_to_process = dataframe.copy()
    
    # Add defensive checks to prevent KeyErrors
    if 'Deployment Type' not in df_to_process.columns:
        df_to_process['Deployment Type'] = ''
    if 'Functional Area' not in df_to_process.columns: 
        df_to_process['Functional Area'] = ''
    if 'Keywords' not in df_to_process.columns: 
        df_to_process['Keywords'] = ''
        
    total_rows = len(df_to_process)
    pb = st.progress(0, f"Starting AI enrichment for {total_rows} rows...")

    for index, row in df_to_process.iterrows():
        pb.progress((index + 1) / total_rows, f"Processing row {index + 1}/{total_rows}...")
        
        content = row['Page Content']
        url = row['Page URL']
        
        if pd.isna(row['Deployment Type']) or row['Deployment Type'] == '':
            prompt = get_deployment_type_prompt(content)
            df_to_process.loc[index, 'Deployment Type'] = call_ai_provider(prompt, api_key, provider, hf_model_id)
            time.sleep(1)
            
        if pd.isna(row['User Role']) or row['User Role'] == '':
            prompt = get_mapping_prompt(content, 'User Role', user_roles)
            df_to_process.loc[index, 'User Role'] = call_ai_provider(prompt, api_key, provider, hf_model_id)
            time.sleep(1)

        if pd.isna(row['Topics']) or row['Topics'] == '':
            prompt = get_mapping_prompt(content, 'Topics', topics, url=url)
            df_to_process.loc[index, 'Topics'] = call_ai_provider(prompt, api_key, provider, hf_model_id)
            time.sleep(1)

        prompt = get_mapping_prompt(content, 'Functional Area', functional_areas)
        ai_response = call_ai_provider(prompt, api_key, provider, hf_model_id)
        df_to_process.loc[index, 'Functional Area'] = ai_response.split(',')[0].strip() if ',' in ai_response else ai_response
        time.sleep(1)

        # Structured keyword generation pipeline
        all_keywords = []
        # 1. Get keywords from prose
        if pd.notna(row['Page Content']) and row['Page Content']:
            prompt = get_prose_keywords_prompt(row['Page Content'])
            prose_keys = call_ai_provider(prompt, api_key, provider, hf_model_id)
            all_keywords.extend([k.strip() for k in prose_keys.split(',') if k.strip()])
            time.sleep(1)
        # 2. Get keywords from titles
        if pd.notna(row['Section Titles']) and row['Section Titles']:
            prompt = get_titles_keywords_prompt(row['Section Titles'].split(','))
            title_keys = call_ai_provider(prompt, api_key, provider, hf_model_id)
            all_keywords.extend([k.strip() for k in title_keys.split(',') if k.strip()])
            time.sleep(1)
        # 3. Get keywords from code
        if pd.notna(row['Code Content']) and row['Code Content']:
            prompt = get_code_keywords_prompt(row['Code Content'])
            code_keys = call_ai_provider(prompt, api_key, provider, hf_model_id)
            all_keywords.extend([k.strip() for k in code_keys.split(',') if k.strip()])
            time.sleep(1)
        # 4. Get keywords from URL (programmatic)
        url_keys = re.findall(r'(?i)(V\s?R?\d+)\b', url)
        all_keywords.extend(url_keys)

        # Remove duplicates case-insensitively, preserving original case of first occurrence
        unique_keywords_cased = []
        seen_keywords_lower = set()
        for keyword in all_keywords:
            lowered_keyword = keyword.lower()
            if lowered_keyword not in seen_keywords_lower:
                unique_keywords_cased.append(keyword)
                seen_keywords_lower.add(lowered_keyword)
        
        # Filter out keywords that are subsets of other keywords
        keywords_to_remove = set()
        for shorter_kw in unique_keywords_cased:
            for longer_kw in unique_keywords_cased:
                if shorter_kw != longer_kw and re.search(r'\b' + re.escape(shorter_kw) + r'\b', longer_kw, re.IGNORECASE):
                    keywords_to_remove.add(shorter_kw)
        
        subset_filtered_keywords = [kw for kw in unique_keywords_cased if kw not in keywords_to_remove]

        # Programmatically filter out stop words, vague identifiers, and dotfiles
        vague_identifier_pattern = re.compile(r'^[a-zA-Z]+-\d+$')
        final_keywords = []
        for kw in subset_filtered_keywords:
            if kw.lower() not in STOP_WORDS and not vague_identifier_pattern.match(kw) and not kw.startswith('.'):
                final_keywords.append(kw)
        
        df_to_process.loc[index, 'Keywords'] = f'"{", ".join(final_keywords)}"'
        
    return df_to_process

# --- UTILITY AND SCRAPING FUNCTIONS ---

@st.cache_data
def analyze_page_content(url):
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.find('title').get_text(strip=True) if soup.find('title') else 'No Title Found'
        return soup, title
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch {url}: {e}")
        return None, "Fetch Error"

def extract_structured_content(soup):
    """Separates a webpage's content into prose, titles, and code."""
    if not soup:
        return {'prose': "Content Not Available", 'titles': [], 'code': ""}

    selectors = ['article', 'main', 'div[role="main"]', '#main-content', '#content', '.main-content', '.content', '#main', '.main']
    main_content = None
    for selector in selectors:
        main_content = soup.select_one(selector)
        if main_content: break
    if not main_content: main_content = soup.body
        
    if not main_content:
        return {'prose': "Main Content Not Found", 'titles': [], 'code': ""}

    for element in main_content.find_all(['nav', 'header', 'footer', 'aside', 'script', 'style', 'form']):
        element.decompose()
    
    titles = [h.get_text(strip=True) for h in main_content.find_all(['h2', 'h3', 'h4', 'h5'])]
    code_content = " ".join([code.get_text() for code in main_content.find_all(['pre', 'code'])])

    for element in main_content.find_all(['h2', 'h3', 'h4', 'h5', 'pre', 'code']):
        element.decompose()
    
    prose = main_content.get_text(separator=' ', strip=True)

    return {'prose': prose, 'titles': titles, 'code': code_content}

def find_items_in_text(text, items):
    if not isinstance(text, str): return ""
    found_items = sorted([item for item in items if re.search(r'\b' + re.escape(item) + r'\b', text, re.IGNORECASE)])
    return ", ".join(found_items)

# --- STREAMLIT UI ---

st.set_page_config(layout="wide")
st.title("📄 AI-Powered Content Mapper")
st.markdown("A five-step tool to scrape, map, and enrich web content using focused AI tasks.")

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
                    structured_content = extract_structured_content(soup)
                    data['Page Content'] = structured_content['prose']
                    data['Section Titles'] = ",".join(structured_content['titles'])
                    data['Code Content'] = structured_content['code']
                else:
                    data.update({'Page Content': 'Fetch Error', 'Section Titles': '', 'Code Content': ''})
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
                
                def augment_user_roles(roles_str):
                    if not isinstance(roles_str, str): return ""
                    roles_list = [r.strip() for r in roles_str.split(',') if r.strip()]
                    if "Steward" in roles_list and "Composer" in roles_list:
                        roles_list.extend(["Server Admin", "Catalog Admin"])
                        return ", ".join(sorted(list(set(roles_list))))
                    return roles_str
                df['User Role'] = df['User Role'].apply(augment_user_roles)
                
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
                
                def add_topic(current_topics, new_topic):
                    if not isinstance(current_topics, str): current_topics = ""
                    if new_topic in current_topics: return current_topics
                    return f"{current_topics}, {new_topic}" if current_topics else new_topic

                def augment_topics(row):
                    topics = row['Topics']
                    if "installconfig/Update/" in row['Page URL']:
                        topics = add_topic(topics, "Customer Managed Server Update")
                    auth_keywords = ['authentication', 'saml', 'sso', 'scim', 'ldap']
                    content_lower = str(row['Page Content']).lower()
                    if any(keyword in content_lower for keyword in auth_keywords):
                        topics = add_topic(topics, "User Accounts")
                    return topics
                
                df['Topics'] = df.apply(augment_topics, axis=1)

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

# --- RESULTS DISPLAY ---

st.markdown("---")
st.subheader("📊 Results")
df_to_show = pd.DataFrame()
if not st.session_state.df_final.empty:
    df_to_show = st.session_state.df_final
elif not st.session_state.df3.empty:
    df_to_show = st.session_state.df3
elif not st.session_state.df2.empty:
    df_to_show = st.session_state.df2
elif not st.session_state.df1.empty:
    df_to_show = st.session_state.df1

if not df_to_show.empty:
    final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords']
    display_columns = [col for col in final_columns if col in df_to_show.columns]
    
    st.dataframe(df_to_show[display_columns])
    csv_data = df_to_show[display_columns].to_csv(index=False).encode('utf-8-sig')
    st.download_button("📥 Download Report (CSV)", csv_data, "enriched_report.csv", "text/csv")
else:
    st.write("Upload a file in Step 1 to begin.")
