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
from collections import defaultdict

# A set of common English stop words to filter from the final keyword list
STOP_WORDS = set([
    "a", "an", "the", "and", "or", "but", "if", "in", "on", "at", "to", "for", "with", "by", "of",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "can", "should", "would", "could", "its", "it", "i", "me", "my", "we", "our", "you", "your"
])

# A set of overly generic standalone concepts to filter from the final keyword list
GENERIC_CONCEPTS_TO_REMOVE = {"version", "memory", "alation", "db", "database", "navigation bar"}

# A set of common command-line utilities to filter from the final keyword list
COMMON_COMMANDS_TO_REMOVE = {"sudo", "rpm", "sh", "bash", "chmod", "chown", "df", "dpkg"}


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
    - Prefer the full, specific names of technologies or frameworks (e.g., "Open Connector Framework" is better than "Open Connector").
    - Avoid generic acronyms (e.g., "DB") if a more specific term (e.g., "MongoDB") is available.
    - Do not include common English stop words (e.g., 'the', 'is', 'for', 'does').
    - Do not include vague internal identifiers (e.g., 'pg-1', 'server-01'), example hostnames (e.g., mycatalog.alation-test.com), or overly generic terms ("Version", "Memory", "Alation").

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
    - Prefer the full, specific names of technologies or frameworks (e.g., "Open Connector Framework" is better than "Open Connector").
    - Avoid generic acronyms (e.g., "DB") if a more specific term (e.g., "MongoDB") is available.
    - Do not include common English stop words (e.g., 'the', 'is', 'for', 'does').
    - Do not include vague internal identifiers (e.g., 'pg-1', 'server-01') or overly generic terms ("Version", "Memory", "Alation").

    **Section Titles**:
    ---
    {titles_str}
    ---

    **Your Response (up to 5 keywords)**:
    """

def get_code_keywords_prompt(code_content):
    """Creates a refined prompt for extracting keywords from code blocks."""
    return f"""
    Analyze the following code snippets. Your task is to extract up to 5 of the most relevant technical keywords. Keywords should represent technologies or concepts.

    **Critical Rules**:
    - Your response MUST ONLY be a comma-separated list.
    - Keywords should ideally be 1-3 words long.
    - You MUST exclude all file paths (e.g., /opt/alation/) and specific script names (e.g., reset_checkpoint.py).
    - Avoid generic command-line utilities like `sudo` or `rpm`, and standalone flags like `-h` or `-i`.

    **Code Content**:
    ---
    {code_content[:3000]}
    ---

    **Your Response (up to 5 keywords)**:
    """

def get_seo_keywords_prompt(existing_keywords, page_title):
    """Creates a prompt for generating SEO-relevant keywords based on existing context."""
    existing_keywords_str = ", ".join(existing_keywords)
    return f"""
    Analyze the following page title and list of existing technical keywords. Your task is to act as an SEO expert and suggest up to 5 additional, related technical keywords or concepts that a user might search for to find this content, even if they are not in the text.

    **Critical Rules**:
    - Your response MUST ONLY be a comma-separated list.
    - The suggested keywords should be technically relevant and complementary to the existing keywords.
    - Do not repeat any keywords already present in the "Existing Keywords" list.
    - Keywords should ideally be 1-3 words long.
    - Do not suggest generic terms, stop words, or file paths.

    **Page Title**:
    ---
    {page_title}
    ---

    **Existing Keywords (for context)**:
    ---
    {existing_keywords_str}
    ---

    **Your Response (up to 5 additional SEO keywords)**:
    """

def get_disambiguation_prompt(ambiguous_keyword, page_titles):
    """Creates a prompt to generate more specific, differentiating keywords."""
    titles_str = "\n".join([f"- {title}" for title in page_titles])
    return f"""
    The following technical keyword is ambiguous because it is associated with multiple distinct pages: "{ambiguous_keyword}".

    Your task is to analyze the page titles below and suggest a more specific, unique keyword for each page to help differentiate them in search results.

    **Page Titles Associated with "{ambiguous_keyword}"**:
    ---
    {titles_str}
    ---

    **Instructions**:
    - For each page title, provide one more specific keyword or phrase.
    - Format your response as a list, with each line containing the original page title followed by "::", and then your new suggested keyword.
    - Example: `Original Page Title::Suggested Differentiating Keyword`

    **Your Response**:
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
            response = client.text_generation(prompt, model=hf_model_id, max_new_tokens=256)
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
        
        # 5. Get SEO keywords from AI's knowledge base
        if all_keywords:
            page_title = row['Page Title']
            prompt = get_seo_keywords_prompt(all_keywords, page_title)
            seo_keys = call_ai_provider(prompt, api_key, provider, hf_model_id)
            all_keywords.extend([k.strip() for k in seo_keys.split(',') if k.strip()])
            time.sleep(1)

        # Final Cleaning and Deduplication
        df_to_process.loc[index, 'Keywords'] = f'"{", ".join(clean_and_filter_keywords(all_keywords, row, df_to_process))}"'
        
    return df_to_process

def clean_and_filter_keywords(keywords_list, current_row, dataframe):
    """A comprehensive function to clean, deduplicate, and filter keywords based on multiple rules."""
    
    # Create a set of all existing metadata terms for this row to avoid duplication
    existing_metadata_terms = set()
    for col in ['Deployment Type', 'User Role', 'Topics', 'Functional Area']:
        if col in dataframe.columns and pd.notna(current_row.get(col)):
            terms = [term.strip().lower() for term in current_row[col].split(',') if term.strip()]
            existing_metadata_terms.update(terms)

    # 1. Remove duplicates case-insensitively, preserving original case and handling spacing
    unique_keywords_cased = []
    seen_keywords_normalized = set()
    for keyword in keywords_list:
        normalized_keyword = keyword.lower().replace(" ", "")
        if normalized_keyword not in seen_keywords_normalized:
            unique_keywords_cased.append(keyword)
            seen_keywords_normalized.add(normalized_keyword)
    
    # 2. Robust subset filter
    sorted_by_len = sorted(unique_keywords_cased, key=len)
    subset_filtered_keywords = []
    for i, shorter_kw in enumerate(sorted_by_len):
        is_subset = False
        for longer_kw in sorted_by_len[i+1:]:
            if re.search(r'\b' + re.escape(shorter_kw) + r'\b', longer_kw, re.IGNORECASE):
                is_subset = True
                break
        if not is_subset:
            subset_filtered_keywords.append(shorter_kw)

    # 3. Final programmatic filtering for unwanted patterns
    vague_identifier_pattern = re.compile(r'^[a-zA-Z]+-\d+$')
    command_flag_pattern = re.compile(r'^--?[a-zA-Z0-9-]+$')
    placeholder_filename_pattern = re.compile(r'\w*####\.\w+')
    example_hostname_pattern = re.compile(r'.*\.(alation-test|example|your-company)\.com$')
    filepath_pattern = re.compile(r'.*/.*')
    filename_pattern = re.compile(r'.*\.(py|sh|log|conf|gz|deb|rpm|json|xml|yaml|yml)$')

    final_keywords = []
    for kw in subset_filtered_keywords:
        kw_lower = kw.lower()
        if (kw_lower not in STOP_WORDS and 
            kw_lower not in GENERIC_CONCEPTS_TO_REMOVE and
            kw_lower not in COMMON_COMMANDS_TO_REMOVE and
            kw_lower not in existing_metadata_terms and
            not vague_identifier_pattern.match(kw) and 
            not command_flag_pattern.match(kw) and
            not placeholder_filename_pattern.match(kw_lower) and
            not example_hostname_pattern.match(kw_lower) and
            not filepath_pattern.match(kw) and
            not filename_pattern.match(kw_lower) and
            not kw.startswith('.')):
            final_keywords.append(kw)
    
    return final_keywords

def analyze_and_refine_uniqueness(dataframe, api_key, provider, hf_model_id=None):
    """Analyzes keyword uniqueness across all pages and refines ambiguous keywords."""
    df = dataframe.copy()
    if 'Keywords' not in df.columns:
        return df, "No keywords to analyze."

    # Invert the data: map each keyword to a list of page titles
    keyword_to_pages = defaultdict(list)
    for index, row in df.iterrows():
        keywords = [k.strip() for k in row['Keywords'].replace('"', '').split(',') if k.strip()]
        for kw in keywords:
            keyword_to_pages[kw].append(row['Page Title'])
    
    # Identify ambiguous keywords (used on more than one page)
    ambiguous_keywords = {kw: pages for kw, pages in keyword_to_pages.items() if len(pages) > 1}
    
    if not ambiguous_keywords:
        df['Uniqueness Score'] = "100%"
        return df, "All keywords are unique. No refinement needed."
        
    st.warning(f"Found {len(ambiguous_keywords)} ambiguous keywords. Attempting to refine...")

    # AI-powered disambiguation
    refined_keywords_map = {} # {page_title: [new_keywords]}
    for kw, titles in ambiguous_keywords.items():
        prompt = get_disambiguation_prompt(kw, titles)
        response = call_ai_provider(prompt, api_key, provider, hf_model_id)
        time.sleep(1)
        for line in response.split('\n'):
            if '::' in line:
                title, new_keyword = line.split('::', 1)
                if title.strip() not in refined_keywords_map:
                    refined_keywords_map[title.strip()] = []
                refined_keywords_map[title.strip()].append(new_keyword.strip())

    # Update the dataframe with refined keywords
    for index, row in df.iterrows():
        if row['Page Title'] in refined_keywords_map:
            current_keywords = [k.strip() for k in row['Keywords'].replace('"', '').split(',') if k.strip()]
            new_suggestions = refined_keywords_map[row['Page Title']]
            # Remove ambiguous terms and add new, more specific ones
            ambiguous_for_this_page = [kw for kw in current_keywords if kw in ambiguous_keywords]
            updated_keywords = [kw for kw in current_keywords if kw not in ambiguous_for_this_page]
            updated_keywords.extend(new_suggestions)
            df.loc[index, 'Keywords'] = f'"{", ".join(list(dict.fromkeys(updated_keywords)))}"'
            
    # Recalculate uniqueness score
    keyword_to_pages_final = defaultdict(list)
    for index, row in df.iterrows():
        keywords = [k.strip() for k in row['Keywords'].replace('"', '').split(',') if k.strip()]
        for kw in keywords:
            keyword_to_pages_final[kw].append(row['Page Title'])
    
    df['Uniqueness Score'] = df.apply(lambda row: f"{calculate_uniqueness(row, keyword_to_pages_final):.0f}%", axis=1)
    
    return df, f"Refined {len(ambiguous_keywords)} ambiguous keywords."

def calculate_uniqueness(row, keyword_map):
    """Calculates the percentage of unique keywords for a given row."""
    keywords = [k.strip() for k in row['Keywords'].replace('"', '').split(',') if k.strip()]
    if not keywords:
        return 0
    unique_count = sum(1 for kw in keywords if len(keyword_map[kw]) == 1)
    return (unique_count / len(keywords)) * 100

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
st.markdown("A multi-step tool to scrape, map, enrich, and refine web content using focused AI tasks.")

if 'df1' not in st.session_state: st.session_state.df1 = pd.DataFrame()
if 'df2' not in st.session_state: st.session_state.df2 = pd.DataFrame()
if 'df3' not in st.session_state: st.session_state.df3 = pd.DataFrame()
if 'df_final' not in st.session_state: st.session_state.df_final = pd.DataFrame()
if 'df_refined' not in st.session_state: st.session_state.df_refined = pd.DataFrame()
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
            st.session_state.df2, st.session_state.df3, st.session_state.df_final, st.session_state.df_refined = pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
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
    if is_disabled: st.info("Complete Step 4 to proceed.")
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
            st.success("✅ AI enrichment complete! You can now download the report below or proceed to Step 6 for uniqueness analysis.")
            st.session_state.df_refined = pd.DataFrame() # Reset refined df

    if not st.session_state.df_final.empty:
        final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords']
        display_columns = [col for col in final_columns if col in st.session_state.df_final.columns]
        csv_data_step5 = st.session_state.df_final[display_columns].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Enriched Report (from Step 5)", csv_data_step5, "enriched_report_step5.csv", "text/csv")

# Step 6: Uniqueness Analysis
with st.expander("Step 6: Uniqueness Analysis and Refinement", expanded=True):
    is_disabled = st.session_state.df_final.empty
    if is_disabled: st.info("Complete Step 5 to enable this step.")

    ai_provider_step6 = st.selectbox("Choose AI Provider", ["Google Gemini", "OpenAI (GPT-4)", "Hugging Face"], disabled=is_disabled, key="step6_provider")
    api_key_label_step6 = "API Key" if ai_provider_step6 != "Hugging Face" else "Hugging Face User Access Token"
    api_key_step6 = st.text_input(f"Enter your {api_key_label_step6}", type="password", disabled=is_disabled, key="step6_apikey")
    hf_model_id_step6 = None
    if ai_provider_step6 == "Hugging Face":
        hf_model_id_step6 = st.text_input("Enter Hugging Face Model ID", help="e.g., mistralai/Mistral-7B-Instruct-v0.2", disabled=is_disabled, key="step6_hf_model")

    if st.button("🔍 Analyze and Refine Uniqueness", disabled=is_disabled):
        if not api_key_step6: st.warning(f"Please enter your {api_key_label_step6}.")
        elif ai_provider_step6 == "Hugging Face" and not hf_model_id_step6: st.warning("Please enter a Hugging Face Model ID.")
        else:
            with st.spinner("Analyzing keyword uniqueness and refining with AI..."):
                df_refined, message = analyze_and_refine_uniqueness(st.session_state.df_final, api_key_step6, ai_provider_step6, hf_model_id_step6)
                st.session_state.df_refined = df_refined
                st.success(f"✅ {message}")
    
    if not st.session_state.df_refined.empty:
        final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords', 'Uniqueness Score']
        display_columns = [col for col in final_columns if col in st.session_state.df_refined.columns]
        csv_data_step6 = st.session_state.df_refined[display_columns].to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 Download Refined Report (from Step 6)", csv_data_step6, "refined_report_step6.csv", "text/csv")


# --- RESULTS DISPLAY ---

st.markdown("---")
st.subheader("📊 Results Editor")
st.info("The table below is interactive. You can make manual edits to the data before downloading the final report.")

df_to_show = pd.DataFrame()
current_data_key = None

if not st.session_state.df_refined.empty:
    df_to_show = st.session_state.df_refined
    current_data_key = 'df_refined'
elif not st.session_state.df_final.empty:
    df_to_show = st.session_state.df_final
    current_data_key = 'df_final'
elif not st.session_state.df3.empty:
    df_to_show = st.session_state.df3
    current_data_key = 'df3'
elif not st.session_state.df2.empty:
    df_to_show = st.session_state.df2
    current_data_key = 'df2'
elif not st.session_state.df1.empty:
    df_to_show = st.session_state.df1
    current_data_key = 'df1'

if not df_to_show.empty:
    final_columns = ['Page Title', 'Page URL', 'Deployment Type', 'User Role', 'Functional Area', 'Topics', 'Keywords', 'Uniqueness Score']
    display_columns = [col for col in final_columns if col in df_to_show.columns]
    
    edited_df = st.data_editor(df_to_show[display_columns], key="data_editor", num_rows="dynamic")

    if st.button("💾 Save Manual Edits"):
        # When saved, update the session state with the edited dataframe
        if current_data_key:
            st.session_state[current_data_key] = edited_df
            st.success("Your edits have been saved! You can now download the updated report from the relevant step.")
            time.sleep(2) # Give user time to read the message
            st.rerun() # Rerun to reflect changes immediately
else:
    st.write("Upload a file in Step 1 to begin.")
