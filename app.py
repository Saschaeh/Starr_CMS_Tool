import streamlit as st
from PIL import Image
import io
import base64
import re
import zipfile
import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import streamlit.components.v1 as components

load_dotenv()

# Forward Streamlit Cloud secrets to env vars (before importing db)
try:
    if hasattr(st, 'secrets'):
        for key in ('TURSO_DB_URL', 'TURSO_AUTH_TOKEN', 'HF_API_TOKEN'):
            if key in st.secrets and key not in os.environ:
                os.environ[key] = st.secrets[key]
except Exception:
    pass

import db

# Initialize database on first run
db.init_db()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def resize_and_crop(img, target_width, target_height):
    """Resize and center-crop image to target dimensions."""
    img_width, img_height = img.size
    target_ratio = target_width / target_height
    img_ratio = img_width / img_height
    
    if img_ratio > target_ratio:
        new_height = img_height
        new_width = int(img_height * target_ratio)
    else:
        new_width = img_width
        new_height = int(img_width / target_ratio)
    
    left = (img_width - new_width) // 2
    top = (img_height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    
    img = img.crop((left, top, right, bottom))
    img = img.resize((target_width, target_height), Image.Resampling.LANCZOS)
    return img

def is_black_and_white(img, threshold=0.1):
    """Check if image is predominantly black and white."""
    img_rgb = img.convert('RGB')
    img_data = img_rgb.getdata()
    
    bw_count = 0
    for pixel in img_data:
        r, g, b = pixel
        if abs(r - g) < threshold * 255 and abs(g - b) < threshold * 255:
            bw_count += 1
    
    return bw_count / len(list(img_data)) > 0.8

def apply_black_overlay(img, opacity_percent):
    """Apply a semi-transparent black overlay to image."""
    img_rgba = img.convert('RGBA')
    overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, int(255 * opacity_percent / 100)))
    result = Image.alpha_composite(img_rgba, overlay)
    result_rgb = Image.new('RGB', result.size, (255, 255, 255))
    result_rgb.paste(result, mask=result.split()[3])
    return result_rgb

def get_base64_of_bin_file(bin_file_path):
    """Get base64 encoded string of a file."""
    try:
        with open(bin_file_path, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except FileNotFoundError:
        return ""

def copy_button(text, key):
    """Render a click-to-copy button using HTML/JS in an iframe."""
    b64 = base64.b64encode(text.encode()).decode()
    components.html(f"""
    <button id="btn_{key}" onclick="
        var text = atob('{b64}');
        navigator.clipboard.writeText(text).then(function() {{
            document.getElementById('btn_{key}').innerText = 'Copied!';
            document.getElementById('btn_{key}').style.background = '#2D7D46';
            document.getElementById('btn_{key}').style.borderColor = '#2D7D46';
            document.getElementById('btn_{key}').style.color = '#FFFFFF';
            setTimeout(function() {{
                document.getElementById('btn_{key}').innerText = 'Copy';
                document.getElementById('btn_{key}').style.background = 'transparent';
                document.getElementById('btn_{key}').style.borderColor = '#031E41';
                document.getElementById('btn_{key}').style.color = '#031E41';
            }}, 1500);
        }});
    " style="
        background: transparent; color: #031E41; border: 1.5px solid #031E41;
        padding: 0.4rem 1.2rem; border-radius: 4px; cursor: pointer; font-size: 0.85rem;
        font-weight: 500; letter-spacing: 0.3px; transition: all 0.2s ease;
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
        line-height: 1.6;
    "
    onmouseover="if(this.innerText==='Copy'){{this.style.background='#031E41';this.style.color='#FFFFFF';}}"
    onmouseout="if(this.innerText==='Copy'){{this.style.background='transparent';this.style.color='#031E41';}}"
    >Copy</button>
    """, height=50)

# ============================================================================
# HUGGING FACE ALT TEXT GENERATION (Free Inference API)
# ============================================================================

def generate_alt_text(pil_image):
    """Generate alt text from image using Hugging Face Inference API (Qwen Vision)."""
    api_token = st.session_state.get('hf_api_token', '')
    if not api_token:
        return None

    # Convert PIL image to base64
    img_buffer = io.BytesIO()
    pil_image.save(img_buffer, format='JPEG', quality=85)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

    try:
        client = InferenceClient(token=api_token)
        result = client.chat_completion(
            model="Qwen/Qwen2.5-VL-7B-Instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                    {"type": "text", "text": "Write a concise, descriptive alt text for this image suitable for ADA/WCAG compliance on a restaurant website. Focus on what is visually depicted. Return only the alt text, no quotes or extra formatting. Keep it under 125 characters."}
                ]
            }],
            max_tokens=100,
        )
        return result.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e).lower()
        if '401' in error_str or 'unauthorized' in error_str:
            st.warning("Alt text generation unavailable: Invalid HF token in .env file.")
        elif '403' in error_str or 'permission' in error_str:
            st.warning("HF token needs 'Inference Providers' permission. Update at huggingface.co/settings/tokens.")
        elif '503' in error_str or 'loading' in error_str:
            st.info("Alt text model is loading, please try again in a few seconds.")
        else:
            st.warning("Alt text generation temporarily unavailable.")
        return None

# ============================================================================
# WEBSITE SCRAPING
# ============================================================================

def scrape_website(url):
    """Scrape text content from a restaurant website URL."""
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return False, "", "Website took too long to respond. Please try again."
    except requests.exceptions.ConnectionError:
        return False, "", "Could not connect to website. Please check the URL."
    except requests.exceptions.HTTPError as e:
        return False, "", f"Website returned error {e.response.status_code}. Please verify the URL."
    except Exception:
        return False, "", "Could not fetch website. Please check the URL and try again."

    soup = BeautifulSoup(response.content, 'html.parser')

    # Remove non-content elements
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()

    # Try semantic content areas first, fall back to body
    content = soup.find('main') or soup.find('article') or soup.find('body')
    if not content:
        return False, "", "Could not extract text from website."

    text = content.get_text(separator=' ', strip=True)
    # Clean up whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if len(text) < 50:
        return False, "", "Website had very little text content. Try a different page or enter copy manually."

    # Truncate to stay within LLM token limits
    return True, text[:4000], ""

# ============================================================================
# COPY GENERATION (Free HF Inference API)
# ============================================================================

# Copy sections: (id, label, word_min, word_max, description)
COPY_SECTIONS = [
    ('the_concept', 'The Concept', 30, 100, "Concise overview of the restaurant's concept, chef's vision, history, and cultural influences"),
    ('the_cuisine', 'The Cuisine', 30, 50, 'Describe the cuisine, key dishes, cooking styles, ingredients, and any fusion or innovative elements'),
    ('group_dining', 'Group Dining', 30, 100, 'Summarize group dining, private events, or large party details in a straightforward style'),
    ('meta_title', 'Website Title (SEO)', 5, 10, 'On-brand and SEO-friendly website title meta tag'),
    ('meta_description', 'Meta Description (SEO)', 15, 30, 'On-brand and SEO-friendly website description meta tag'),
]

DEFAULT_COPY_INSTRUCTIONS = """Generate content for the following sections on a new restaurant website being built.

Source Website: Website provided as prompt and sometimes with word document with copy and details inside you can use.

Thoroughly analyze the source site's pages (e.g., home, about, menu, private/group dining) to extract and adapt relevant details. First, determine the site's unique tone and style by examining its existing copy - such as word choice, sentence structure, formality, and overall vibe (e.g., it might be elegant and promotional with narrative, descriptive language highlighting innovation, heritage, and sensory experiences, but adapt based on what you observe). Ensure all new content matches this analyzed tone and style without being overly salesy. To preserve authenticity, incorporate as much of the original words and sentences from the source as available, possible, and makes sense - while still ensuring the content flows naturally and meets word count requirements.

Sections to generate:
1. **The Concept**
   Craft a concise overview of the restaurant's overall concept, drawing from the source site's about page or similar. Focus on the chef's vision, history, unique selling points, and cultural influences. If no explicit details exist, create an original description based on the site's analyzed tone and inferred elements (e.g., from imagery, menu themes, or homepage).
2. **The Cuisine**
   Describe the cuisine in a tone and style matching the source site's analyzed voice (e.g., evocative and refined if that's what you observe), emphasizing key dishes, cooking styles, ingredients, and any fusion or innovative elements based on menu or description pages. If no explicit details exist, create an original description based on the site's tone and inferred elements (e.g., from menu items or photos), erring on the shorter word count of around 30-50 words.
3. **Group Dining**
   Summarize any available details on group dining, private events, or large parties in a straightforward, matter-of-fact style from the source site's relevant page (e.g., private dining or events). If no details exist, use a generic placeholder like: "For groups or private events, please contact us directly to discuss customized options and availability."

Also write on brand and SEO friendly Website Title and Description meta tags.

Guidelines:
- Each section must be 30-100 words (except for The Cuisine, which should be 30-50 words if creating original content i.e.: you can't copy cuisine copy section from source).
- Ensure content is original, engaging, and aligned with the source site's analyzed professional voice (e.g., vivid yet refined descriptions if applicable). Incorporate original wording from the source exactly where possible if it makes sense and does not disrupt flow; otherwise, rephrase creatively while staying factual.
- Research the source site thoroughly via browsing tools if needed for accurate, up-to-date details.
- Do not use these dashes in the copy: "-"
- Never speak in first person.
- In Group Dining section you do not need to add contact details or email.
"""

MASTER_INSTRUCTIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'master_copy_instructions.json')

def load_master_instructions():
    """Load master copy instructions from disk, falling back to the hardcoded default."""
    try:
        with open(MASTER_INSTRUCTIONS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('instructions', DEFAULT_COPY_INSTRUCTIONS)
    except (FileNotFoundError, json.JSONDecodeError):
        return DEFAULT_COPY_INSTRUCTIONS

def save_master_instructions(instructions):
    """Save copy instructions as the new master default to disk."""
    with open(MASTER_INSTRUCTIONS_FILE, 'w', encoding='utf-8') as f:
        json.dump({'instructions': instructions}, f, ensure_ascii=False, indent=2)

def generate_copy(website_text, restaurant_name, section=None, instructions=None):
    """Generate marketing copy from website text using HF Inference API."""
    api_token = st.session_state.get('hf_api_token', '')
    if not api_token:
        return False, {}, "HF API token not configured. Add HF_API_TOKEN to .env file."

    if not instructions:
        instructions = DEFAULT_COPY_INSTRUCTIONS

    if section:
        # Regenerate a single section
        section_info = next((s for s in COPY_SECTIONS if s[0] == section), None)
        if not section_info:
            return False, {}, f"Unknown section: {section}"
        sid, label, wmin, wmax, desc = section_info
        prompt = (
            f"You are a professional copywriter for upscale restaurants.\n\n"
            f"INSTRUCTIONS:\n{instructions}\n\n"
            f"Based on this website content for {restaurant_name}:\n{website_text[:3000]}\n\n"
            f"Write a {label} ({desc}). STRICT word limit: {wmin}-{wmax} words.\n"
            f"Return ONLY the copy text, nothing else."
        )
        max_tokens = 300
    else:
        # Generate all sections
        section_list = "\n".join(
            f"{i+1}. [{s[0].upper()}] {s[1]} (STRICT: {s[2]}-{s[3]} words) - {s[4]}"
            for i, s in enumerate(COPY_SECTIONS)
        )
        prompt = (
            f"You are a professional copywriter for upscale restaurants.\n\n"
            f"INSTRUCTIONS:\n{instructions}\n\n"
            f"Based on this website content for {restaurant_name}:\n{website_text[:3000]}\n\n"
            f"Generate marketing copy for these {len(COPY_SECTIONS)} sections:\n{section_list}\n\n"
            f"Format your response EXACTLY as:\n"
            f"[THE_CONCEPT]\nyour text\n[/THE_CONCEPT]\n\n"
            f"[THE_CUISINE]\nyour text\n[/THE_CUISINE]\n\n"
            f"[GROUP_DINING]\nyour text\n[/GROUP_DINING]\n\n"
            f"[META_TITLE]\nyour text\n[/META_TITLE]\n\n"
            f"[META_DESCRIPTION]\nyour text\n[/META_DESCRIPTION]\n\n"
            f"IMPORTANT: You MUST stay within the word limits for each section. Count your words carefully."
        )
        max_tokens = 1500

    try:
        client = InferenceClient(token=api_token)
        result = client.chat_completion(
            model="Qwen/Qwen2.5-7B-Instruct",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        response_text = result.choices[0].message.content.strip()
    except Exception as e:
        error_str = str(e).lower()
        if '401' in error_str or 'unauthorized' in error_str:
            return False, {}, "Invalid HF token in .env file."
        elif '403' in error_str or 'permission' in error_str:
            return False, {}, "HF token needs 'Inference Providers' permission."
        elif '503' in error_str or 'loading' in error_str:
            return False, {}, "Model is loading, please try again in a few seconds."
        elif '429' in error_str or 'rate' in error_str:
            return False, {}, "Rate limit reached. Please wait a minute and try again."
        else:
            return False, {}, "Copy generation temporarily unavailable."

    if section:
        return True, {section: response_text}, ""

    # Parse sections from tagged response
    copy_dict = {}
    tag_map = {
        'the_concept': 'THE_CONCEPT',
        'the_cuisine': 'THE_CUISINE',
        'group_dining': 'GROUP_DINING',
        'meta_title': 'META_TITLE',
        'meta_description': 'META_DESCRIPTION',
    }
    for key, tag in tag_map.items():
        pattern = rf'\[{tag}\](.*?)\[/{tag}\]'
        match = re.search(pattern, response_text, re.DOTALL | re.IGNORECASE)
        copy_dict[key] = match.group(1).strip() if match else ""

    if not any(copy_dict.values()):
        return False, {}, "Could not parse generated copy. Please try again."

    return True, copy_dict, ""

# ============================================================================
# PAGE SETUP
# ============================================================================

st.set_page_config(page_title="Starr Restaurants", layout="wide")

# ============================================================================
# MASTER CSS — Starr Brand Theme
# ============================================================================
st.markdown("""
<style>
/* === GLOBAL === */
[data-testid="stDecoration"] { display: none !important; }

.block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

img { border-radius: 0 !important; }


/* === BRANDED HEADER === */
.starr-header {
    background: linear-gradient(135deg, #031E41 0%, #0A3366 100%);
    padding: 1.5rem 2rem;
    margin: 3.5rem -1rem 1.5rem -1rem;
    position: relative;
    border-top: 3px solid #C5A258;
    border-bottom: 3px solid #C5A258;
}
.starr-header h1 {
    color: #FFFFFF !important;
    font-size: 1.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px;
    margin: 0 !important;
    padding: 0 !important;
}
.starr-header .starr-subtitle {
    color: #C5A258;
    font-size: 0.9rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-top: 0.25rem;
    font-weight: 400;
}
.starr-header .made-tooled {
    position: absolute;
    bottom: 0.5rem;
    right: 1rem;
    color: #FFFFFF;
    font-size: 0.8rem;
    letter-spacing: 0.5px;
    font-weight: 400;
}


/* === TAB NAVIGATION === */
[data-testid="stTabs"] button[data-baseweb="tab"] {
    font-size: 0.95rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    padding: 0.75rem 1.5rem !important;
    color: #5A5A6E !important;
    border-bottom: 3px solid transparent !important;
    transition: color 0.2s, border-color 0.2s;
}
[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
    color: #031E41 !important;
    font-weight: 600 !important;
    border-bottom: 3px solid #C5A258 !important;
}
[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
    color: #031E41 !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    display: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab-border"] {
    background-color: #E8E5DE !important;
}


/* === BUTTONS === */
[data-testid="stButton"] > button {
    border: 1.5px solid #031E41 !important;
    color: #031E41 !important;
    background-color: transparent !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.3px;
    padding: 0.4rem 1.2rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stButton"] > button:hover {
    background-color: #031E41 !important;
    color: #FFFFFF !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background-color: #031E41 !important;
    color: #FFFFFF !important;
    border: 1.5px solid #031E41 !important;
    font-weight: 600 !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #0A3366 !important;
    border-color: #0A3366 !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] > button {
    border: 1.5px solid #031E41 !important;
    color: #031E41 !important;
    background-color: transparent !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    border-radius: 4px !important;
    transition: all 0.2s ease !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #031E41 !important;
    color: #FFFFFF !important;
}


/* === TEXT INPUTS & TEXT AREAS === */
[data-testid="stTextInput"] input,
[data-testid="stTextArea"] textarea {
    border: 1px solid #E8E5DE !important;
    border-radius: 4px !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stTextInput"] input:focus,
[data-testid="stTextArea"] textarea:focus {
    border-color: #031E41 !important;
    box-shadow: 0 0 0 1px #031E41 !important;
}


/* === FILE UPLOADER === */
[data-testid="stFileUploader"] {
    border: 2px dashed #D4D2CC !important;
    border-radius: 6px !important;
    padding: 1rem !important;
    background-color: #FAFAF7 !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #C5A258 !important;
}
[data-testid="stFileUploader"] button {
    background-color: #031E41 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 4px !important;
}


/* === SECTION HEADINGS === */
[data-testid="stHeading"] h2 {
    color: #031E41 !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    letter-spacing: 0.3px;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid #C5A258;
    margin-bottom: 1rem !important;
}
[data-testid="stHeading"] h3 {
    color: #031E41 !important;
    font-weight: 600 !important;
    font-size: 1.15rem !important;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    padding-bottom: 0.35rem;
    border-bottom: 1px solid #C5A258;
    margin-bottom: 2rem !important;
}


/* === DIVIDERS === */
hr {
    border: none !important;
    border-top: 1px solid #E8E5DE !important;
    margin: 1.5rem 0 !important;
}


/* === ALERTS === */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-size: 0.88rem !important;
    padding: 0.75rem 1rem !important;
}


/* === EXPANDER === */
[data-testid="stExpander"] {
    border: 1px solid #E8E5DE !important;
    border-radius: 6px !important;
    background-color: #FAFAF7 !important;
}
[data-testid="stExpander"] summary {
    font-weight: 500 !important;
    color: #031E41 !important;
}


/* === METRIC === */
[data-testid="stMetric"] {
    background-color: #F2F0EB;
    padding: 0.5rem 0.75rem;
    border-radius: 6px;
    border-left: 3px solid #C5A258;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #031E41 !important;
    font-weight: 600 !important;
}


/* === SLIDER === */
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: #031E41 !important;
    border-color: #031E41 !important;
}


/* === CUSTOM: IMAGE FIELD CARD === */
.image-field-card {
    background: #FFFFFF;
    border: 1px solid #E8E5DE;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 3px rgba(3, 30, 65, 0.06);
}
.image-field-card .field-title {
    color: #031E41;
    font-weight: 600;
    font-size: 1rem;
    margin-bottom: 0.15rem;
}
.image-field-card .field-requirement {
    color: #5A5A6E;
    font-size: 0.82rem;
}


/* === RESTAURANT LIST BUTTONS (lighter/subtler, half-width, left-aligned) === */
button.rest-btn-light {
    border-color: #A0B4C8 !important;
    color: #A0B4C8 !important;
    font-weight: 400 !important;
    font-size: 0.8rem !important;
    width: auto !important;
    min-width: 0 !important;
}
button.rest-btn-light:hover {
    background-color: #A0B4C8 !important;
    color: #FFFFFF !important;
}
@media (max-width: 640px) {
    button.rest-btn-light {
        margin-top: 0.25rem !important;
    }
}
.restaurant-separator {
    border: none;
    border-top: 1px solid #E8E5DE;
    margin: -0.35rem 0;
}

/* Center restaurant rows horizontally */
[data-testid="stHorizontalBlock"]:has(.restaurant-row) {
    max-width: 750px;
    margin-left: auto;
    margin-right: auto;
}

/* === CUSTOM: RESTAURANT ROW === */
.restaurant-row {
    background: #FFFFFF;
    border: 1px solid #E8E5DE;
    border-radius: 6px;
    padding: 0.75rem 1rem;
    margin: 0;
    transition: border-color 0.2s, box-shadow 0.2s;
    cursor: pointer;
}
.restaurant-row:hover {
    border-color: #C5A258;
    box-shadow: 0 1px 4px rgba(3, 30, 65, 0.08);
    background: #F8F6F1;
}
.restaurant-row.active {
    border-left: 4px solid #C5A258;
    background: #F8F6F1;
}
.restaurant-row .rest-name {
    font-weight: 600;
    color: #031E41;
    font-size: 0.95rem;
}
.restaurant-row .rest-stats {
    color: #5A5A6E;
    font-size: 0.82rem;
    margin-top: 0.25rem;
}
.restaurant-row .rest-url {
    color: #7A7A8E;
    font-size: 0.78rem;
    font-style: italic;
}


/* === CUSTOM: PROGRESS PILLS === */
.progress-pill {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    font-size: 0.75rem;
    font-weight: 500;
    margin-right: 0.4rem;
}
.progress-pill.images { background: #e8eef5; color: #0A3366; }
.progress-pill.alt    { background: #e8f5ec; color: #2D7D46; }
.progress-pill.copy   { background: #fdf3e0; color: #B8860B; }


/* === CUSTOM: COPY SECTION CARD === */
.copy-section-card {
    background: #FFFFFF;
    border: 1px solid #E8E5DE;
    border-radius: 6px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 3px rgba(3, 30, 65, 0.06);
}
.copy-section-card .section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.copy-section-card .section-label {
    color: #031E41;
    font-weight: 600;
    font-size: 1rem;
}
.copy-section-card .section-desc {
    color: #5A5A6E;
    font-size: 0.82rem;
    margin-top: 0.1rem;
}


/* === CUSTOM: WORD COUNT BADGES === */
.word-count-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 600;
    white-space: nowrap;
}
.word-count-badge.in-range { background: #e8f5ec; color: #2D7D46; }
.word-count-badge.warn     { background: #fdf3e0; color: #B8860B; }
.word-count-badge.over     { background: #fce8e8; color: #A63D40; }
.word-count-badge.empty    { background: #F2F0EB; color: #7A7A8E; }


/* === CUSTOM: INLINE LABEL === */
.field-label {
    color: #031E41;
    font-weight: 600;
    font-size: 0.95rem;
    margin: 0.5rem 0;
}


/* === FOOTER === */
footer { visibility: hidden; }
footer:after {
    content: 'Starr Restaurant Group \u2014 CMS Content Manager';
    visibility: visible;
    display: block;
    position: relative;
    padding: 5px;
    top: 2px;
    color: #7A7A8E;
    font-size: 0.75rem;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)

# Branded header
st.markdown("""
<div class="starr-header">
    <h1>Starr Restaurants</h1>
    <div class="starr-subtitle">Restaurant Website Content Tool</div>
    <div class="made-tooled">Made{<em>Tooled</em>}</div>
</div>
""", unsafe_allow_html=True)

# Initialize session state keys
if 'hf_api_token' not in st.session_state:
    st.session_state['hf_api_token'] = os.getenv('HF_API_TOKEN', '')

# Load persisted data from SQLite on first run of this session
if 'db_loaded' not in st.session_state:
    st.session_state['db_loaded'] = True
    with st.spinner("Loading restaurant data..."):
        saved_restaurants = db.get_all_restaurants()
        st.session_state['restaurants_list'] = [r['name'] for r in saved_restaurants]

        # Restore URLs, copy, alt text, and overlay settings per restaurant
        for r in saved_restaurants:
            rname = r['name']
            if r['website_url']:
                st.session_state[f"{rname}_website_url"] = r['website_url']

            # Restore copy sections
            copy_data = db.get_copy_for_restaurant(rname)
            for sec_id, content in copy_data.items():
                st.session_state[f"{rname}_copy_{sec_id}"] = content

            # Restore image metadata (alt text, overlay)
            img_data = db.get_images_for_restaurant(rname)
            for field_name, info in img_data.items():
                if info['alt_text']:
                    st.session_state[f"{rname}_{field_name}_alt"] = info['alt_text']
                if field_name in ('Hero_Image_Desktop', 'Hero_Image_Mobile'):
                    st.session_state[f"{rname}_{field_name}_opacity"] = info['overlay_opacity']
                # Mark that a persisted image exists in the database
                if info.get('has_image'):
                    st.session_state[f"{rname}_{field_name}_persisted"] = True

        # Select first restaurant if none selected
        if st.session_state['restaurants_list']:
            st.session_state.setdefault('restaurant_name_cleaned', st.session_state['restaurants_list'][0])
        else:
            st.session_state.setdefault('restaurant_name_cleaned', None)

if 'restaurants_list' not in st.session_state:
    st.session_state['restaurants_list'] = []
if 'restaurant_name_cleaned' not in st.session_state:
    st.session_state['restaurant_name_cleaned'] = None

# Image mappings: (name) -> (target_width, target_height, aspect_ratio)
image_mappings = {
    'Hero_Image_Desktop': (1920, 1080, 16/9),
    'Hero_Image_Mobile': (750, 472, 1.588),
    'Concept_1': (600, 800, 3/4),
    'Concept_2': (600, 600, 1/1),
    'Concept_3': (600, 600, 1/1),
    'Cuisine_1': (600, 800, 3/4),
    'Cuisine_2': (600, 600, 1/1),
    'Menu_1': (1920, 1080, 16/9),
    'Chef_1': (600, 800, 3/4),
    'Chef_2': (600, 800, 3/4),
    'Chef_3': (600, 800, 3/4),
}

fields = [
    ('Hero_Image_Desktop', "Main Desktop Banner Image (Horizontal)", "Image Requirement: Horizontal image with <u>estimated</u> aspect ratio of 16:9."),
    ('Hero_Image_Mobile', "Main Mobile Banner Image (Horizontal)", "Image Requirement: Horizontal image with <u>estimated</u> aspect ratio of 1.588:1."),
    ('Concept_1', "First Concept Image (Vertical)", "Image Requirement: Vertical image with <u>estimated</u> aspect ratio of 3:4."),
    ('Concept_2', "Second Concept Image (Square)", "Image Requirement: Square image with <u>estimated</u> aspect ratio of 1:1."),
    ('Concept_3', "Third Concept Image (Square)", "Image Requirement: Square image with <u>estimated</u> aspect ratio of 1:1."),
    ('Cuisine_1', "First Cuisine Image (Vertical)", "Image Requirement: Vertical image with <u>estimated</u> aspect ratio of 3:4."),
    ('Cuisine_2', "Second Cuisine Image (Square)", "Image Requirement: Square image with <u>estimated</u> aspect ratio of 1:1."),
    ('Menu_1', "Menu Image (Horizontal)", "Image Requirement: Horizontal image with <u>estimated</u> aspect ratio of 16:9."),
    ('Chef_1', "First Chef Image (Vertical + Black&White) (Optional)", "Image Requirement: Vertical image with <u>estimated</u> aspect ratio of 3:4."),
    ('Chef_2', "Second Chef Image (Vertical + Black&White) (Optional)", "Image Requirement: Vertical image with <u>estimated</u> aspect ratio of 3:4."),
    ('Chef_3', "Third Chef Image (Vertical + Black&White) (Optional)", "Image Requirement: Vertical image with <u>estimated</u> aspect ratio of 3:4.")
]

# ============================================================================
# MAIN UI - TABBED LAYOUT
# ============================================================================

tab_restaurants, tab_images, tab_copy = st.tabs(["Restaurants", "Images", "Copy & Metadata"])

# ==============================================================================
# TAB 1: RESTAURANTS
# ==============================================================================
with tab_restaurants:
    st.header("Manage Restaurants")

    col_name, col_url, col_add = st.columns([2, 2, 1])
    with col_name:
        restaurant_input = st.text_input("Restaurant name:", placeholder="e.g., The Capital Grille")
    with col_url:
        restaurant_url_input = st.text_input("Website URL:", placeholder="https://www.restaurant.com")
    with col_add:
        st.markdown("<div style='margin-bottom:1px'>&nbsp;</div>", unsafe_allow_html=True)
        if st.button("Add"):
            if restaurant_input.strip():
                cleaned_name = re.sub(r'[^a-zA-Z0-9_]', '_', restaurant_input.strip())
                st.session_state['restaurant_name_cleaned'] = cleaned_name
                if cleaned_name not in st.session_state['restaurants_list']:
                    st.session_state['restaurants_list'].append(cleaned_name)
                # Save URL for this restaurant
                url_val = restaurant_url_input.strip()
                if url_val:
                    st.session_state[f"{cleaned_name}_website_url"] = url_val
                # Persist to database
                db.add_restaurant(cleaned_name, restaurant_input.strip(), url_val)
                st.success(f"Restaurant '{restaurant_input}' added as: {cleaned_name}")
                st.rerun()

    st.subheader("Restaurants Content Progress")

    if st.session_state['restaurants_list']:
        rest_list = st.session_state['restaurants_list']
        for rest_idx, rest_name in enumerate(rest_list):
            col1, col2 = st.columns([3, 1])
            with col1:
                # Count uploaded images
                image_count = 0
                for field_name, _, _ in fields:
                    uploader_key = f"{rest_name}_{field_name}"
                    if uploader_key in st.session_state:
                        val = st.session_state.get(uploader_key)
                        if val is not None:
                            image_count += 1

                # Count completed alt texts
                alt_count = 0
                for field_name, _, _ in fields:
                    alt_key = f"{rest_name}_{field_name}_alt"
                    if alt_key in st.session_state and st.session_state[alt_key].strip():
                        alt_count += 1

                # Count completed copy sections
                copy_count = 0
                for sid, _, _, _, _ in COPY_SECTIONS:
                    if st.session_state.get(f"{rest_name}_copy_{sid}", "").strip():
                        copy_count += 1

                is_active = rest_name == st.session_state.get('restaurant_name_cleaned')
                active_class = " active" if is_active else ""
                rest_url = st.session_state.get(f"{rest_name}_website_url", "")
                url_html = f'<div class="rest-url">{rest_url}</div>' if rest_url else ''
                star = '&#9733; ' if is_active else ''
                display_name = rest_name.replace('_', ' ')
                st.markdown(f"""
                <div class="restaurant-row{active_class}" data-name="{rest_name}">
                    <div class="rest-name">{star}{display_name}</div>
                    {url_html}
                    <div class="rest-stats">
                        <span class="progress-pill images">Images: {image_count}/11</span>
                        <span class="progress-pill alt">Alt Text: {alt_count}</span>
                        <span class="progress-pill copy">Copy: {copy_count}/{len(COPY_SECTIONS)}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                if st.button("Select", key=f"select_{rest_name}"):
                    st.session_state['restaurant_name_cleaned'] = rest_name
                    st.rerun()
                if st.button("Delete", key=f"delete_{rest_name}"):
                    db.delete_restaurant(rest_name)
                    st.session_state['restaurants_list'].remove(rest_name)
                    if st.session_state.get('restaurant_name_cleaned') == rest_name:
                        st.session_state['restaurant_name_cleaned'] = (
                            st.session_state['restaurants_list'][0]
                            if st.session_state['restaurants_list'] else None
                        )
                    st.rerun()

            # Divider between restaurant rows
            if rest_idx < len(rest_list) - 1:
                st.markdown('<hr class="restaurant-separator">', unsafe_allow_html=True)

        # Wire up card clicks, lighten Select/Delete buttons
        components.html("""
        <script>
        setTimeout(() => {
            const doc = window.parent.document;
            const rows = doc.querySelectorAll('.restaurant-row[data-name]');

            // Wire up card clicks to trigger the corresponding Select button
            rows.forEach((row) => {
                row.addEventListener('click', function() {
                    const name = this.getAttribute('data-name');
                    const buttons = doc.querySelectorAll('button');
                    for (const btn of buttons) {
                        if (btn.textContent.trim() === 'Select' &&
                            btn.closest('[data-testid]') &&
                            btn.id && btn.id.includes(name)) {
                            btn.click();
                            return;
                        }
                    }
                    // Fallback: match by index
                    const allRows = Array.from(doc.querySelectorAll('.restaurant-row[data-name]'));
                    const idx = allRows.indexOf(this);
                    const selectBtns = Array.from(buttons).filter(b => b.textContent.trim() === 'Select');
                    if (selectBtns[idx]) selectBtns[idx].click();
                });
            });

            // Lighten Select/Delete buttons
            doc.querySelectorAll('button').forEach((btn) => {
                const txt = btn.textContent.trim();
                if (txt === 'Select' || txt === 'Delete') {
                    btn.classList.add('rest-btn-light');
                }
            });

        }, 500);
        </script>
        """, height=0)
    else:
        st.info("No restaurants added yet. Create one above!")

# ==============================================================================
# TAB 2: IMAGES
# ==============================================================================
with tab_images:
    restaurant_name = st.session_state.get('restaurant_name_cleaned')
    if not restaurant_name:
        st.warning("Please select or create a restaurant in the 'Restaurants' tab first.")
    else:
        st.header(f"Upload Images for {restaurant_name.replace('_', ' ')}")

        uploaded_files = {}
        for i, (name, header, description) in enumerate(fields):
            with st.container():
                st.markdown(f"""
                <div class="image-field-card">
                    <div class="field-title">{header}</div>
                    <div class="field-requirement">{description}</div>
                </div>
                """, unsafe_allow_html=True)

                uploader_key = f"{restaurant_name}_{name}"
                uploaded_file = st.file_uploader(
                    "Upload image",
                    type=['jpg', 'jpeg', 'png'],
                    key=uploader_key,
                    label_visibility="collapsed"
                )
                uploaded_files[name] = uploaded_file

                # Determine image source: fresh upload or persisted in database
                persisted_flag_key = f"{restaurant_name}_{name}_persisted"
                has_persisted = bool(st.session_state.get(persisted_flag_key, False))

                resized_img = None
                img_format = 'JPEG'
                ext = 'jpg'
                target_width, target_height, target_ratio = image_mappings[name]
                new_filename = f"{restaurant_name}_{name}_{target_width}x{target_height}.jpg"
                is_fresh_upload = False

                if uploaded_file:
                    is_fresh_upload = True
                    img = Image.open(uploaded_file)

                    # Handle EXIF orientation
                    try:
                        exif = img._getexif()
                        if exif and 274 in exif:
                            orientation = exif[274]
                            if orientation == 3:
                                img = img.rotate(180, expand=True)
                            elif orientation == 6:
                                img = img.rotate(-90, expand=True)
                            elif orientation == 8:
                                img = img.rotate(90, expand=True)
                    except:
                        pass

                    width, height = img.size
                    original_ratio = width / height

                    if name == 'Concept_3':
                        target_ratio = 1.0

                    allowed_deviation = target_ratio * 0.3
                    aspect_ok = abs(original_ratio - target_ratio) <= allowed_deviation

                    is_chef = name in ['Chef_1', 'Chef_2', 'Chef_3']
                    bw_ok = True
                    if is_chef:
                        with st.spinner('Checking if image is grayscale...'):
                            bw_ok = is_black_and_white(img)

                    if not aspect_ok:
                        st.warning("Oops Funky Ingredients: The aspect ratio deviates by more than 30% from the target. Processing may crop substantially.")
                    if is_chef and not bw_ok:
                        st.warning("Brand guidelines suggest Black&White images of the chefs to keep with the editorial look.")
                    if aspect_ok and (not is_chef or bw_ok):
                        st.success("Perfect, looks delicious!")

                    resized_img = resize_and_crop(img, target_width, target_height)

                    ext = uploaded_file.name.split('.')[-1].lower()
                    format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
                    img_format = format_map.get(ext, 'JPEG')
                    new_filename = f"{restaurant_name}_{name}_{target_width}x{target_height}.{ext}"

                elif has_persisted:
                    # Load previously saved image from database blob
                    blob_data = db.get_image_data(restaurant_name, name)
                    if blob_data:
                        resized_img = Image.open(io.BytesIO(blob_data))
                        record = db.get_image_record(restaurant_name, name)
                        orig_fn = record.get('original_filename', '') if record else ''
                        ext = orig_fn.rsplit('.', 1)[-1].lower() if '.' in orig_fn else 'jpg'
                        format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
                        img_format = format_map.get(ext, 'JPEG')
                        new_filename = f"{restaurant_name}_{name}_{target_width}x{target_height}.{ext}"
                        st.caption("Previously saved image loaded from storage.")

                if resized_img:
                    st.markdown('<div class="field-label">Preview</div>', unsafe_allow_html=True)

                    # Hero images: show overlay slider
                    if name in ['Hero_Image_Desktop', 'Hero_Image_Mobile']:
                        opacity_key = f"{restaurant_name}_{name}_opacity"
                        if opacity_key not in st.session_state:
                            st.session_state[opacity_key] = 40

                        st.markdown('<div class="field-label">Filter Opacity</div>', unsafe_allow_html=True)
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.session_state[opacity_key] = st.slider(
                                "Opacity",
                                min_value=0,
                                max_value=100,
                                value=st.session_state[opacity_key],
                                step=5,
                                key=f"slider_{name}",
                                label_visibility="collapsed"
                            )
                        with col2:
                            st.metric("Opacity", f"{st.session_state[opacity_key]}%")

                        overlay_value = st.session_state[opacity_key]
                        resized_img_no_overlay = resized_img
                        if overlay_value > 0:
                            resized_img_with_overlay = apply_black_overlay(resized_img_no_overlay, overlay_value)
                        else:
                            resized_img_with_overlay = resized_img_no_overlay

                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.caption("Without Filter")
                            st.image(resized_img_no_overlay, width=300)
                        with col_after:
                            st.caption(f"With Filter ({overlay_value}% opacity)")
                            st.image(resized_img_with_overlay, width=300)

                        final_opacity = st.session_state[opacity_key]
                        if final_opacity > 0:
                            resized_img = apply_black_overlay(resized_img_no_overlay, final_opacity)
                        else:
                            resized_img = resized_img_no_overlay

                        # Persist overlay setting
                        db.update_overlay(restaurant_name, name, final_opacity)
                    else:
                        st.image(resized_img, width=300)

                    # Save processed image to disk on fresh upload
                    img_buffer = io.BytesIO()
                    resized_img.save(img_buffer, format=img_format, quality=95)
                    img_buffer.seek(0)
                    img_bytes = img_buffer.getvalue()

                    if is_fresh_upload:
                        with st.spinner("Saving image..."):
                            db.save_image(
                                restaurant_name, name, img_bytes,
                                uploaded_file.name,
                                alt_text=st.session_state.get(f"{restaurant_name}_{name}_alt", ''),
                                overlay_opacity=st.session_state.get(f"{restaurant_name}_{name}_opacity", 40)
                            )
                        st.session_state[persisted_flag_key] = True

                    # Download button for individual image
                    st.download_button(
                        label=f"Download Resized {name}",
                        data=img_bytes,
                        file_name=new_filename,
                        mime=f"image/{ext}",
                        key=f"download_{name}"
                    )

                    # Alt text inline
                    alt_key = f"{restaurant_name}_{name}_alt"
                    if alt_key not in st.session_state:
                        st.session_state[alt_key] = ""

                    # Auto-generate on first upload if API key is set and alt text is empty
                    auto_key = f"{restaurant_name}_{name}_auto_generated"
                    if is_fresh_upload and st.session_state.get('hf_api_token') and not st.session_state[alt_key] and not st.session_state.get(auto_key):
                        with st.spinner("Generating alt text..."):
                            alt_text = generate_alt_text(resized_img)
                        if alt_text:
                            st.session_state[alt_key] = alt_text
                            st.session_state[auto_key] = True
                            db.update_alt_text(restaurant_name, name, alt_text)

                    st.markdown('<div class="field-label">Alt Text (ADA)</div>', unsafe_allow_html=True)
                    new_alt = st.text_area(
                        f"Alt text for {header}",
                        value=st.session_state[alt_key],
                        key=alt_key,
                        label_visibility="collapsed",
                        height=68
                    )

                    # Persist alt text if changed
                    if new_alt != st.session_state.get(f"{restaurant_name}_{name}_alt_prev", ''):
                        st.session_state[f"{restaurant_name}_{name}_alt_prev"] = new_alt
                        if has_persisted or is_fresh_upload:
                            db.update_alt_text(restaurant_name, name, new_alt)

                    if st.session_state[alt_key].strip():
                        copy_button(st.session_state[alt_key], f"copy_alt_{name}")

            if i < len(fields) - 1:
                st.markdown("---")
        
        # Batch download
        if any(uploaded_files.values()):
            st.markdown("---")
            if st.button("Download All Resized Images"):
                with st.spinner("Preparing ZIP file..."):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for name, file in uploaded_files.items():
                            if file:
                                target_width, target_height, _ = image_mappings[name]
                                img = Image.open(file)

                                try:
                                    exif = img._getexif()
                                    if exif and 274 in exif:
                                        orientation = exif[274]
                                        if orientation == 3:
                                            img = img.rotate(180, expand=True)
                                        elif orientation == 6:
                                            img = img.rotate(-90, expand=True)
                                        elif orientation == 8:
                                            img = img.rotate(90, expand=True)
                                except:
                                    pass

                                resized_img = resize_and_crop(img, target_width, target_height)

                                if name in ['Hero_Image_Desktop', 'Hero_Image_Mobile']:
                                    opacity_key = f"{restaurant_name}_{name}_opacity"
                                    overlay_value = st.session_state.get(opacity_key, 40)
                                    if overlay_value > 0:
                                        resized_img = apply_black_overlay(resized_img, overlay_value)

                                ext = file.name.split('.')[-1].lower()
                                format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
                                img_format = format_map.get(ext, 'JPEG')
                                new_filename = f"{restaurant_name}_{name}_{target_width}x{target_height}.{ext}"

                                img_buffer = io.BytesIO()
                                resized_img.save(img_buffer, format=img_format, quality=95)
                                img_buffer.seek(0)
                                zip_file.writestr(f"Resized/{new_filename}", img_buffer.read())
                
                zip_buffer.seek(0)
                st.download_button(
                    label="Download ZIP of All Resized Images",
                    data=zip_buffer,
                    file_name=f"{restaurant_name}_resized_images.zip",
                    mime="application/zip",
                    key="download_all"
                )

# ==============================================================================
# TAB 3: COPY & METADATA
# ==============================================================================
with tab_copy:
    restaurant_name = st.session_state.get('restaurant_name_cleaned')
    if not restaurant_name:
        st.header("Copy & Metadata")
        st.warning("Please select or create a restaurant in the 'Restaurants' tab first.")
    else:
        st.header(f"Copy & Metadata for {restaurant_name.replace('_', ' ')}")
        url_key = f"{restaurant_name}_website_url"
        stored_url = st.session_state.get(url_key, "")

        # --- URL + Generate ---
        if stored_url:
            st.markdown(f'<div class="field-label">Source URL: <span style="font-weight:400;color:#5A5A6E">{stored_url}</span></div>', unsafe_allow_html=True)
        else:
            st.info("Add a website URL for this restaurant in the Restaurants tab to enable copy generation.")

        col_gen, col_url_edit = st.columns([1, 3])
        with col_gen:
            generate_all = st.button("Generate Copy", type="primary", disabled=not stored_url)
        with col_url_edit:
            new_url = st.text_input(
                "Update URL",
                value=stored_url,
                placeholder="https://www.restaurant.com",
                key=f"{url_key}_copy_input",
                label_visibility="collapsed"
            )
            if new_url != stored_url:
                st.session_state[url_key] = new_url
                stored_url = new_url
                db.update_restaurant_url(restaurant_name, new_url)

        # --- Edit Copy Instructions ---
        if 'copy_instructions' not in st.session_state:
            st.session_state['copy_instructions'] = load_master_instructions()

        with st.expander("Edit Copy Instructions"):
            st.markdown(
                "These instructions are sent to the AI along with the scraped website content. "
                "They control the **tone, style, word counts, and formatting rules** the AI follows "
                "when generating copy. Edit them to change how the AI writes for your restaurants."
            )
            st.session_state['copy_instructions'] = st.text_area(
                "Copy Instructions",
                value=st.session_state['copy_instructions'],
                height=300,
                key="copy_instructions_editor",
                label_visibility="collapsed"
            )

            col_reset, col_save = st.columns(2)
            with col_reset:
                if st.button("Reset to Default"):
                    st.session_state['copy_instructions'] = DEFAULT_COPY_INSTRUCTIONS
                    st.rerun()
            with col_save:
                if st.button("Save As Master"):
                    save_master_instructions(st.session_state['copy_instructions'])
                    st.success("Saved as new master instructions.")
                    st.rerun()

        if generate_all:
            if not stored_url.strip():
                st.error("Please add a website URL in the Restaurants tab first.")
            elif not st.session_state.get('hf_api_token'):
                st.error("HF API token not configured. Add HF_API_TOKEN to .env file.")
            else:
                with st.spinner("Scraping website content..."):
                    ok, content, err = scrape_website(stored_url)
                if not ok:
                    st.error(err)
                else:
                    with st.spinner("Generating marketing copy with AI - this may take 30-60 seconds..."):
                        ok, copy_dict, err = generate_copy(content, restaurant_name, instructions=st.session_state.get('copy_instructions'))
                    if not ok:
                        st.error(err)
                    else:
                        for sec_key, sec_val in copy_dict.items():
                            st.session_state[f"{restaurant_name}_copy_{sec_key}"] = sec_val
                        # Persist all generated copy to database
                        db.save_all_copy(restaurant_name, copy_dict)
                        st.success("Copy generated!")
                        st.rerun()

        st.markdown("---")

        # === WEBSITE COPY SECTIONS ===
        st.subheader("Website Copy")

        # Sections that are website copy (not meta)
        copy_section_ids = ['the_concept', 'the_cuisine', 'group_dining']
        meta_section_ids = ['meta_title', 'meta_description']

        for section_id, section_label, word_min, word_max, description in COPY_SECTIONS:
            if section_id not in copy_section_ids:
                continue

            section_key = f"{restaurant_name}_copy_{section_id}"
            if section_key not in st.session_state:
                st.session_state[section_key] = ""

            text = st.session_state[section_key]
            word_count = len(text.split()) if text.strip() else 0
            if word_count == 0:
                badge_class = "empty"
            elif word_min <= word_count <= word_max:
                badge_class = "in-range"
            elif word_count < word_min or word_count <= word_max * 1.2:
                badge_class = "warn"
            else:
                badge_class = "over"

            st.markdown(f"""
            <div class="copy-section-card">
                <div class="section-header">
                    <div>
                        <div class="section-label">{section_label}</div>
                        <div class="section-desc">{description}</div>
                    </div>
                    <span class="word-count-badge {badge_class}">{word_count} / {word_min}-{word_max} words</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.text_area(
                f"Edit {section_label}",
                value=st.session_state[section_key],
                key=section_key,
                height=120,
                placeholder="No content generated yet. Click 'Generate Copy' above to create content.",
                label_visibility="collapsed"
            )

            # Persist copy text when edited
            db.save_copy_section(restaurant_name, section_id, st.session_state[section_key])

            if st.session_state[section_key].strip():
                copy_button(st.session_state[section_key], f"copy_{section_id}")

            st.markdown("---")

        # === SEO META TAGS ===
        st.markdown("---")
        st.subheader("SEO Meta Tags")
        st.caption("These are the HTML meta title and description tags for search engine optimization.")

        for section_id, section_label, word_min, word_max, description in COPY_SECTIONS:
            if section_id not in meta_section_ids:
                continue

            section_key = f"{restaurant_name}_copy_{section_id}"
            if section_key not in st.session_state:
                st.session_state[section_key] = ""

            text = st.session_state[section_key]
            word_count = len(text.split()) if text.strip() else 0
            if word_count == 0:
                badge_class = "empty"
            elif word_min <= word_count <= word_max:
                badge_class = "in-range"
            elif word_count < word_min or word_count <= word_max * 1.2:
                badge_class = "warn"
            else:
                badge_class = "over"

            st.markdown(f"""
            <div class="copy-section-card">
                <div class="section-header">
                    <div>
                        <div class="section-label">{section_label}</div>
                        <div class="section-desc">{description}</div>
                    </div>
                    <span class="word-count-badge {badge_class}">{word_count} / {word_min}-{word_max} words</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            text_height = 68 if section_id == 'meta_title' else 80
            st.text_area(
                f"Edit {section_label}",
                value=st.session_state[section_key],
                key=section_key,
                height=text_height,
                placeholder="No content generated yet. Click 'Generate Copy' above to create content.",
                label_visibility="collapsed"
            )

            # Persist copy text when edited
            db.save_copy_section(restaurant_name, section_id, st.session_state[section_key])

            if st.session_state[section_key].strip():
                copy_button(st.session_state[section_key], f"copy_{section_id}")

            st.markdown("---")
