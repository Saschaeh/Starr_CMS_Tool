import streamlit as st
from PIL import Image
import io
import base64
import re
import zipfile
import os
import json
from collections import Counter
from urllib.parse import urlparse, urljoin
import numpy as np
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

def fix_exif_orientation(img):
    """Apply EXIF orientation rotation if present."""
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
    except Exception:
        pass
    return img

def make_image_filename(restaurant, field_name, width, height, ext, alt_text=''):
    """Generate WordPress-style filename: Restaurant_Field_alt_text_snippet_WxH.ext
    Alt text is trimmed to ~6 words for SEO-friendly filenames."""
    if alt_text and alt_text.strip():
        slug = re.sub(r'[^a-z0-9\s]', '', alt_text.strip().lower())
        words = slug.split()
        words = [w for w in words if w not in ('a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'and', 'with', 'for', 'is', 'by')]
        slug = '_'.join(words[:6])
        return f"{restaurant}_{field_name}_{slug}_{width}x{height}.{ext}"
    return f"{restaurant}_{field_name}_{width}x{height}.{ext}"

def is_black_and_white(img, threshold=0.1):
    """Check if image is predominantly black and white using vectorized NumPy ops."""
    arr = np.array(img.convert('RGB'), dtype=np.int16)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    limit = threshold * 255
    bw_mask = (np.abs(r - g) < limit) & (np.abs(g - b) < limit)
    return bw_mask.mean() > 0.8

def apply_black_overlay(img, opacity_percent):
    """Apply a semi-transparent black overlay to image."""
    img_rgba = img.convert('RGBA')
    overlay = Image.new('RGBA', img_rgba.size, (0, 0, 0, int(255 * opacity_percent / 100)))
    return Image.alpha_composite(img_rgba, overlay).convert('RGB')

def render_copy_section(restaurant_name, section_id, section_label, word_min, word_max, description, height=120):
    """Render a copy section card with word count badge, text area, and copy button."""
    section_key = f"{restaurant_name}_copy_{section_id}"
    if section_key not in st.session_state:
        st.session_state[section_key] = ""

    text = st.session_state[section_key]
    word_count = len(text.split()) if text.strip() else 0
    if word_count == 0:
        badge_class = "empty"
    elif word_min <= word_count <= word_max:
        badge_class = "in-range"
    elif word_count > word_max * 1.2:
        badge_class = "over"
    else:
        badge_class = "warn"

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
        height=height,
        placeholder="No content generated yet. Click 'Generate Copy' above to create content.",
        label_visibility="collapsed"
    )

    if st.session_state[section_key].strip():
        copy_button(st.session_state[section_key], f"copy_{section_id}")

    st.markdown("---")

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

@st.cache_resource
def _get_hf_client(token):
    """Return a cached InferenceClient instance for the given token."""
    return InferenceClient(token=token)

@st.cache_data(show_spinner=False)
def _load_persisted_image(restaurant, field_name):
    """Load image blob and metadata from DB (cached to avoid re-fetching every rerun)."""
    blob_data = db.get_image_data(restaurant, field_name)
    record = db.get_image_record(restaurant, field_name)
    return blob_data, record

def generate_alt_text(pil_image):
    """Generate alt text from image using Hugging Face Inference API (Qwen Vision)."""
    api_token = st.session_state.get('hf_api_token', '')
    if not api_token:
        return None

    # Convert PIL image to base64 (convert to RGB for JPEG compatibility)
    img_buffer = io.BytesIO()
    if pil_image.mode in ('RGBA', 'LA', 'PA', 'P'):
        pil_image = pil_image.convert('RGB')
    pil_image.save(img_buffer, format='JPEG', quality=85)
    img_b64 = base64.b64encode(img_buffer.getvalue()).decode()

    try:
        client = _get_hf_client(api_token)
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

def _fetch_page_text(url, headers):
    """Fetch a single page and return cleaned text, or empty string on failure."""
    try:
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(response.content, 'html.parser')
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    content = soup.find('main') or soup.find('article') or soup.find('body')
    if not content:
        return ""
    text = content.get_text(separator=' ', strip=True)
    return re.sub(r'\s+', ' ', text).strip()


# Subpage path keywords to look for when scraping restaurant sites
_SUBPAGE_KEYWORDS = [
    'about', 'concept', 'story', 'menu', 'cuisine', 'food',
    'group', 'private', 'dining', 'event', 'party', 'parties',
    'reserve', 'reservation', 'chef',
]

# Common restaurant subpage paths to always try (single-page sites often
# don't link to these in standard <a> tags, using anchor links instead)
_COMMON_SUBPATHS = [
    '/about/', '/about-us/', '/group-dining/', '/private-dining/',
    '/private-events/', '/menu/', '/events/', '/the-cuisine/',
    '/the-concept/', '/chef/',
]


def _extract_primary_color(soup, base_url):
    """Extract the likely primary brand color from a webpage.

    Strategy (in priority order):
    1. <meta name="theme-color">
    2. CSS custom properties with 'primary/brand/accent/main' in the name
    3. Most frequent non-neutral hex color across inline + external CSS
    """
    # 1. <meta name="theme-color"> — most explicit signal
    meta_theme = soup.find('meta', attrs={'name': 'theme-color'})
    if meta_theme and meta_theme.get('content'):
        val = meta_theme['content'].strip()
        if re.match(r'^#[0-9a-fA-F]{3,8}$', val):
            return val[:7]  # keep only #RRGGBB

    def _normalize_hex(h):
        """Normalize 3-char hex to 6-char lowercase."""
        if len(h) == 3:
            return (h[0]*2 + h[1]*2 + h[2]*2).lower()
        return h.lower()

    def _is_neutral(h):
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        return max(r, g, b) - min(r, g, b) < 30

    def _extract_colors(css_text):
        hex6 = re.findall(r'#([0-9a-fA-F]{6})\b', css_text)
        hex3 = re.findall(r'#([0-9a-fA-F]{3})\b', css_text)
        all_c = [_normalize_hex(c) for c in hex6 + hex3]
        return [c for c in all_c if not _is_neutral(c)]

    # Gather inline CSS (<style> blocks + style attributes) — site-specific
    inline_parts = []
    for style_tag in soup.find_all('style'):
        if style_tag.string:
            inline_parts.append(style_tag.string)
    for tag in soup.find_all(style=True):
        inline_parts.append(tag['style'])
    inline_css = '\n'.join(inline_parts)

    # 2. CSS custom properties containing primary/brand/accent/main (inline first)
    var_pattern = re.compile(
        r'--[a-zA-Z0-9_-]*(?:primary|brand|accent|main)[a-zA-Z0-9_-]*\s*:\s*(#[0-9a-fA-F]{3,8})\b',
        re.IGNORECASE,
    )
    var_match = var_pattern.search(inline_css)
    if var_match:
        val = var_match.group(1)
        if len(val) == 4:
            val = '#' + val[1]*2 + val[2]*2 + val[3]*2
        return val[:7]

    # 3. Check inline CSS for a dominant color first (avoids shared theme noise)
    inline_colors = _extract_colors(inline_css)
    if inline_colors:
        top = Counter(inline_colors).most_common(1)[0]
        if top[1] >= 3:  # strong enough signal from inline CSS alone
            return '#' + top[0]

    # 4. Fetch external stylesheets and combine with inline
    skip_domains = ('fonts.googleapis.com', 'use.typekit.net', 'cdnjs.cloudflare.com', 'cdn.jsdelivr.net')
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
    external_parts = []
    for link in soup.find_all('link', rel='stylesheet'):
        href = link.get('href', '')
        if not href or any(d in href for d in skip_domains):
            continue
        full_url = urljoin(base_url, href)
        try:
            css_resp = requests.get(full_url, headers=headers, timeout=5, stream=True)
            if css_resp.ok:
                css_text = css_resp.text[:500_000]  # cap at 500KB to avoid memory issues
                external_parts.append(css_text)
        except Exception:
            pass

    external_css = '\n'.join(external_parts)

    # Check external CSS for custom properties too
    var_match = var_pattern.search(external_css)
    if var_match:
        val = var_match.group(1)
        if len(val) == 4:
            val = '#' + val[1]*2 + val[2]*2 + val[3]*2
        return val[:7]

    # Combined: inline colors weighted 3x to prioritise site-specific over theme
    all_colors = inline_colors * 3 + _extract_colors(external_css)
    if all_colors:
        return '#' + Counter(all_colors).most_common(1)[0][0]

    return ""


def _detect_booking_platform(html_bytes):
    """Detect whether a restaurant uses Resy or OpenTable from raw HTML.

    Returns "Resy", "OpenTable", or "".
    """
    html_str = html_bytes.decode('utf-8', errors='ignore').lower()
    if any(m in html_str for m in ('widgets.resy.com', 'resywidget', 'resy.com/cities/')):
        return "Resy"
    if any(m in html_str for m in ('opentable.com/widget', 'opentable.com/r/')):
        return "OpenTable"
    return ""


@st.cache_data(ttl=300, show_spinner=False)
def scrape_website(url):
    """Scrape text content from a restaurant website and key subpages.

    Returns (ok, text, error, primary_color, booking_platform).
    """
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}

    # Fetch the main page
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        return False, "", "Website took too long to respond. Please try again.", "", ""
    except requests.exceptions.ConnectionError:
        return False, "", "Could not connect to website. Please check the URL.", "", ""
    except requests.exceptions.HTTPError as e:
        return False, "", f"Website returned error {e.response.status_code}. Please verify the URL.", "", ""
    except Exception:
        return False, "", "Could not fetch website. Please check the URL and try again.", "", ""

    soup = BeautifulSoup(response.content, 'html.parser')
    parsed_base = urlparse(response.url)  # Use final URL after redirects
    base_url = f"{parsed_base.scheme}://{parsed_base.netloc}"
    primary_color = _extract_primary_color(soup, base_url)
    booking_platform = _detect_booking_platform(response.content)
    base_domain = parsed_base.netloc

    # Discover relevant subpage links on the same domain
    subpage_urls = set()
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        full_url = urljoin(response.url, href)
        parsed = urlparse(full_url)
        if parsed.netloc != base_domain:
            continue
        path_lower = parsed.path.lower().strip('/')
        if path_lower and any(kw in path_lower for kw in _SUBPAGE_KEYWORDS):
            clean_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
            subpage_urls.add(clean_url)

    # Also try common restaurant subpaths by convention (single-page sites
    # often use anchor links in nav, so these pages won't be discovered above)
    for subpath in _COMMON_SUBPATHS:
        subpage_urls.add(base_url + subpath)

    # Extract main page text
    for tag in soup.find_all(['script', 'style', 'nav', 'footer', 'header']):
        tag.decompose()
    content = soup.find('main') or soup.find('article') or soup.find('body')
    main_text = ""
    if content:
        main_text = content.get_text(separator=' ', strip=True)
        main_text = re.sub(r'\s+', ' ', main_text).strip()

    # Scrape subpages and combine (subpages first so factual details
    # like group dining numbers aren't truncated by the token limit)
    subpage_parts = []
    for sub_url in sorted(subpage_urls)[:10]:
        page_text = _fetch_page_text(sub_url, headers)
        if page_text and len(page_text) > 30:
            path_label = urlparse(sub_url).path.strip('/').upper().replace('-', ' ')
            subpage_parts.append(f"[{path_label}]\n{page_text}")

    all_text_parts = subpage_parts
    if main_text:
        all_text_parts.append(f"[HOME PAGE]\n{main_text}")

    combined_text = "\n\n".join(all_text_parts)

    if len(combined_text) < 50:
        return False, "", "Website had very little text content. Try a different page or enter copy manually.", "", ""

    # Truncate to stay within LLM token limits
    return True, combined_text[:8000], "", primary_color, booking_platform

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
   Summarize group dining, private events, or large party details in a straightforward, matter-of-fact style. CRITICAL: If the source text contains specific seating capacities or guest counts, you MUST copy those exact numbers verbatim. Do NOT paraphrase, round, or estimate any numbers. If no group dining details exist in the source, use exactly: "For groups or private events, please contact us directly to discuss customized options and availability."

Also write on brand and SEO friendly Website Title and Description meta tags.

Guidelines:
- Each section must be 30-100 words (except for The Cuisine, which should be 30-50 words if creating original content i.e.: you can't copy cuisine copy section from source).
- Ensure content is original, engaging, and aligned with the source site's analyzed professional voice (e.g., vivid yet refined descriptions if applicable). Incorporate original wording from the source exactly where possible if it makes sense and does not disrupt flow; otherwise, rephrase creatively while staying factual.
- Research the source site thoroughly via browsing tools if needed for accurate, up-to-date details.
- Do not use these dashes in the copy: "-"
- Never speak in first person.
- In Group Dining section you do not need to add contact details or email.
- IMPORTANT: Never invent, approximate, or round any numbers. If the source text states specific figures (seating capacity, guest counts, square footage, etc.), use exactly those numbers. If the source does not provide specific numbers, omit them entirely rather than guessing.
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
            f"Based on this website content for {restaurant_name}:\n{website_text[:6000]}\n\n"
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
            f"Based on this website content for {restaurant_name}:\n{website_text[:6000]}\n\n"
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
        client = _get_hf_client(api_token)
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
    position: relative;
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
.progress-pill.chef   { background: #f0ecf5; color: #6B5B8D; }
.progress-pill.alt    { background: #e8f5ec; color: #2D7D46; }
.progress-pill.copy   { background: #fdf3e0; color: #B8860B; }
.booking-label {
    position: absolute;
    top: 1.85rem;
    right: 0.5rem;
    font-size: 0.68rem;
    color: #da3743;
    font-weight: 500;
}
.progress-pill.color  {
    cursor: pointer;
    border: 1px solid #ddd;
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
}
.progress-pill.color:hover { opacity: 0.85; }
.progress-pill.color .swatch {
    display: inline-block;
    width: 10px; height: 10px;
    border-radius: 50%;
    margin-right: 4px;
    vertical-align: middle;
    border: 1px solid rgba(0,0,0,0.15);
}
.color-copied-toast {
    position: fixed; bottom: 2rem; left: 50%;
    transform: translateX(-50%);
    background: #333; color: #fff;
    padding: 0.4rem 1rem; border-radius: 6px;
    font-size: 0.8rem; z-index: 9999;
    animation: fadeout 1.5s ease-in-out forwards;
}
@keyframes fadeout { 0%,60% { opacity:1; } 100% { opacity:0; } }


/* === CUSTOM: RESTAURANT CHECKLIST === */
[data-testid="stHorizontalBlock"]:has(.restaurant-row) ~ [data-testid="stVerticalBlock"] .stCheckbox label,
[data-testid="stHorizontalBlock"] .stCheckbox label {
    font-size: 0.7rem !important;
}
[data-testid="stHorizontalBlock"] .stCheckbox label span[data-testid="stCheckboxLabel"] {
    font-size: 0.7rem !important;
}
[data-testid="stHorizontalBlock"] .stCheckbox {
    transform: scale(0.8);
    transform-origin: left center;
    margin-bottom: -0.4rem;
}


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
            if r.get('notes'):
                st.session_state[f"{rname}_notes"] = r['notes']
            if r.get('primary_color'):
                st.session_state[f"{rname}_primary_color"] = r['primary_color']
            if r.get('booking_platform'):
                st.session_state[f"{rname}_booking_platform"] = r['booking_platform']
            if r.get('checklist'):
                try:
                    cl = json.loads(r['checklist'])
                    for ck, cv in cl.items():
                        st.session_state[f"{rname}_check_{ck}"] = cv
                except Exception:
                    pass

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

# Image mappings: (name) -> (target_width, target_height)
image_mappings = {
    'Hero_Image_Desktop': (1920, 1080),
    'Hero_Image_Mobile': (750, 472),
    'Concept_1': (696, 825),
    'Concept_2': (525, 544),
    'Concept_3': (696, 693),
    'Cuisine_1': (529, 767),
    'Cuisine_2': (696, 606),
    'Menu_1': (1321, 558),
    'Chef_1': (600, 800),
    'Chef_2': (600, 800),
    'Chef_3': (600, 800),
}

fields = [
    ('Hero_Image_Desktop', "Main Desktop Banner Image (Horizontal)", "Target: 1920x1080px — Horizontal image, 16:9 aspect ratio."),
    ('Hero_Image_Mobile', "Main Mobile Banner Image (Horizontal)", "Target: 750x472px — Horizontal image, ~1.59:1 aspect ratio."),
    ('Concept_1', "First About Us Image (Vertical)", "Target: 696x825px — Vertical image, ~5:6 aspect ratio."),
    ('Concept_2', "Second About Us Image (Near-Square)", "Target: 525x544px — Near-square image, ~1:1 aspect ratio."),
    ('Concept_3', "Third About Us Image (Near-Square)", "Target: 696x693px — Near-square image, ~1:1 aspect ratio."),
    ('Cuisine_1', "First Cuisine Image (Vertical)", "Target: 529x767px — Vertical image, ~2:3 aspect ratio."),
    ('Cuisine_2', "Second Cuisine Image (Landscape)", "Target: 696x606px — Landscape image, ~1.15:1 aspect ratio."),
    ('Menu_1', "Menu Image (Wide Horizontal)", "Target: 1321x558px — Wide horizontal image, ~2.4:1 aspect ratio."),
    ('Chef_1', "First Chef Image (Vertical + Black&White)", "Target: 600x800px — Vertical image, 3:4 aspect ratio."),
    ('Chef_2', "Second Chef Image (Vertical + Black&White)", "Target: 600x800px — Vertical image, 3:4 aspect ratio."),
    ('Chef_3', "Third Chef Image (Vertical + Black&White)", "Target: 600x800px — Vertical image, 3:4 aspect ratio.")
]

# ============================================================================
# MAIN UI - TABBED LAYOUT
# ============================================================================

tab_restaurants, tab_images, tab_copy, tab_brand = st.tabs(["Restaurants", "Images", "Copy & Metadata", "Brand / Tools"])

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
                # Auto-detect primary color if URL provided
                if url_val:
                    try:
                        ok, _, _, detected_color, detected_booking = scrape_website(url_val)
                        if ok and detected_color:
                            st.session_state[f"{cleaned_name}_primary_color"] = detected_color
                            db.update_restaurant_color(cleaned_name, detected_color)
                        if ok and detected_booking:
                            st.session_state[f"{cleaned_name}_booking_platform"] = detected_booking
                            db.update_restaurant_booking(cleaned_name, detected_booking)
                    except Exception:
                        pass
                st.success(f"Restaurant '{restaurant_input}' added as: {cleaned_name}")
                st.rerun()

    st.subheader("Restaurants Content Progress")

    if st.session_state['restaurants_list']:
        rest_list = st.session_state['restaurants_list']
        for rest_idx, rest_name in enumerate(rest_list):
            col1, col2, col3, col4 = st.columns([3, 1, 2, 1.5], vertical_alignment="center")
            with col1:
                # Count uploaded images (required vs optional chef)
                image_count = 0
                chef_count = 0
                _CHEF_FIELDS = {'Chef_1', 'Chef_2', 'Chef_3'}
                for field_name, _, _ in fields:
                    uploader_key = f"{rest_name}_{field_name}"
                    persisted_key = f"{rest_name}_{field_name}_persisted"
                    has_upload = st.session_state.get(uploader_key) is not None
                    has_persisted = st.session_state.get(persisted_key, False)
                    if has_upload or has_persisted:
                        if field_name in _CHEF_FIELDS:
                            chef_count += 1
                        else:
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
                primary_color = st.session_state.get(f"{rest_name}_primary_color", "")
                color_pill = ""
                if primary_color:
                    color_pill = (
                        f'<span class="progress-pill color" '
                        f'data-color="{primary_color}" '
                        f'title="Click to copy {primary_color}">'
                        f'<span class="swatch" style="background:{primary_color};"></span>'
                        f'{primary_color}</span>'
                    )
                booking = st.session_state.get(f"{rest_name}_booking_platform", "")
                booking_label = ""
                if booking:
                    booking_label = f'<span class="booking-label">{booking}</span>'
                st.markdown(
                    f'<div class="restaurant-row{active_class}" data-name="{rest_name}">'
                    f'{color_pill}{booking_label}'
                    f'<div class="rest-name">{star}{display_name}</div>'
                    f'{url_html}'
                    f'<div class="rest-stats">'
                    f'<span class="progress-pill images">Images: {image_count}/8</span>'
                    f'<span class="progress-pill chef">Chef: {chef_count}/3</span>'
                    f'</div>'
                    f'<div class="rest-stats" style="margin-top:0.3rem">'
                    f'<span class="progress-pill alt">Alt Text: {alt_count}</span>'
                    f'<span class="progress-pill copy">Copy: {copy_count}/{len(COPY_SECTIONS)}</span>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

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

            with col3:
                notes_key = f"{rest_name}_notes"
                saved_notes_key = f"{rest_name}_saved_notes"
                st.session_state.setdefault(notes_key, "")
                st.session_state.setdefault(saved_notes_key, st.session_state[notes_key])
                notes_val = st.text_area(
                    "Notes",
                    key=notes_key,
                    placeholder="Add comments, requests and requirements here.",
                    height=100,
                    label_visibility="collapsed",
                )
                if notes_val != st.session_state[saved_notes_key]:
                    st.session_state[saved_notes_key] = notes_val
                    db.update_restaurant_notes(rest_name, notes_val)

            with col4:
                _CHECKLIST_ITEMS = [
                    ('hosting', 'Create Hosting'),
                    ('cms', 'Updated CMS'),
                    ('dns', 'Update DNS'),
                ]
                for ck, cl in _CHECKLIST_ITEMS:
                    ckey = f"{rest_name}_check_{ck}"
                    st.session_state.setdefault(ckey, False)
                    st.checkbox(cl, key=ckey)
                cl_dict = {ck: bool(st.session_state.get(f"{rest_name}_check_{ck}", False))
                           for ck, _ in _CHECKLIST_ITEMS}
                cl_json = json.dumps(cl_dict)
                prev_cl_key = f"{rest_name}_prev_checklist"
                if cl_json != st.session_state.get(prev_cl_key, ""):
                    st.session_state[prev_cl_key] = cl_json
                    db.update_restaurant_checklist(rest_name, cl_json)

            # Divider between restaurant rows
            if rest_idx < len(rest_list) - 1:
                st.markdown('<hr class="restaurant-separator">', unsafe_allow_html=True)

        # Lighten Select/Delete buttons + wire up color pill click-to-copy
        components.html("""
        <script>
        setTimeout(() => {
            const doc = window.parent.document;
            doc.querySelectorAll('button').forEach((btn) => {
                const txt = btn.textContent.trim();
                if (txt === 'Select' || txt === 'Delete') {
                    btn.classList.add('rest-btn-light');
                }
            });
            doc.querySelectorAll('.progress-pill.color[data-color]').forEach((pill) => {
                pill.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const hex = pill.getAttribute('data-color');
                    navigator.clipboard.writeText(hex);
                    const t = document.createElement('div');
                    t.className = 'color-copied-toast';
                    t.textContent = 'Copied ' + hex;
                    doc.body.appendChild(t);
                    setTimeout(() => t.remove(), 1600);
                });
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
            if name == 'Chef_1':
                st.subheader("Optional Chef Pictures")

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
                target_width, target_height = image_mappings[name]
                target_ratio = target_width / target_height
                alt_for_filename = st.session_state.get(f"{restaurant_name}_{name}_alt", "")
                new_filename = make_image_filename(restaurant_name, name, target_width, target_height, 'jpg', alt_for_filename)
                is_fresh_upload = False

                if uploaded_file:
                    is_fresh_upload = True
                    img = fix_exif_orientation(Image.open(uploaded_file))

                    width, height = img.size
                    original_ratio = width / height

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
                    new_filename = make_image_filename(restaurant_name, name, target_width, target_height, ext, alt_for_filename)

                elif has_persisted:
                    # Load previously saved image from database blob (cached)
                    blob_data, record = _load_persisted_image(restaurant_name, name)
                    if blob_data:
                        resized_img = Image.open(io.BytesIO(blob_data))
                        orig_fn = record.get('original_filename', '') if record else ''
                        ext = orig_fn.rsplit('.', 1)[-1].lower() if '.' in orig_fn else 'jpg'
                        format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
                        img_format = format_map.get(ext, 'JPEG')
                        alt_for_filename = st.session_state.get(f"{restaurant_name}_{name}_alt", "")
                        new_filename = make_image_filename(restaurant_name, name, target_width, target_height, ext, alt_for_filename)
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

                    # Auto-generate alt text when a new/different image is uploaded
                    auto_key = f"{restaurant_name}_{name}_auto_generated"
                    alt_source_key = f"{restaurant_name}_{name}_alt_source"
                    if is_fresh_upload and uploaded_file:
                        file_fingerprint = f"{uploaded_file.name}_{uploaded_file.size}"
                        if file_fingerprint != st.session_state.get(alt_source_key, ''):
                            st.session_state[alt_source_key] = file_fingerprint
                            st.session_state.pop(auto_key, None)

                    if is_fresh_upload and st.session_state.get('hf_api_token') and not st.session_state.get(auto_key):
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

                    if st.session_state[alt_key].strip():
                        copy_button(st.session_state[alt_key], f"copy_alt_{name}")

            if i < len(fields) - 1:
                st.markdown("---")

        # Save all alt text changes with one button
        st.markdown("---")
        if st.button("Save Alt Text", key="save_all_alt"):
            saved_count = 0
            for field_name, _, _ in fields:
                alt_key = f"{restaurant_name}_{field_name}_alt"
                if alt_key in st.session_state and st.session_state[alt_key].strip():
                    db.update_alt_text(restaurant_name, field_name, st.session_state[alt_key])
                    saved_count += 1
            if saved_count:
                st.success(f"Saved alt text for {saved_count} images.")
            else:
                st.info("No alt text to save.")

        # Batch download
        if any(uploaded_files.values()):
            st.markdown("---")
            if st.button("Download All Resized Images"):
                with st.spinner("Preparing ZIP file..."):
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for name, file in uploaded_files.items():
                            if file:
                                target_width, target_height = image_mappings[name]
                                img = fix_exif_orientation(Image.open(file))

                                resized_img = resize_and_crop(img, target_width, target_height)

                                if name in ['Hero_Image_Desktop', 'Hero_Image_Mobile']:
                                    opacity_key = f"{restaurant_name}_{name}_opacity"
                                    overlay_value = st.session_state.get(opacity_key, 40)
                                    if overlay_value > 0:
                                        resized_img = apply_black_overlay(resized_img, overlay_value)

                                ext = file.name.split('.')[-1].lower()
                                format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
                                img_format = format_map.get(ext, 'JPEG')
                                alt_for_filename = st.session_state.get(f"{restaurant_name}_{name}_alt", "")
                                new_filename = make_image_filename(restaurant_name, name, target_width, target_height, ext, alt_for_filename)

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

        # --- URL + Generate + Save ---
        col_url_edit, col_gen, col_save = st.columns([5, 1.2, 1.2], vertical_alignment="bottom")
        with col_url_edit:
            new_url = st.text_input(
                "Website URL",
                value=stored_url,
                placeholder="https://www.restaurant.com",
                key=f"{url_key}_copy_input",
            )
            if new_url != stored_url:
                st.session_state[url_key] = new_url
                stored_url = new_url
                db.update_restaurant_url(restaurant_name, new_url)
        with col_gen:
            generate_all = st.button("Generate Copy", type="primary", disabled=not stored_url)
        with col_save:
            if st.button("Save Changes", type="primary", key="save_copy"):
                copy_dict = {}
                for sid, _, _, _, _ in COPY_SECTIONS:
                    skey = f"{restaurant_name}_copy_{sid}"
                    copy_dict[sid] = st.session_state.get(skey, "")
                db.save_all_copy(restaurant_name, copy_dict)
                st.success("All copy and metadata saved.")

        if not stored_url:
            st.info("Enter a website URL above to enable copy generation.")

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

            col_reset, col_save, _ = st.columns([1, 1, 4])
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
                    ok, content, err, detected_color, detected_booking = scrape_website(stored_url)
                if not ok:
                    st.error(err)
                else:
                    # Auto-fill primary color if not already set
                    if detected_color:
                        c_key = f"{restaurant_name}_primary_color"
                        if not st.session_state.get(c_key):
                            st.session_state[c_key] = detected_color
                            db.update_restaurant_color(restaurant_name, detected_color)
                    # Auto-fill booking platform if not already set
                    if detected_booking:
                        b_key = f"{restaurant_name}_booking_platform"
                        if not st.session_state.get(b_key):
                            st.session_state[b_key] = detected_booking
                            db.update_restaurant_booking(restaurant_name, detected_booking)
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

        copy_section_ids = ['the_concept', 'the_cuisine', 'group_dining']
        meta_section_ids = ['meta_title', 'meta_description']

        for section_id, section_label, word_min, word_max, description in COPY_SECTIONS:
            if section_id in copy_section_ids:
                render_copy_section(restaurant_name, section_id, section_label, word_min, word_max, description)

        # === SEO META TAGS ===
        st.markdown("---")
        st.subheader("SEO Meta Tags")
        st.caption("These are the HTML meta title and description tags for search engine optimization.")

        for section_id, section_label, word_min, word_max, description in COPY_SECTIONS:
            if section_id in meta_section_ids:
                height = 68 if section_id == 'meta_title' else 80
                render_copy_section(restaurant_name, section_id, section_label, word_min, word_max, description, height=height)

# ==============================================================================
# TAB 4: BRAND
# ==============================================================================
with tab_brand:
    restaurant_name = st.session_state.get('restaurant_name_cleaned')
    if not restaurant_name:
        st.header("Brand / Tools")
        st.warning("Please select or create a restaurant in the 'Restaurants' tab first.")
    else:
        st.header(f"Brand / Tools for {restaurant_name.replace('_', ' ')}")
        stored_url = st.session_state.get(f"{restaurant_name}_website_url", "")
        detect_all = st.button("Detect", key=f"{restaurant_name}_detect_all", disabled=not stored_url)

        # --- Primary Color ---
        st.subheader("Primary Color")
        color_key = f"{restaurant_name}_primary_color"
        canonical = st.session_state.get(color_key, "")

        col_color_picker, col_color_hex, _ = st.columns(
            [0.5, 1.5, 5.2], vertical_alignment="bottom"
        )
        with col_color_picker:
            picked = st.color_picker(
                "Pick color",
                value=canonical or "#000000",
                label_visibility="collapsed",
            )
        with col_color_hex:
            typed = st.text_input(
                "Hex color",
                value=canonical,
                placeholder="#000000",
                label_visibility="collapsed",
            )
        # Sync: picker or hex input changes → update canonical + DB
        if typed and re.match(r'^#[0-9a-fA-F]{6}$', typed) and typed != canonical:
            st.session_state[color_key] = typed
            db.update_restaurant_color(restaurant_name, typed)
        elif picked != (canonical or "#000000"):
            st.session_state[color_key] = picked
            db.update_restaurant_color(restaurant_name, picked)

        # --- Reservation ---
        st.subheader("Reservation")
        if detect_all and stored_url:
            with st.spinner("Detecting brand color & booking platform..."):
                ok, _, _, detected_color, detected_booking = scrape_website(stored_url)
            if ok and detected_color:
                st.session_state[color_key] = detected_color
                db.update_restaurant_color(restaurant_name, detected_color)
            if ok and detected_booking:
                booking_key = f"{restaurant_name}_booking_platform"
                st.session_state[booking_key] = detected_booking
                db.update_restaurant_booking(restaurant_name, detected_booking)
            if ok and (detected_color or detected_booking):
                st.rerun()
            elif not ok or (not detected_color and not detected_booking):
                st.warning("Could not detect brand color or booking platform from the website.")
        booking_val = st.session_state.get(f"{restaurant_name}_booking_platform", "")
        if booking_val:
            st.caption(f"Booking platform: **{booking_val}**")

        if not stored_url:
            st.caption("Add a website URL in the Restaurants tab to enable auto-detection.")
