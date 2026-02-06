import streamlit as st
from PIL import Image
import io
import base64
import re
import zipfile

# Function to get base64 of image
def get_base64_of_bin_file(bin_file_path):
    with open(bin_file_path, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Mapping of image base names to target dimensions (width, height) and aspect ratio
image_mappings = {
    'Hero_Image_Desktop': (1920, 1080, 1920/1080),  # 16:9 - Horizontal
    'Hero_Image_Mobile': (1080, 680, 1080/680),  # ~1.588 - Horizontal
    'Concept_1': (696, 825, 696/825),  # ~0.843 (close to 3:4) - Vertical
    'Concept_2': (525, 544, 525/544),  # ~0.965 (close to 1:1) - Square
    'Concept_3': (696, 693, 696/693),  # ~1.004 (near 1:1) - Square
    'Cuisine_1': (529, 767, 529/767),  # ~0.690 (close to 3:4) - Vertical
    'Cuisine_2': (696, 606, 696/606),  # ~1.149 (close to 1:1) - Square
    'Menu_1': (1321, 558, 1321/558),  # ~2.367 - Horizontal
    'Chef_1': (698, 836, 698/836),  # ~0.835 (close to 3:4) - Vertical
    'Chef_2': (698, 836, 698/836),  # ~0.835 (close to 3:4) - Vertical
    'Chef_3': (698, 836, 698/836),  # ~0.835 (close to 3:4) - Vertical
}

def resize_and_crop(img, target_width, target_height):
    original_width, original_height = img.size
    target_ratio = target_width / target_height
    original_ratio = original_width / original_height
    
    if original_ratio > target_ratio:
        # Image is wider: resize based on height, crop width
        new_height = target_height
        new_width = int(new_height * original_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        left = (new_width - target_width) // 2
        img = img.crop((left, 0, left + target_width, new_height))
    else:
        # Image is taller: resize based on width, crop height
        new_width = target_width
        new_height = int(new_width / original_ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Center crop
        top = (new_height - target_height) // 2
        img = img.crop((0, top, target_width, top + target_height))
    
    return img

def is_black_and_white(img):
    # Check if the image mode is grayscale ('L' or '1')
    if img.mode in ('L', '1'):
        return True
    
    # Convert to RGB if not already, and check if all pixels are grayscale (R==G==B) with tolerance
    img_rgb = img.convert('RGB')
    pixels = list(img_rgb.getdata())
    return all(abs(r - g) < 15 and abs(g - b) < 15 and abs(b - r) < 15 for r, g, b in pixels)  # Increased tolerance to 15

# Streamlit app
st.title("Starr Restaurant Image Editor")

# Add background image styling
bg_img = get_base64_of_bin_file('bg.jpg')  # Use relative path for deployment
css = f"""
<style>
[data-testid="stAppViewContainer"] {{
    position: relative;
    background-image: url("data:image/jpeg;base64,{bg_img}");
    background-size: cover;
    background-position: center;
}}
[data-testid="stAppViewContainer"]::before {{
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background-color: rgba(0, 0, 0, 0.4);
    z-index: 1;
}}
[data-testid="stAppViewContainer"] > * {{
    position: relative;
    z-index: 2;
    color: white !important;
}}
div.stButton > button {{
    width: 80% !important; /* Reduce width by 20% (from 100% to 80%) */
    float: right;
}}
</style>
"""
st.markdown(css, unsafe_allow_html=True)

# Step 1: Input restaurant name with Apply button right-aligned and thinner
with st.form(key="restaurant_form"):
    restaurant_name = st.text_input("Enter Restaurant Name", help="This name will be the first word of the resized image filenames and should avoid special characters for best results.")
    apply_button = st.form_submit_button(label="Apply", help="Click to apply the restaurant name")
    if apply_button:
        if not restaurant_name:
            st.error("Please enter a restaurant name to proceed.")
        else:
            st.session_state.restaurant_name_cleaned = re.sub(r'[^a-zA-Z0-9_]', '', restaurant_name.replace(' ', '_'))
            st.rerun()  # Rerun to update the app state

# Use the cleaned restaurant name from session state
restaurant_name = st.session_state.get("restaurant_name_cleaned")
if not restaurant_name:
    st.stop()

# Step 2: Upload fields with descriptions, styled blocks, and immediate processing
uploaded_files = {}
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

for i, (name, header, description) in enumerate(fields):
    with st.container():
        # Apply custom styling to the container with larger headers
        st.markdown(f"""
            <style>
            div.stContainer-{{id}} {{
                border: 2px solid #ccc;
                padding: 10px;
                margin-bottom: 10px;
                border-radius: 5px;
            }}
            div.stContainer-{{id}} h4 {{
                font-size: 1.2em; /* 20% larger than default */
            }}
            </style>
        """, unsafe_allow_html=True)
        st.write(f"**{header}**")
        st.markdown(description, unsafe_allow_html=True)
        uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key=name, label_visibility="collapsed")
        uploaded_files[name] = uploaded_file
        
        # Clear previous messages for this field
        st.empty()  # Clear previous warnings or success messages
        
        # Process image immediately if uploaded
        if uploaded_file:
            img = Image.open(uploaded_file)
            # Handle EXIF orientation
            exif = img._getexif()
            if exif and 274 in exif:  # Orientation tag
                orientation = exif[274]
                if orientation == 3:  # 180° rotation
                    img = img.rotate(180, expand=True)
                elif orientation == 6:  # 90° clockwise
                    img = img.rotate(-90, expand=True)
                elif orientation == 8:  # 90° counterclockwise
                    img = img.rotate(90, expand=True)
            width, height = img.size
            original_ratio = width / height
            target_width, target_height, target_ratio = image_mappings[name]
            # Override target_ratio for Concept_3 to expect square (1:1)
            if name == 'Concept_3':
                target_ratio = 1.0  # 1:1 as per UI description
            allowed_deviation = target_ratio * 0.3  # 30% deviation
            
            # Aspect ratio check
            aspect_ok = abs(original_ratio - target_ratio) <= allowed_deviation
            
            # Black and white check for Chef images with loading spinner
            is_chef = name in ['Chef_1', 'Chef_2', 'Chef_3']
            bw_ok = False
            if is_chef:
                with st.spinner('Checking if image is grayscale...'):
                    bw_ok = is_black_and_white(img)
            else:
                bw_ok = True  # Non-Chef images don't need this check
            
            # Show warnings independently
            if not aspect_ok:
                st.warning("💡Oops Funky Ingredients: You can use this but the image's aspect ratio deviates by more than 30% from the target. Processing may crop substantially and not look as delicious as intended.")
            
            if is_chef and not bw_ok:
                st.warning("💡Brand guidelines strongly suggest Black&White images of the chefs to keep with the editorial look.")
            
            # Show success only if both conditions are met for Chef images, or just aspect for others
            if aspect_ok and (not is_chef or bw_ok):
                st.success("✅ Perfect, looks delicious!")
            
            # Process and provide download link for individual image
            resized_img = resize_and_crop(img, target_width, target_height)
            
            # Map file extension to PIL-compatible format
            ext = uploaded_file.name.split('.')[-1].lower()
            format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
            img_format = format_map.get(ext, 'JPEG')  # Default to JPEG if extension not recognized
            
            # New filename
            new_filename = f"{restaurant_name}_{name}_{target_width}x{target_height}.{ext}"
            
            # Save to in-memory buffer
            img_buffer = io.BytesIO()
            resized_img.save(img_buffer, format=img_format, quality=95)
            img_buffer.seek(0)
            
            # Provide download link for this image
            st.download_button(
                label=f"Download Resized {name}",
                data=img_buffer,
                file_name=new_filename,
                mime=f"image/{ext}",
                key=f"download_{name}"
            )
    
    # Add horizontal line after each section (except the last one)
    if i < len(fields) - 1:
        st.markdown("<hr style='border: 1px solid #ccc; margin: 10px 0;'>", unsafe_allow_html=True)

# Batch download button for all uploaded images
if any(uploaded_files.values()):
    if st.button("Download All Resized Images"):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for name, file in uploaded_files.items():
                if file:
                    target_width, target_height, _ = image_mappings[name]
                    img = Image.open(file)
                    resized_img = resize_and_crop(img, target_width, target_height)
                    
                    # Map file extension to PIL-compatible format
                    ext = file.name.split('.')[-1].lower()
                    format_map = {'jpg': 'JPEG', 'jpeg': 'JPEG', 'png': 'PNG'}
                    img_format = format_map.get(ext, 'JPEG')  # Default to JPEG if extension not recognized
                    
                    # New filename
                    new_filename = f"{restaurant_name}_{name}_{target_width}x{target_height}.{ext}"
                    
                    # Save to in-memory buffer and add to ZIP under 'Resized' folder
                    img_buffer = io.BytesIO()
                    resized_img.save(img_buffer, format=img_format, quality=95)
                    img_buffer.seek(0)
                    zip_file.writestr(f"Resized/{new_filename}", img_buffer.read())
        
        # Provide download link for ZIP
        zip_buffer.seek(0)
        st.download_button(
            label="Download ZIP of All Resized Images",
            data=zip_buffer,
            file_name=f"{restaurant_name}_resized_images.zip",
            mime="application/zip",
            key="download_all"
        )
