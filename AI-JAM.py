import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import gc

st.set_page_config(layout="wide", page_title="AI-JAM", page_icon="🎨")

st.markdown("""
    <style>
    .main { background-color: #1E1E1E; }
    .big-font {
        font-size: 60px !important;
        font-weight: bold;
        color: white;
        text-align: center;
        padding: 20px 0 0 0;
        margin-bottom: 0;
    }
    .subheader {
        font-size: 30px;
        font-style: italic;
        color: white;
        text-align: center;
        margin-top: 5px;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #3498db;
        color: white;
        font-weight: bold;
    }
    img {
        max-width: 768px !important;
        margin: auto !important;
        display: block !important;
    }
    .progress-bar-text {
        color: white;
        text-align: center;
        font-size: 18px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-JAM</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator</p>', unsafe_allow_html=True)

@st.cache_resource
def load_pipe():
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info('Loading model... This may take a few minutes the first time...')
        
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
            requires_safety_checker=False
        ).to(device)
        
        if device == "cuda":
            pipe.enable_attention_slicing()
        
        return pipe
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

pipe = load_pipe()
progress_text = st.empty()
progress_bar = st.empty()


def update_progress_callback(step: int, timestep: int, latents: torch.FloatTensor):
    progress = min((step + 1) / 100, 1.0) * 100
    progress_bar.progress(min(progress / 100, 1.0))
    progress_text.markdown(f"<p class='progress-bar-text'>Generation Progress: {min(progress, 100):.0f}%</p>", 
                         unsafe_allow_html=True)
    
def generate_art(prompt):
    if pipe is None:
        st.error("Model failed to load. Please try refreshing the page.")
        return None
        
    try:
        device = pipe.device
        with torch.inference_mode():
            progress_bar.progress(0)
            progress_text.markdown("<p class='progress-bar-text'>Starting generation...</p>", 
                                 unsafe_allow_html=True)
            
            image = pipe(
                prompt,
                num_inference_steps=100,
                guidance_scale=7.5,
                callback=update_progress_callback,
                callback_steps=1
            ).images[0]
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        
        return image
    except Exception as e:
        st.error(f"Generation error: {str(e)}")
        return None

predefined_flavors = [
    "A refreshing summer drink with notes of citrus and mint",
    "A decadent dessert combining dark chocolate and raspberry",
    "An aromatic coffee blend with hints of caramel and cinnamon",
    "A savory appetizer balancing aged cheese and truffle",
    "A unique ice cream inspired by lavender and honey"
]

st.sidebar.markdown("### 🍽️ Flavor Palette")
selected_flavor = st.sidebar.radio(
    "Select a flavor inspiration or create your own:",
    ["Create your own"] + predefined_flavors,
    index=0,
    format_func=lambda x: x if x != "Create your own" else "✨ Create your own flavor"
)

if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "", key="flavor_input")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor, key="flavor_input")

if st.button("🎨 Generate your masterpiece"):
    if not flavor_description:
        st.warning("Please enter a flavor description first!")
    else:
        if pipe is None:
            st.error("Model is not loaded. Please try refreshing the page.")
        else:
            st.markdown("<p style='color: #cccccc; font-style: italic; text-align: center;'>Generation usually takes 30-40 seconds</p>", unsafe_allow_html=True)
            prompt = f"A vibrant, artistic representation of {flavor_description}. Digital art, colorful, abstract, food illustration."
            image = generate_art(prompt)
                
            if image:
                progress_bar.empty()
                progress_text.empty()
                
                st.image(image, caption=f"Artistic impression of: {flavor_description}", width=None)
         
                buf = io.BytesIO()
                image.save(buf, format="PNG", quality=100)
                st.download_button(
                    label="📥 Download Your Artwork",
                    data=buf.getvalue(),
                    file_name=f"{flavor_description.replace(' ', '_')}_art.png",
                    mime="image/png"
                )

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-JAM")
st.sidebar.markdown("AI-JAM is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence.")