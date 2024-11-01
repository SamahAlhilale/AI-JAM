import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
import io
import gc

st.set_page_config(layout="wide", page_title="AI-JAM", page_icon="🎨")

st.markdown("""
    <style>
    .main {
        background-color: #1E1E1E;
    }
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
    .stTextInput>div>div>input {
        background-color: #ecf0f1;
        color: black !important;
    }
    .stProgress > div > div > div > div {
        background-color: #3498db;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-JAM</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator by Alsherazi Club</p>', unsafe_allow_html=True)

# Model loading
if 'models_loaded' not in st.session_state:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "runwayml/stable-diffusion-v1-5"
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe = pipe.to(device)
    
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = clip_model.to(device)
    
    st.session_state.pipe = pipe
    st.session_state.clip_model = clip_model
    st.session_state.tokenizer = tokenizer
    st.session_state.device = device
    st.session_state.models_loaded = True

def generate_image_from_flavor(flavor_description, progress_bar, status_text):
    prompt = f"A vibrant, artistic representation of {flavor_description}. Digital art, colorful, abstract, food illustration."
    
    # Reduced steps for faster generation
    num_inference_steps = 20
    
    try:
        # Update progress for tokenization
        progress_bar.progress(0.1)
        status_text.text("Processing text... 10%")
        
        inputs = st.session_state.tokenizer(
            prompt, 
            padding="max_length", 
            max_length=st.session_state.tokenizer.model_max_length, 
            truncation=True, 
            return_tensors="pt"
        )
        
        progress_bar.progress(0.2)
        status_text.text("Analyzing flavor description... 20%")
        
        with torch.no_grad():
            # Get text embeddings
            text_embeddings = st.session_state.clip_model(**inputs.to(st.session_state.device)).last_hidden_state
            progress_bar.progress(0.3)
            status_text.text("Starting image generation... 30%")
            
            # Define callback for progress updates
            def callback(step: int, timestep: int, latents: torch.FloatTensor):
                progress = 0.3 + (step / num_inference_steps * 0.7)  # Scale from 30% to 100%
                progress_bar.progress(progress)
                percentage = int(progress * 100)
                status_text.text(f"Generating image... {percentage}%")
            
            # Generate image with progress tracking
            image = st.session_state.pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=7.5,
                callback=callback,
                callback_steps=1
            ).images[0]
            
            progress_bar.progress(1.0)
            status_text.text("Generation complete! 100%")
            
        return image
    except Exception as e:
        st.error(f"Error in image generation: {str(e)}")
        return None

# Flavor menu
st.sidebar.markdown("### 🍽️ Flavor Palette")
predefined_flavors = [
    "A refreshing summer drink with notes of citrus and mint",
    "A decadent dessert combining dark chocolate and raspberry",
    "An aromatic coffee blend with hints of caramel and cinnamon",
    "A savory appetizer balancing aged cheese and truffle",
    "A unique ice cream inspired by lavender and honey"
]

selected_flavor = st.sidebar.radio(
    "Select a flavor inspiration or create your own:", 
    ["Create your own"] + predefined_flavors, 
    index=0,
    format_func=lambda x: x if x != "Create your own" else "✨ Create your own flavor"
)

# Main area
if selected_flavor == "Create your own":
    flavor_description = st.text_input("Enter your flavor description:", "", key="flavor_input")
else:
    flavor_description = st.text_input("Enter your flavor description:", selected_flavor, key="flavor_input")

if st.button("🎨 Generate Artistic Impression"):
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        image = generate_image_from_flavor(flavor_description, progress_bar, status_text)
        
        if image is not None:
            st.image(image, caption=f"Artistic impression of: {flavor_description}", use_column_width=True)
            
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            st.download_button(
                label="📥 Download Your Artwork",
                data=buf.getvalue(),
                file_name=f"{flavor_description.replace(' ', '_')}_art.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"An unexpected error occurred: {str(e)}")
    finally:
        progress_bar.empty()
        gc.collect()
        torch.cuda.empty_cache()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-JAM")
st.sidebar.markdown("AI-JAM is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence. Our goal is to create a unique sensory experience by translating flavors into visual art.")