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
    </style>
    """, unsafe_allow_html=True)

st.markdown('<p class="big-font">AI-JAM</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Flavor-Inspired Art Generator by Alsherazi Club</p>', unsafe_allow_html=True)

@st.cache_resource
def load_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to("cuda")
    return pipe

pipe = load_pipe()

def generate_image(prompt):
    with torch.no_grad():
        image = pipe(
            prompt,
            num_inference_steps=20,
            guidance_scale=7.5,
        ).images[0]
    return image

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

if st.button("🎨 Generate Artistic Impression"):
    if not flavor_description:
        st.warning("Please enter a flavor description first!")
    else:
        with st.spinner("Creating your art..."):
            prompt = f"A vibrant, artistic representation of {flavor_description}, digital art, colorful food illustration"
            image = generate_image(prompt)
            
            if image:
                st.image(image, caption=f"Artistic impression of: {flavor_description}", use_column_width=True)
                
                buf = io.BytesIO()
                image.save(buf, format="PNG")
                st.download_button(
                    label="📥 Download Your Artwork",
                    data=buf.getvalue(),
                    file_name=f"{flavor_description.replace(' ', '_')}_art.png",
                    mime="image/png"
                )
        
        torch.cuda.empty_cache()
        gc.collect()

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎨 About AI-JAM")
st.sidebar.markdown("AI-JAM is a project by Alsherazi Club that explores the intersection of culinary inspiration and artificial intelligence.")