import os
from huggingface_hub import snapshot_download

def download_model():
    # Create models directory if it doesn't exist
    os.makedirs("model", exist_ok=True)
    
    # Download model from Hugging Face
    print("Downloading model files... This may take a while...")
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir="model",
        ignore_patterns=["*.md", "*.txt"]
    )
    print("Download complete! You can now run the app.")

if __name__ == "__main__":
    download_model()