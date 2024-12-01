import os
from huggingface_hub import snapshot_download

def download_model():
    # Get absolute path for model directory
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model"))
    os.makedirs(model_path, exist_ok=True)
    
    print(f"Downloading model files to {model_path}...")
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=model_path,
        ignore_patterns=["*.md", "*.txt"],
        local_dir_use_symlinks=False
    )
    print("Download complete! You can now run the app.")

if __name__ == "__main__":
    download_model()