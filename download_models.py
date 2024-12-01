import os
from huggingface_hub import snapshot_download

def download_model():
    model_path = os.path.join(os.getcwd(), "model")
    os.makedirs(model_path, exist_ok=True)

    print("Downloading model files... This may take a while...")
    snapshot_download(
        repo_id="runwayml/stable-diffusion-v1-5",
        local_dir=model_path,
        ignore_patterns=["*.md", "*.txt"],
        local_dir_use_symlinks=False
    )
    print("Download complete! You can now run the app.")

if __name__ == "__main__":
    download_model()