import os
import requests

MODEL_DIR = "models"

# Direct download URLs
MODEL_URLS = {
    "career_encoder.pkl": "https://drive.google.com/uc?export=download&id=1q1an6T4lI-J4Q3HHCzqp5d8yJLQJhljk",
    "cluster_encoder.pkl": "https://drive.google.com/uc?export=download&id=1qwDmtY1pMpygbaOMVBwuOTljg-aR_KIv",
    "cluster_model.pkl": "https://drive.google.com/uc?export=download&id=1yr7v15kSsopcmCfoifumeDrrZ12dIQ6d",
    "career_model.pkl": "https://drive.google.com/uc?export=download&id=1c9kKWJRD4XxQA1yBJoi7AIxiUWyk_0fG",
    "feature_scaler.pkl": "https://drive.google.com/uc?export=download&id=1kL7PRu5jxJvCTicoGNeiOuPOodTOdc1S"
}

def download_file(url: str, destination: str):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(destination, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {destination}")
    else:
        print(f"Failed to download {destination}, HTTP {response.status_code}")


def ensure_models_exist():
    # Create model directory if missing
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    # Download each model file if not already present
    for filename, url in MODEL_URLS.items():
        path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(path):
            print(f"Model missing: {filename}, downloading...")
            download_file(url, path)
        else:
            print(f"Model already exists: {filename}")
