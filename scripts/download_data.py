import os
import requests
import zipfile
import io

def download_wikitext():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    print(f"Downloading WikiText-103 from {url}...")
    
    response = requests.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            zip_ref.extractall(data_dir)
        print("Success! Data extracted to data/wikitext-103")
    else:
        print(f"Failed to download. Status code: {response.status_code}")

if __name__ == "__main__":
    download_wikitext()
