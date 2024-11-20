import os
import sys

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from openai import OpenAI

import requests
from dotenv import load_dotenv
from utils import generate_image_from_description

load_dotenv()

def get_description_from_centrala() -> str:
    base_url = os.getenv("CENTRALA_URL")
    api_key = os.getenv("AIDEVS3_API_KEY")
    
    if not base_url or not api_key:
        raise EnvironmentError("CENTRALA_URL or AIDEVS3_API_KEY environment variable not set.")
    
    url = f"{base_url}/data/{api_key}/robotid.json"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ConnectionError(f"Failed to fetch data from {url}, status code: {response.status_code}")
    
    data = response.json()
    description = data.get("description", "")
    
    if not description:
        raise ValueError("Description not found in the JSON response.")
    
    return description

def main() -> None:
    description = get_description_from_centrala()
    print(description)

    image_url = generate_image_from_description(description)
    print(image_url)

if __name__ == "__main__":
    main()
