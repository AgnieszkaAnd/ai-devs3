import os
import sys

from typing import Dict, List
from dotenv import load_dotenv

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import openai_vision_create, create_openai_client

load_dotenv()

def categorize_file_content_with_openai(content: str) -> Dict[str, bool]:
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    client = create_openai_client(openai_api_key)
    messages = [
        {
            "role": "system",
            "content": (
                "Categorize the following Polish text into: "
                "'people' for information about people, "
                "'nobody' for explicit absence of people, "
                "'hardware' for hardware details, "
                "'software' for software details, "
                "or 'other' for any other content. "
                "Only respond with 'people' or 'hardware' if certain. "
                "Exclude software-related content in the final response."
            )
        },
        {
            "role": "user",
            "content": content
        }
    ]
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=20
    )
    
    result = response.choices[0].message.content.strip().lower().split(',')
    is_people = 'people' in result
    is_hardware = 'hardware' in result

    # Simplified categorization without re-check
    return {
        "people": is_people,
        "hardware": is_hardware
    }

def get_text_from_files(directory_path: str) -> List[Dict[str, str]]:
    return [
        {"filename": filename, "content": open(os.path.join(directory_path, filename), 'r', encoding='utf-8').read()}
        for filename in os.listdir(directory_path) if filename.endswith('.txt')
    ]

def get_text_from_images(directory_path: str) -> List[Dict[str, str]]:
    return [
        {
            "filename": filename,
            "content": openai_vision_create(
                system_template="Extract text from images.",
                human_template="",
                images=[open(os.path.join(directory_path, filename), 'rb')],
                model="gpt-4o",
                temperature=0.1
            ).content
        }
        for filename in os.listdir(directory_path) if filename.endswith('.png')
    ]

def transcribe_audio_files(directory_path: str, client) -> List[Dict[str, str]]:
    return [
        {
            "filename": filename,
            "content": client.audio.transcriptions.create(
                model="whisper-1",
                file=open(os.path.join(directory_path, filename), 'rb')
            ).text
        }
        for filename in os.listdir(directory_path) if filename.endswith('.mp3')
    ]

def categorize_by_content(texts: List[Dict[str, str]]) -> Dict[str, List[str]]:
    categorized = [categorize_file_content_with_openai(text["content"]) for text in texts]
    people_related = [texts[i]["filename"] for i, category in enumerate(categorized) if category["people"]]
    hardware_related = [texts[i]["filename"] for i, category in enumerate(categorized) if category["hardware"]]

    return {
        "people": people_related,
        "hardware": hardware_related
    }

def extract_relevant_files() -> Dict[str, List[str]]:
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/pliki_z_fabryki'))
    return categorize_by_content(get_text_from_files(directory_path))

def extract_relevant_images() -> Dict[str, List[str]]:
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/pliki_z_fabryki'))
    return categorize_by_content(get_text_from_images(directory_path))

def extract_relevant_audio(client) -> Dict[str, List[str]]:
    directory_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/pliki_z_fabryki'))
    return categorize_by_content(transcribe_audio_files(directory_path, client))

def merge_and_sort_collections(relevant_files: Dict[str, List[str]], 
                               relevant_images: Dict[str, List[str]], 
                               relevant_audio: Dict[str, List[str]]) -> Dict[str, List[str]]:
    merged_people = set(relevant_files["people"] + relevant_images["people"] + relevant_audio["people"])
    merged_hardware = set(relevant_files["hardware"] + relevant_images["hardware"] + relevant_audio["hardware"])
    
    return {
        "people": sorted(merged_people),
        "hardware": sorted(merged_hardware)
    }

def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    client = create_openai_client(api_key)

    relevant_files = extract_relevant_files()
    print("People related files:", relevant_files["people"])
    print("Hardware related files:", relevant_files["hardware"])

    relevant_images = extract_relevant_images()
    print("People related images:", relevant_images["people"])
    print("Hardware related images:", relevant_images["hardware"])

    relevant_audio = extract_relevant_audio(client)
    print("People related audio:", relevant_audio["people"])
    print("Hardware related audio:", relevant_audio["hardware"])

    merged_collections = merge_and_sort_collections(relevant_files, relevant_images, relevant_audio)
    print("Merged and sorted people related files:", merged_collections["people"])
    print("Merged and sorted hardware related files:", merged_collections["hardware"])

if __name__ == "__main__":
    main()
