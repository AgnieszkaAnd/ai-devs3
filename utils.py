import base64
import os
import re
import requests
#import whisper
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI
from openai.types import ImagesResponse
from typing import Any, Dict, List, Literal, Optional, Union


load_dotenv()
client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")


def generate_local_llm_response(
    system_template: str,
    human_template: str,
    model: str = "llama2:7b",
    stream: bool = False,
    response_format: str = "json",
    api_url: str = "http://localhost:11434/api/generate",
) -> str:
    payload = {
        "model": model,
        "prompt": human_template,
        "stream": stream,
        "format": response_format,
        "system": system_template,
    }
    try:
        response = requests.post(api_url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", result)
    except requests.exceptions.RequestException as e:
        return f"error: {str(e)}"


def create_openai_client(api_key: Optional[str] = None) -> OpenAI:
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.error("OpenAI API key is not set.")
        raise ValueError("OpenAI API key is required.")
    client = OpenAI()
    client.api_key = api_key
    return client

def encode_images_to_base64(images: List) -> List[Dict[str, str]]:
    encoded_images = []
    for image in images:
        encoded_image = base64.b64encode(image.read()).decode('utf-8')
        encoded_images.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}
        })
    return encoded_images

def generate_openai_request_content(system_template: str, human_template: str, images: List) -> List[Dict[str, Any]]:
    content = [{"type": "text", "text": human_template}]
    content.extend(encode_images_to_base64(images))
    return [
        {"role": "system", "content": system_template},
        {"role": "user", "content": content}
    ]

def openai_vision_create(
    system_template: str,
    human_template: str,
    images: List,
    model: str = "gpt-4o",
    temperature: float = 0.5,
    full_response: bool = False,
) -> Union[Dict[str, Any], str]:
    content = [{"type": "text", "text": human_template}]
    for image in images:
        content.append(
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64, {base64.b64encode(image.read()).decode('utf-8')}"
                },
            }
        )
    client = OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY")

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_template},
            {"role": "user", "content": content},
        ],
        temperature=temperature,
    )
    return response if full_response else response.choices[0].message

def generate_image_from_description(description: str) -> str: 
    client = OpenAI()
    client.api_key = os.getenv("OPENAI_API_KEY")

    response = client.images.generate(
        model="dall-e-3",
        prompt=description,
        n=1,
        size="1024x1024",
        quality="standard",
        response_format="url"
    )
    
    image_url = response.data[0].url
    return image_url

# def whisper_transcribe(
#     path: str, model_name: str = "turbo", full_response: bool = False
# ) -> Union[Dict[str, Any], str]:
#     model = whisper.load_model(model_name)
#     result = model.transcribe(path)
#     return result if full_response else result["text"]


def aidevs_send_answer(task: str, answer: str) -> requests.Response:
    apikey: str = os.getenv("AIDEVS3_API_KEY")
    url: str = os.getenv("AIDEVS3_API_URL")
    payload: Dict[str, Any] = {"task": task, "apikey": apikey, "answer": answer}
    return requests.post(url, json=payload)


def group_files_by_type(
    directory: str, file_types={".png": "Images", ".mp3": "Audio", ".txt": "Text"}
) -> Dict[str, List[str]]:
    """
    Groups files in the given directory by their type (.png, .mp3, .txt).
    Args:
        directory (str): The path to the directory to scan.
    Returns:
        Dict[str, List[str]]: A dictionary where keys are file types and values are lists of file paths.
    """
    grouped_files = defaultdict(list)

    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)

        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file_name)
            if ext in file_types:
                grouped_files[file_types[ext]].append(file_name)

    return grouped_files


def extract_answer(text: str) -> Optional[str]:
    match = re.search(r"<ANSWER>(.*?)</ANSWER>", text, re.DOTALL)
    return match.group(1).strip() if match else None