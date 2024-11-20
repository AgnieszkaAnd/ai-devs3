import os
import sys

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from loguru import logger
from typing import Any, Dict, List, Literal, Optional, Union
import base64
from openai import OpenAI
from utils import openai_vision_create


def run():
    system_template: str = """
    Prompt:
    You are navigation assistant fluent in topology of Polish cities. You analyze maps carefully. You know street names in every Polish city.
    Your task is to identify the name of a city in Poland.
    
    Guidelines:
    
        You will receive four images, each showing a map of part of a city.
        Three images depict the same city in Poland, while one is a decoy designed to mislead you.
        Carefully analyze and list all visible crossroads in each image to aid your reasoning.
        Carefully analyze also landmarks and their locations such as stores, cemeteries, bus stations, and other notable features to help identify the city.
        Use your observations to determine which city is represented in the majority of the images.
        Think out loud and provide at least 3 candidates for the city. Then do critique of your reasoning.
        Choose the city for which you have the most hard evidence.
        Provide your answer in the format:
        <answer>City Name</answer>.
    """
    human_template: str = ""
    directory: str = "../data/map/"
    image_names: List[str] = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    images = [open(f"{directory}{image_name}", "rb") for image_name in image_names]
    response = openai_vision_create(
        system_template, human_template, images, model="gpt-4o", temperature=0.1
    )
    logger.info(f"{response.content}")


if __name__ == "__main__":
    run()