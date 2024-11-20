import os
import sys
import glob
from typing import List, Optional

# Add the parent directory to the system path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.openai_client import OpenAIClient

def transcribe_audio_file(file_path: str, client: OpenAIClient, model: str = "whisper-1") -> str:
    with open(file_path, "rb") as audio_file:
        transcription = client.client.audio.transcriptions.create(model=model, file=audio_file)
    return transcription.text

def generate_transcripts_from_directory(directory_path: str, client: OpenAIClient) -> List[str]:
    audio_files = glob.glob(os.path.join(directory_path, "*.m4a"))
    return [transcribe_audio_file(file_path, client) for file_path in audio_files]

def get_street_name_from_transcript(transcript_content: str, client: OpenAIClient) -> Optional[str]:
    user_message = (
        "What is the name of the street of university where Andrzej Maj is a lecturer? "
        "Provide answer in Polish. Provide your thinking process in steps. Then provide critique. "
        "Use your general knowledge about Polish university campuses and its street locations. "
        "Provide only one final street name in <ANSWER> </ANSWER> tags."
    )
    response = client.get_response(transcript_content, user_message)
    return response.strip() if response else None

def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY environment variable not set.")
    
    client = OpenAIClient(api_key=api_key)
    transcripts = generate_transcripts_from_directory("../data/recordings", client)
    combined_transcripts = "\n".join(transcripts)
    print(combined_transcripts)

    street_name = get_street_name_from_transcript(combined_transcripts, client)
    print(street_name if street_name else "Street name could not be determined.")

if __name__ == "__main__":
    main()
