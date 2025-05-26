import time
from typing import Optional
import re
from src.clients.text_models_clients import DeepSeekAPIClient
from dotenv import load_dotenv
load_dotenv()

CHARS_PER_TOKEN = 4.0

def estimate_tokens(text: str) -> int:
    """Estimate the number of tokens in a given text."""
    return len(text) // CHARS_PER_TOKEN

def clean_generated_text(text: str) -> str:
    """
    Clean the generated text by removing content between parentheses and double asterisks.
    
    Args:
        text: The text to clean.
        
    Returns:
        The cleaned text with non-spoken elements removed.
    """
    # Remove text between parentheses, e.g., (scene fades out)
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove text between double asterisks, e.g., **upbeat music**
    text = re.sub(r'\*\*.*?\*\*', '', text)
    # Remove extra whitespace resulting from removals
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def find_last_complete_sentence(text: str, max_length: int) -> str:
    """
    Trim the text to the last complete sentence within max_length.
    
    Args:
        text: The text to trim.
        max_length: Maximum length in characters.
        
    Returns:
        The trimmed text ending with a complete sentence.
    """
    if len(text) <= max_length:
        return text
    
    # Find the last sentence-ending punctuation within max_length
    trimmed = text[:max_length]
    last_sentence_end = max(
        trimmed.rfind("."),
        trimmed.rfind("?"),
        trimmed.rfind("!")
    )
    
    # If no sentence-ending punctuation is found, fall back to last space
    if last_sentence_end == -1:
        last_space = trimmed.rfind(" ")
        return trimmed[:last_space] + "..." if last_space != -1 else trimmed
    return trimmed[:last_sentence_end + 1]

class YouTubeScriptGenerator:
    """
    A class to generate YouTube video scripts using a provided API client.
    
    Attributes:
        api_client: The API client to use for text generation.
        max_tokens_per_request (int): Maximum tokens per API request.
    """
    
    def __init__(self, api_client: DeepSeekAPIClient, max_tokens_per_request: int = 8192):
        """
        Initialize the YouTubeScriptGenerator.
        
        Args:
            api_client: An instance of an API client (e.g., DeepSeekAPIClient).
            max_tokens_per_request: Maximum tokens per API request (default: 8192).
        """
        self.api_client = api_client
        self.max_tokens_per_request = max_tokens_per_request

    def generate_youtube_script(self, title: str, description: str, target_length: int) -> Optional[str]:
        """
        Generate a YouTube video script based on the description and target character length.
        The script contains only the narrator's spoken dialogue, with no scene descriptions, music cues, or meta-commentary.
        
        Args:
            description: The description of the video content.
            target_length: Desired script length in characters.
            
        Returns:
            The generated script or None if generation fails.
        """
        if not description or not isinstance(description, str):
            raise ValueError("Description must be a non-empty string")
        if not isinstance(target_length, int) or target_length <= 0:
            raise ValueError("Target length must be a positive integer")
            
        # Initialize variables
        current_script = ""
        remaining_chars = target_length
        
        # Define a safe max tokens for each API call (reserve some for prompt)
        safe_max_tokens = self.max_tokens_per_request - 100
        base_prompt = (
            f"Write a YouTube video script based on the following title: \n {title} and description: \n{description}. "
            f"The script should be engaging, clear, and suitable for a YouTube audience. "
            f"Include only the spoken dialogue for the narrator, excluding any scene descriptions, music cues, or meta-commentary. "
            f"Target approximately {target_length} characters total, but for this part, "
            f"""Each image prompt should:
                1. Capture a specific vivid scene or moment from the script
                2. Include detailed visual elements (colors, lighting, composition)
                3. Be self-contained and specific enough to generate a coherent image
                4. Be 2-4 sentences long
                Important: Your response should be a JSON array of strings, each string being a complete image prompt.
                Do not include any explanations, comments, or additional text"""
            )
        
        while remaining_chars > 0:
            # Adjust max tokens for this request
            tokens_needed = min(safe_max_tokens, estimate_tokens(current_script + " ") + 100)
            
            # If nearing the target length, instruct to conclude the script
            is_final_segment = remaining_chars < (safe_max_tokens * CHARS_PER_TOKEN * 0.5)
            if current_script:
                prompt = (
                    f"Continue the following YouTube video script without repeating content and maintaining coherence: "
                    f"'{current_script[-1000:] if len(current_script) > 1000 else current_script}'. "
                    f"Based on the original description: {description}. "
                    f"Generate the next part of the script, including only the spoken dialogue for the narrator, "
                    f"excluding any scene descriptions, music cues, or meta-commentary, "
                    f"targeting up to {safe_max_tokens * CHARS_PER_TOKEN} characters."
                )
                if is_final_segment:
                    prompt += (
                        f" Since this is near the end of the script, conclude the narrative coherently with a strong closing statement or question, "
                        f"ensuring the total length approaches {target_length} characters."
                    )
            else:
                prompt = base_prompt
                
            # Call the API via the client
            generated_text = self.api_client.generate_text(prompt, int(tokens_needed))
            if not generated_text:
                print("Failed to generate script segment.")
                return None
                
            # Clean the generated text to remove non-spoken elements
            cleaned_text = clean_generated_text(generated_text)
            
            # Append cleaned text to the script
            current_script += cleaned_text
            current_length = len(current_script)
            remaining_chars = max(0, target_length - current_length)
            
            # Break if we've reached or exceeded the target length
            if current_length >= target_length:
                break
                
            # Add a small delay to avoid rate limiting
            time.sleep(1)
        
        # Trim to the last complete sentence within target_length
        current_script = find_last_complete_sentence(current_script, target_length)
            
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(current_script)

        return current_script

def generate_script(title: str, description: str, target_length: int) -> Optional[str]:
    """
    Convenience function to generate a YouTube script.
    
    Args:
        description: The description of the video content.
        target_length: Desired script length in characters.
        
    Returns:
        The generated script or None if generation fails.
    """
    api_client = DeepSeekAPIClient()
    generator = YouTubeScriptGenerator(api_client)
    return generator.generate_youtube_script(title, description, target_length)

