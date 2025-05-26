from typing import Optional, List
from src.clients.text_models_clients import DeepSeekAPIClient 

class DeepSeekImagePromptGenerator:
    """
    A class to generate a list of image prompts for text-to-image models based on a YouTube script and description.
    
    Attributes:
        api_client (DeepSeekAPIClient): The API client to interact with DeepSeek.
        max_tokens (int): Maximum tokens for API responses.
        default_style (str): Default artistic style for prompts (e.g., 'realism').
        default_detail_level (str): Default detail level (e.g., 'high').
    """
    
    def __init__(
        self,
        api_client: DeepSeekAPIClient,
        max_tokens: int = 1000,
        default_style: str = "realism",
        default_detail_level: str = "high"
    ):
        """
        Initialize the DeepSeekImagePromptGenerator.
        
        Args:
            api_client: An instance of DeepSeekAPIClient for API interactions.
            max_tokens: Maximum tokens for API responses (default: 1000).
            default_style: Default artistic style (e.g., 'realism', 'anime') (default: 'realism').
            default_detail_level: Default detail level ('low', 'medium', 'high', 'ultra') (default: 'high').
        """
        self.api_client = api_client
        self.max_tokens = max_tokens
        self.default_style = default_style
        self.default_detail_level = default_detail_level

    def generate_image_prompts(
        self,
        script: str,
        description: str,
        num_prompts: Optional[int] = None
    ) -> List[str]:
        """
        Generate a list of image prompts based on a YouTube script and its description.
        
        Args:
            script: The YouTube script string to analyze for visual scenes.
            description: The description used to generate the script, providing context.
            num_prompts: Number of image prompts to generate. If None, the API determines an appropriate number.
            
        Returns:
            A list of image prompt strings, each suitable for text-to-image models.
            Returns an empty list if generation fails.
        """
        if not script or not isinstance(script, str):
            raise ValueError("Script must be a non-empty string")
        if not description or not isinstance(description, str):
            raise ValueError("Description must be a non-empty string")
        if num_prompts is not None and (not isinstance(num_prompts, int) or num_prompts <= 0):
            raise ValueError("Number of prompts must be a positive integer or None")

        # Construct the prompt for the DeepSeek API
        prompt_instruction = (
            f"You are a professional image prompt engineer. I need you to create {num_prompts} detailed image generation prompts based on the following script content: "
            f"Identify key visual scenes or moments from the script that would make compelling images, using the description for context (e.g., tone, theme). "
            f"Each prompt should be descriptive, including specific details about lighting, colors, composition, and atmosphere. "
            f"Return the prompts as a list separated by newlines. "
            f"Ensure prompts are vivid, standalone, and optimized for high-quality image generation. ", 
            f"""
                Each image prompt should:
                1. Capture a specific vivid scene or moment from the script
                2. Include detailed visual elements (colors, lighting, composition)
                3. Be self-contained and specific enough to generate a coherent image
                4. Be 2-4 sentences long
                
                Important: Your response should be a list of strings, each string being a complete image prompt on a new line.
                Do not include any explanations, comments, or additional text."""
            )

        if num_prompts is not None:
            prompt_instruction += f"Generate exactly {num_prompts} image prompts. "
        else:
            prompt_instruction += (
                f"Generate an appropriate number of image prompts (e.g., 3-10) based on the script’s visual content, ",
                f"ensuring each prompt corresponds to a distinct, significant scene. "
            )

        # Include script and description, truncating script to avoid token limits
        base_prompt = (
            f"{prompt_instruction}\n\n"
            f"Description: '{description}' (truncated if longer).\n"
            f"Script: '{script[:60000]}' (truncated if longer)."
        )

        generated_text = self.api_client.generate_text(base_prompt, self.max_tokens)

        if not generated_text:
            print("Failed to generate image prompts.")
            return []

        prompts = []
        try:
            prompts = generated_text.split("\n")
        except Exception as e:
            print(f"Error parsing image prompts: {e}")
            return []

                    
        with open("prompts.txt", "w", encoding="utf-8") as f:
            f.write("".join(prompts))

        prompts = [prompt for prompt in prompts if prompt != ""]
        return prompts[:num_prompts] if num_prompts is not None else prompts

def generate_prompts(
    script: str,
    description: str,
    num_prompts: Optional[int] = None
) -> List[str]:
    """
    Convenience function to generate image prompts for a YouTube script and description.
    
    Args:
        script: The YouTube script string to analyze for visual scenes.
        description: The description used to generate the script, providing context.
        num_prompts: Number of image prompts to generate. If None, the API determines an appropriate number.
        
    Returns:
        A list of image prompt strings, each suitable for text-to-image models.
        Returns an empty list if generation fails.
    """
    # Initialize the API client
    api_client = DeepSeekAPIClient()
    
    # Initialize the image prompt generator
    prompt_generator = DeepSeekImagePromptGenerator(
        api_client,
        max_tokens=1000,
        default_style="realism",
        default_detail_level="high"
    )
    
    # Generate and return the prompts
    return prompt_generator.generate_image_prompts(script, description, num_prompts)