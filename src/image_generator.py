import os
import time
import json
import requests
from typing import Dict, List, Optional
from src.clients.image_models_clients import TensorArtClient

def load_config(config_path: str, config_name: str = "images") -> Dict:
    """
    Load configuration from a JSON file.
    
    Args:
        config_path: Path to the configuration JSON file.
        config_name: Name of the configuration to use from the JSON file (default: "images").
        
    Returns:
        Dictionary containing the configuration, or default stages if loading fails.
    """
    default_config = {
        "stages": [
            {"type": "INPUT_INITIALIZE", "inputInitialize": {}},
            {"type": "DIFFUSION", "diffusion": {}}
        ]
    }
    
    try:
        with open(config_path, "r") as f:
            all_configs = json.load(f)
        
        if not isinstance(all_configs, dict):
            print(f"Invalid config format in {config_path}. Using default config.")
            return default_config
            
        if config_name in all_configs:
            config = all_configs[config_name]
            if not isinstance(config, dict) or "stages" not in config:
                print(f"Config '{config_name}' missing 'stages'. Using default config.")
                return default_config
            return config
        else:
            print(f"Config '{config_name}' not found in {config_path}. Available: {list(all_configs.keys())}")
            print("Using default config.")
            return default_config
            
    except FileNotFoundError:
        print(f"Config file not found at {config_path}. Using default config.")
        return default_config
    except json.JSONDecodeError:
        print(f"Error decoding JSON from {config_path}. Using default config.")
        return default_config
    except Exception as e:
        print(f"Unexpected error loading config from {config_path}: {e}. Using default config.")
        return default_config

class TensorArtGenerator:
    """
    A class to generate and save images using the TensorArt API for single or multiple prompts.
    
    Attributes:
        client (TensorArtClient): The TensorArt API client for image generation.
        output_folder (str): Directory to save generated images.
        output_format (str): File format for saved images (e.g., 'png', 'jpg').
        default_stages (List[Dict]): Default stages configuration for API requests.
    """
    
    def __init__(
        self,
        app_id: str = os.getenv("TENSOR_ART_APP_ID"),
        api_key: str = os.getenv("TENSOR_ART_API_KEY"),
        output_folder: str = "./output",
        output_format: str = "png",
        config_path: Optional[str] = "./config",
        config_name: str = "images"
    ):
        """
        Initialize the TensorArtGenerator.
        
        Args:
            app_id: The TensorArt application ID (default: from TENSORART_APP_ID env variable).
            api_key: The TensorArt API key (default: from TENSORART_API_KEY env variable).
            output_folder: Directory to save generated images (default: './output').
            output_format: File format for saved images (default: 'png').
            config_path: Path to a JSON config file for stages (default: None).
            config_name: Name of the configuration in the JSON file (default: 'images').
        """
        if not app_id or not api_key:
            raise ValueError("app_id and api_key must be provided, either directly or via environment variables")
        if not output_folder or not isinstance(output_folder, str):
            raise ValueError("output_folder must be a non-empty string")
        if not output_format or not isinstance(output_format, str):
            raise ValueError("output_format must be a non-empty string")

        self.client = TensorArtClient(app_id=app_id, api_key=api_key)
        self.output_folder = output_folder.rstrip("/")
        self.output_format = output_format.lstrip(".")
        self.default_stages = load_config(config_path, config_name).get("stages", [
            {"type": "INPUT_INITIALIZE", "inputInitialize": {}},
            {"type": "DIFFUSION", "diffusion": {}}
        ])

    def _save_output(self, image_url: str, image_path: str) -> bool:
        """
        Download and save an image from a URL to the specified path.
        
        Args:
            image_url: The URL of the image to download.
            image_path: The file path to save the image.
            
        Returns:
            True if the image is saved successfully, False otherwise.
        """
        try:
            if not image_url or not isinstance(image_url, str):
                print("Invalid image URL.")
                return False
            
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            response = requests.get(image_url, stream=True, timeout=30)
            response.raise_for_status()
            
            with open(image_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            
            print(f"Image downloaded successfully: {image_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to download image from {image_url}: {e}")
            return False
        except OSError as e:
            print(f"Failed to save image to {image_path}: {e}")
            return False
        except Exception as e:
            print(f"Unexpected error saving image: {e}")
            return False

    def generate_image(self, prompt: str, output_path: Optional[str] = None) -> Optional[str]:
        """
        Generate and save a single image for a given prompt using the TensorArt API.
        
        Args:
            prompt: The text prompt for image generation.
            output_path: The file path to save the image (default: auto-generated in output_folder).
            
        Returns:
            The file path of the saved image, or None if generation or saving fails.
        """
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("Prompt must be a non-empty string")

        # Generate default output path if not provided
        timestamp = str(int(time.time() * 1000))
        output_path = output_path or f"{self.output_folder}/image_{timestamp}.{self.output_format}"

        try:
            # Generate image via TensorArtClient
            image_urls = self.client.generate_images(
                prompts=[prompt],
                stages=self.default_stages,
                max_wait_time=300,
                poll_interval=15
            )

            if not image_urls:
                print(f"Failed to generate image for prompt: {prompt}")
                return None

            # Save the first image
            if self._save_output(image_urls[0], output_path):
                return output_path
            return None

        except Exception as e:
            print(f"Error generating image: {e}")
            return None

    def generate_images(self, prompts: List[str], output_path: str) -> List[str]:
        """
        Generate and save images for a list of prompts using the TensorArt API.
        
        Args:
            prompts: List of text prompts for image generation.
            output_path: Directory to save the images.
            
        Returns:
            A list of file paths for saved images, or an empty list if generation or saving fails.
        """
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("Prompts must be a non-empty list of non-empty strings")
        if not output_path or not isinstance(output_path, str):
            raise ValueError("output_path must be a non-empty string")

        output_path = output_path.rstrip("/")
        saved_paths = []

        try:
            # Generate images via TensorArtClient
            image_urls = self.client.generate_images(
                prompts=prompts,
                stages=self.default_stages,
                max_wait_time=300,
                poll_interval=15
            )

            if not image_urls:
                print("Failed to generate images.")
                return []

            # Save each image
            for i, image_url in enumerate(image_urls):
                image_path = f"{output_path}/{i}_{int(time.time() * 1000)}.{self.output_format}"
                if self._save_output(image_url, image_path):
                    saved_paths.append(image_path)

            return saved_paths

        except Exception as e:
            print(f"Error generating images: {e}")
            return []

def generate_image(prompt: str, output_path: Optional[str] = None) -> Optional[str]:
    """
    Convenience function to generate and save a single image for a given prompt using the TensorArt API.
    
    Args:
        prompt: The text prompt for image generation.
        output_path: The file path to save the image (default: auto-generated in ./output).
        
    Returns:
        The file path of the saved image, or None if generation or saving fails.
    """
    try:
        generator = TensorArtGenerator()
        return generator.generate_image(prompt, output_path)
    except Exception as e:
        print(f"Error in convenience function generate_image: {e}")
        return None

def generate_images(prompts: List[str], output_path: str = "./output") -> List[str]:
    """
    Convenience function to generate and save images for a list of prompts using the TensorArt API.
    
    Args:
        prompts: List of text prompts for image generation.
        output_path: Directory to save the images.
        
    Returns:
        A list of file paths for saved images, or an empty list if generation or saving fails.
    """
    try:
        generator = TensorArtGenerator()
        return generator.generate_images(prompts, output_path)
    except Exception as e:
        print(f"Error in convenience function generate_images: {e}")
        return []