import time
import hmac
import hashlib
import json
import requests
from typing import Dict, List, Optional
import os

from dotenv import load_dotenv
load_dotenv()

class TensorArtClient:
    """
    A class to handle interactions with the TensorArt API for image generation.
    
    Attributes:
        app_id (str): The TensorArt application ID.
        api_key (str): The TensorArt API key.
        base_url (str): The TensorArt API endpoint URL.
    """
    
    def __init__(
        self,
        app_id: str = os.getenv("TENSOR_ART_APP_ID"),
        api_key: str = os.getenv("TENSOR_ART_API_KEY"),
        base_url: str = "https://ap-east-1.tensorart.art/v1/jobs"
    ):
        """
        Initialize the TensorArtClient.
        
        Args:
            app_id: The TensorArt application ID (default: from TENSORART_APP_ID env variable).
            api_key: The TensorArt API key (default: from TENSORART_API_KEY env variable).
            base_url: The API endpoint URL (default: TensorArt v1 endpoint).
        """
        if not app_id or not api_key:
            raise ValueError("app_id and api_key must be provided, either directly or via environment variables")
        self.app_id = app_id
        self.api_key = api_key
        self.base_url = base_url

    def _generate_signature(self, timestamp: str) -> str:
        """
        Generate HMAC SHA256 signature for API authentication.
        
        Args:
            timestamp: The current timestamp as a string (milliseconds since epoch).
            
        Returns:
            The HMAC SHA256 signature as a hexadecimal string.
        """
        message = f"appId={self.app_id}&timestamp={timestamp}"
        return hmac.new(
            self.api_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256
        ).hexdigest()

    def _prepare_headers(self) -> Dict[str, str]:
        """
        Prepare authenticated API request headers.
        
        Returns:
            A dictionary of headers including Content-Type, appId, timestamp, signature, and Authorization.
        """
        timestamp = str(int(time.time() * 1000))
        return {
            "Content-Type": "application/json",
            "appId": self.app_id,
            "timestamp": timestamp,
            "signature": self._generate_signature(timestamp),
            "Authorization": f"Bearer {self.api_key}"
        }

    def _prepare_stages(self, stages: List[Dict], prompts: List[str]) -> List[Dict]:
        """
        Prepare the stages configuration for the API request by updating prompt count and prompt texts.
        
        Args:
            stages: List of stage configurations.
            prompts: List of prompt strings for image generation.
            
        Returns:
            The updated stages configuration.
        """
        updated_stages = stages.copy()  # Avoid modifying the input
        for stage in updated_stages:
            if stage.get("type") == "INPUT_INITIALIZE":
                stage.setdefault("inputInitialize", {})["count"] = len(prompts)
            if stage.get("type") == "DIFFUSION":
                stage.setdefault("diffusion", {})["prompts"] = [{"text": prompt} for prompt in prompts]
        return updated_stages

    def generate_images(
        self,
        prompts: List[str],
        stages: List[Dict],
        max_wait_time: int = 300,
        poll_interval: int = 15
    ) -> List[str]:
        """
        Synchronously generate images via the TensorArt API and return their URLs.
        
        Args:
            prompts: List of text prompts for image generation.
            stages: List of stage configurations for the API request.
            max_wait_time: Maximum time to wait for job completion in seconds (default: 300).
            poll_interval: Time between status checks in seconds (default: 15).
            
        Returns:
            A list of image URLs on success, or an empty list if the request fails.
        """
        if not prompts or not all(isinstance(p, str) and p.strip() for p in prompts):
            raise ValueError("Prompts must be a non-empty list of non-empty strings")
        if not stages or not isinstance(stages, list):
            raise ValueError("Stages must be a non-empty list of dictionaries")
        if max_wait_time <= 0 or poll_interval <= 0:
            raise ValueError("max_wait_time and poll_interval must be positive")

        headers = self._prepare_headers()
        request_id = str(int(time.time() * 1000))
        payload = {
            "requestId": request_id,
            "stages": self._prepare_stages(stages, prompts)
        }

        try:
            # Submit the job
            post_response = requests.post(self.base_url, headers=headers, data=json.dumps(payload))
            post_response.raise_for_status()  # Raises for non-200 status
            job_response = post_response.json()
            job_id = job_response["job"]["id"]

            # Poll for job completion
            start_time = time.time()
            retry_count = 0
            max_retries = 1

            while time.time() - start_time < max_wait_time:
                try:
                    status_response = requests.get(f"{self.base_url}/{job_id}", headers=headers)
                    status_response.raise_for_status()
                    status_data = status_response.json()
                    job_status = status_data.get("job", {}).get("status")

                    if job_status == "SUCCESS":
                        print("Job completed successfully!")
                        images = status_data["job"].get("successInfo", {}).get("images", [])
                        return [image["url"] for image in images if "url" in image]

                    elif job_status == "FAILED":
                        print("Job failed.")
                        return []

                    time.sleep(poll_interval)

                except requests.exceptions.RequestException as e:
                    print(f"Status check failed: {e}")
                    return []

            # Handle timeout with one retry
            if retry_count < max_retries:
                print("Job timed out. Retrying once...")
                retry_count += 1
                return self.generate_images(prompts, stages, max_wait_time, poll_interval)

            print("Job timed out after retry.")
            return []

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            return []
        except (KeyError, IndexError, ValueError) as e:
            print(f"Error parsing API response: {e}")
            return []