from src.script_generator import YouTubeScriptGenerator, DeepSeekAPIClient
from src.prompt_generator import DeepSeekPromptGenerator, generate_prompts
from src.image_generator import TensorArtGenerator, generate_image, generate_images, generate_images_batch
from src.audio_generator import generate_audio, emotions
from typing import Optional
import asyncio


def _generate_script(title: str, description: str, target_length: int, output_path: str) -> Optional[str]:
    api_client = DeepSeekAPIClient()
    generator = YouTubeScriptGenerator(api_client)
    return generator.generate_youtube_script(title, description, target_length, output_path)

def _generate_audio(text: str, client: str = "zonos", output_path: str = "output.mp3"):
    chunk_lenght = 200
    if client == "zonos": 
        generate_audio(text, chunk_lenght, output_path)
        
def main():
    description = """
        Narrative script of describing a fun fact in history that involves some sexy theme.
        The story must have some degree of veridicity meaning it shouldn't be completely made up but it doesn't need
        to be entirely true. Use your extensive training knowledge to write a script.

    """
    target_length = 12000  # Desired script length in characters
    title = ""

    output_path = "new_opt.txt"
    script = _generate_script(title, description, target_length, output_path)

    image_description = "detailed images of realistic characters, convey emotions and be descriptive with the settings."
    # script = "beutiful girl having a stink problem she is hated by everyone. she likes garlic very much. and everyone seem to subtly act as they know it."
    # image_prompts = generate_prompts(script, image_description)
    # print(image_description)
    # generate_images(image_prompts)

    audio_script = asyncio.run(_generate_audio(script))
        

    if script:
        print("Generated Script:")
    else:
        print("Script generation failed.")
    # if image_prompts: 
    #     print("Generated Image prompts")

if __name__ == "__main__":
    main()