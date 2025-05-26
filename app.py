from src.script_generator import YouTubeScriptGenerator, generate_script
from src.prompt_generator import DeepSeekImagePromptGenerator, generate_prompts
from src.image_generator import TensorArtGenerator, generate_image, generate_images

def main():
    description = """
    Informative video about the life in a day of a medioeval person. It should start describing from early 
    from early in the morning until night. Describe the ordinary and the special occasions as well.
    Be highly detailed, ironical and make references to the audience. Be witty. Be also accurate in the historical details.
    Try to elicit, curiosity and surprise in the audience
    """
    target_length = 12000  # Desired script length in characters
    title = ""

    # script = generate_script(title, description, target_length)
    image_description = "Depict ordinary scenes of life of medioeval family. Be highly descriptive and depict the emotions" \
    "and imagery of the script."
    script = "beutiful girl having a stink problem she is hated by everyone. she likes garlic very much. and everyone seem to subtly act as they know it."
    image_prompts = generate_prompts(script, image_description)
    print(type(image_prompts))

    generate_images(image_prompts)

    if script:
        print("Generated Script:")
    else:
        print("Script generation failed.")
    if image_prompts: 
        print("Generated Image prompts")

if __name__ == "__main__":
    main()