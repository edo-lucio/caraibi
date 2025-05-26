
lora_config = {
    "water": [{
                "type": "INPUT_INITIALIZE",
                "inputInitialize": {
                    "seed": "2669222025",
                    "count": 1
                    }
                },
                {
                    "type": "DIFFUSION",
                    "diffusion": {
                        "width": 1920,
                        "height": 1080,
                        "prompts": [
                            {
                                "text": ""
                            }
                        ],
                        "negativePrompts": [
                            {}
                        ],
                        "sdModel": "836320460259045440",
                        "sdVae": "ae.sft",
                        "sampler": "DPM2",
                        "steps": 25,
                        "cfgScale": 8,
                        "clipSkip": 2,
                        "lora": {
                            "items": [
                                {
                                    "loraModel": "827308403396275703",
                                    "weight": 0.8
                                }
                            ]
                        }
                    }
                },
                {
                    "type": "IMAGE_TO_UPSCALER",
                    "image_to_upscaler": {
                        "hr_upscaler": "4x-UltraSharp",
                        "hrResizeX": 1536,
                        "hrResizeY": 2304,
                        "hr_scale": 2,
                        "hr_second_pass_steps": 10,
                        "denoising_strength": 0.3
                    }
                }
            ]}