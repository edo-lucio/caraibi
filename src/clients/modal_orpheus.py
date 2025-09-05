import modal

app = modal.App("orpheus-tts-app")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchaudio",
        "transformers",
        "sentencepiece",
        "numpy",
        "scipy",
        "protobuf",
        "wave",
        "huggingface_hub"
    )
    .apt_install("git")
    .run_commands(
        "git clone https://github.com/canopyai/Orpheus-TTS.git",
        "cd Orpheus-TTS && pip install orpheus-speech"
    )
)

@app.function(gpu="A100-40GB", image=image, secrets=[modal.Secret.from_name("huggingface-secret")])
@modal.fastapi_endpoint(docs=True)
def generate_speech_endpoint(prompt: str, voice: str):
    from orpheus_tts import OrpheusModel
    from fastapi.responses import Response
    import wave
    import io
    import time
    import torch

    model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod", dtype=torch.float16)
    start_time = time.monotonic()
    syn_tokens = model.generate_speech(prompt=prompt, voice=voice, max_tokens=8192)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(24000)

        total_frames = 0
        for audio_chunk in syn_tokens:
            frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
            total_frames += frame_count
            wf.writeframes(audio_chunk)

    duration = total_frames / 24000
    end_time = time.monotonic()
    print(f"It took {end_time - start_time:.2f} seconds to generate {duration:.2f} seconds of audio")

    buffer.seek(0)
    return Response(content=buffer.read(), media_type="audio/wav")
