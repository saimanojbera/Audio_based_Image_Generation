# Audio-Based Image Generation

## Overview
This project integrates **OpenAI's Whisper** for **speech-to-text transcription** with **Stable Diffusion (img2img)** to modify images based on transcribed audio. Users can upload an **image** and an **audio file**, and the system will **transcribe the speech from the audio** and use it as a prompt to modify the image.

The project is built using:
- **Hugging Face Transformers & Diffusers**
- **Stable Diffusion v1.5**
- **Gradio for the UI**
- **OpenAI's Whisper API for speech-to-text conversion**

## Features
- **Audio Transcription**: Converts speech from an uploaded audio file into text.
- **Text-Based Image Modification**: Uses Stable Diffusion to modify the uploaded image based on the transcribed text.
- **Interactive Gradio Interface**: Allows users to upload files and view results in real-time.
- **GPU Acceleration**: Uses CUDA for fast image generation.

## Installation

Install the necessary dependencies:

```bash
pip install gradio diffusers torch transformers accelerate safetensors python-dotenv pillow requests
```

## How It Works
1. **Load the Stable Diffusion Model**: The pipeline loads `runwayml/stable-diffusion-v1-5` using Hugging Face Diffusers.
2. **Audio Transcription**: 
   - The uploaded audio file is sent to OpenAI's Whisper API.
   - The response contains the transcribed text.
3. **Image Modification**:
   - The text from Whisper is used as a prompt for Stable Diffusion.
   - The input image is processed and modified using `img2img`.
4. **Gradio Interface**:
   - Users upload an **image** and an **audio file**.
   - The system displays the transcribed text and the **modified image**.

## Code Breakdown

### 1. **Load Stable Diffusion Model**
```python
from diffusers import StableDiffusionImg2ImgPipeline

def load_pipeline():
    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    )
    pipe.to("cuda")  # Use GPU for better performance
    return pipe

pipe = load_pipeline()
```

### 2. **Transcribe Audio using Whisper API**
```python
def transcribe_audio(audio_filepath):
    with open(audio_filepath, "rb") as f:
        response = requests.post(WHISPER_API_URL, headers=HEADERS, files={"file": f})

    if response.status_code == 200:
        result = response.json()
        return result.get("text", "No transcription available")
    else:
        return f"Error: {response.status_code}, {response.text}"
```

### 3. **Modify the Image Based on Audio Transcription**
```python
def generate_image(input_image, audio_filepath):
    transcription = transcribe_audio(audio_filepath)
    
    if transcription.startswith("Error"):
        return transcription, None

    prompt = transcription
    image = input_image.convert("RGB").resize((512, 512))  # Preprocess image

    result = pipe(
        prompt=prompt,
        image=image,
        strength=0.75,        
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]

    return f"Prompt taken from audio file: {prompt}", result
```

### 4. **Gradio Interface**
```python
import gradio as gr

iface = gr.Interface(
    fn=generate_image,
    inputs=[
        gr.Image(type="pil", label="Upload an Image"),
        gr.Audio(type="filepath", label="Upload an Audio File")
    ],
    outputs=[
        gr.Textbox(label="Prompt taken from audio"),
        gr.Image(type="pil", label="Modified Image")
    ],
    title="Audio-Powered Image Modification",
    description="Upload an image and an audio file. The system will transcribe the speech and modify the image."
)

iface.launch(share=True)
```

## Credits
- **Stable Diffusion**: [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
- **Whisper API**: [OpenAI Whisper](https://openai.com/research/whisper)
- **Gradio**: [Gradio Library](https://gradio.app)

---