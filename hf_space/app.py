import os
import time
import requests
import gradio as gr
from fastapi import FastAPI
from pydantic import BaseModel

HF_MODEL_ID = "AICPADSLIM/PortSLM-Qwen2.5-VL-3B"
# Using HF serverless inference API
HF_TOKEN = os.environ.get("HF_TOKEN")

def generate_text(prompt, max_tokens=512, temperature=0.7):
    """Fallback to mock if HF inference fails, otherwise call the HF inference API."""
    if not HF_TOKEN:
        return "ERROR: HF_TOKEN environment variable not set. Cannot run inference."
        
    api_url = f"https://api-inference.huggingface.co/models/{HF_MODEL_ID}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_tokens, "temperature": temperature}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=payload, timeout=60)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and result:
                return result[0].get("generated_text", "")
        else:
            return f"Error {response.status_code}: {response.text}"
    except Exception as e:
        return f"Request failed: {str(e)}"
    
    return "Generation failed."

def gradio_interface(prompt, temperature, max_tokens):
    return generate_text(prompt, max_tokens, temperature)

with gr.Blocks(title="PortSLM (Qwen2.5-VL-3B)") as demo:
    gr.Markdown("# ⚓ PortSLM Inference API (Hugging Face Spaces)")
    gr.Markdown("인천신항 컨테이너 터미널 도메인 특화 SLM")
    
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(lines=5, label="Input Prompt", placeholder="DG 컨테이너 적재 규칙에 대해 설명해줘.")
            temperature = gr.Slider(minimum=0.0, maximum=2.0, value=0.7, label="Temperature")
            max_tokens = gr.Slider(minimum=10, maximum=1024, value=512, label="Max Tokens")
            submit = gr.Button("Generate")
        with gr.Column():
            output = gr.Textbox(lines=10, label="Model Output")
            
    submit.click(fn=gradio_interface, inputs=[prompt, temperature, max_tokens], outputs=output)

app = FastAPI()
app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
