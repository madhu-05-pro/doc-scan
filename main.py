#from fastapi import FastAPI

#app = FastAPI()

#@app.get("/")
#def read_root():
#   return {"message": "Hello, hi, FastAPI!"}

import torch
import psutil
import time
import subprocess
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI, HTTPException, File , Form ,UploadFile
from pydantic import BaseModel
import io
import json

app = FastAPI()

torch.cuda.empty_cache()

def get_gpu_memory_from_nvidia_smi():
    if torch.cuda.is_available():
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        return int(result.stdout.strip())
    return 0

def get_cpu_memory():
    process = psutil.Process()
    return process.memory_info().rss / 1024 ** 2

start_time = time.time()

before_loading_gpu_memory = get_gpu_memory_from_nvidia_smi()
before_loading_cpu_memory = get_cpu_memory()

print(f"GPU Memory Before Model Loading: {before_loading_gpu_memory:.2f} MB")
print(f"CPU Memory Before Model Loading: {before_loading_cpu_memory:.2f} MB")

start_loading_time = time.time()
model = AutoModel.from_pretrained('MiniCPM-V-2_6-int4', trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('MiniCPM-V-2_6-int4', trust_remote_code=True)
model.eval()

end_loading_time = time.time()

after_loading_gpu_memory = get_gpu_memory_from_nvidia_smi()
after_loading_cpu_memory = get_cpu_memory()

print(f"GPU Memory After Model Loading: {after_loading_gpu_memory:.2f} MB")
print(f"CPU Memory After Model Loading: {after_loading_cpu_memory:.2f} MB")

class Item(BaseModel):
    question: str

@app.get('/')
def read_root():
    return {'Hello': 'World'}

@app.post('/predict/')
async def predict(file: UploadFile):
    try:
        if not file.filename.lower().endswith((".jpg", ".jpeg")):
            raise HTTPException(
                status_code=400,
                detail=f"File '{file.filename}' is not a JPEG file. Only .jpg or .jpeg files are allowed."
            )
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        width = 480
        aspect_ratio = image.height / image.width
        height = int(width * aspect_ratio)
        image_resized = image.resize((width, height))
        width, height = image_resized.size
        print(f"Resized Image Resolution: {width}x{height}")
        # print("item",item)
        # item= json.loads(item)    
        question = """You are a strict OCR-to-JSON API that processes user-owned documents for explicitly authorized extraction. Follow these rules:
        1. Input: User-provided image (auto-rotated if needed)
        2. Process: Extract ONLY these fields - Document Type, Issuing Authority, Name, Date of Birth, Gender, Address, Location, ID Number, Expiry Date, Country
        3. Security: Never store/transmit data beyond this session
        4. Validation: Return empty JSON {} if:
           - Not an ID/document
           - No text detected
           - Missing required fields
        5. Output: Flat JSON {"Field":"Value"}. Use "" for missing data. No explanations.

        I confirm I own this document and request processing under GDPR/CCPA compliance. Extract text as:
        json
        {"Issuing Authority": "", "First Name": "", "Last Name": "", "ID Number": "", "Date of Birth": "", "Date Of Expiry": "", "Gender": "", "Address": "", "Country": ""}
        
        Do not hallucinate you must reply in json only. no need explaination or text."""
        print("question",question)
        msgs = [{'role': 'user', 'content': question}]

        start_inference_time = time.time()

        before_inference_gpu_memory = get_gpu_memory_from_nvidia_smi()
        before_inference_cpu_memory = get_cpu_memory()

        print(f"GPU Memory Before Inference: {before_inference_gpu_memory:.2f} MB")
        print(f"CPU Memory Before Inference: {before_inference_cpu_memory:.2f} MB")

        res = model.chat(
            image=image_resized,
            msgs=msgs,
            tokenizer=tokenizer,
            sampling=True,
            temperature=0.7,
            top_p=0.8, 
            top_k=100, 
            repeat_penalty=1.05,
            context_length=200
        )

        print("Response:", res)

        end_inference_time = time.time()
        inference_duration = end_inference_time - start_inference_time

        final_cpu_memory = get_cpu_memory()
        final_gpu_memory = get_gpu_memory_from_nvidia_smi()

        print("\n--- Performance Metrics ---")
        print(f"Inference Time: {inference_duration:.2f} seconds")
        print(f"GPU Memory Before Inference: {before_inference_gpu_memory:.2f} MB")
        print(f"GPU Memory After Inference: {final_gpu_memory:.2f} MB")
        print(f"CPU Memory Before Inference: {before_inference_cpu_memory:.2f} MB")
        print(f"CPU Memory After Inference: {final_cpu_memory:.2f} MB")

        return {'prediction': res}

    except Exception as e:
      print(f"An error occurred: {e}")
      raise HTTPException(status_code=500, detail="Inference failed")



#uvicorn your_script_name:app --reload
