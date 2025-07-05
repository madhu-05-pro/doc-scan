import torch
import psutil
import time
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import subprocess
import matplotlib.pyplot as plt


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
model = AutoModel.from_pretrained("openbmb/MiniCPM-V-2_6-int4", use_auth_token="xxxxxxxxxxxxxx", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-V-2_6-int4', trust_remote_code=True)
model.eval()
end_loading_time = time.time()

after_loading_gpu_memory = get_gpu_memory_from_nvidia_smi()
after_loading_cpu_memory = get_cpu_memory()

print(f"GPU Memory After Model Loading: {after_loading_gpu_memory:.2f} MB")
print(f"CPU Memory After Model Loading: {after_loading_cpu_memory:.2f} MB")

image_path = "IMG-20230930-WA0003.jpg"
image = Image.open(image_path).convert('RGB')
width = 560
aspect_ratio = image.height / image.width
height = int(width * aspect_ratio)
image_resized = image.resize((width, height))
width, height = image_resized.size
print(f"Resized Image Resolution: {width}x{height}")


#correct question
question = "You are a document ID API that extracts and sends details from driving licenses in JSON format . Extract the following details: Document Type, Issuing Authority, Name, Date of Birth, Gender, Address or Location (Location should be separate), Document Number, Expiry Date, Nationality, Other Identifiers.If the image is upside down,turned left or right ,  fix it and get the details. If the image is missing details please set the value as empty string like this "" , If the image is not looks like ID means send an empty JSON. Only reply in JSON."
#question = "You are a document ID API that extracts and sends details from driving licenses in JSON format . Extract the following details: Document Type, Issuing Authority, Name, Date of Birth, Gender, Address or Location (Location should be separate), Document Number, Expiry Date, Nationality, Other Identifiers.If the image is upside down,turned left or right ,  fix it and get the details. If the image is bad or Name or Document Number is missing, send an empty JSON. Only reply in JSON."

msgs = [{'role': 'user', 'content': question}]

start_inference_time = time.time()

before_inference_gpu_memory = get_gpu_memory_from_nvidia_smi()
before_inference_cpu_memory = get_cpu_memory()

print(f"GPU Memory Before Inference: {before_inference_gpu_memory:.2f} MB")
print(f"CPU Memory Before Inference: {before_inference_cpu_memory:.2f} MB")

try:
    res = model.chat(
        image=image_resized,
        msgs=msgs,
        tokenizer=tokenizer,
        sampling=True,
        temperature=0.7,
        top_p=0.8, 
        top_k=100, 
        repeat_penalty=1.05,
        context_length=4096
    )
    print("Response:", res)
except Exception as e:
    print(f"An error occurred while calling the model: {e}")

# generated_text = ""
# try:
#     for new_text in model.chat(
#         image=image_resized,
#         msgs=msgs,
#         tokenizer=tokenizer,
#         sampling=True,
#         temperature=0.7,
#     ):
#         generated_text += new_text
#         # print(new_text, flush=True, end='')
# except Exception as e:
#     print(f"Streaming error: {e}")

# print("final output",generated_text)
end_inference_time = time.time()
inference_duration = end_inference_time - start_inference_time

final_cpu_memory = get_cpu_memory()
final_gpu_memory = get_gpu_memory_from_nvidia_smi()

after_inference_cpu_memory = get_cpu_memory()

end_time = time.time()
total_duration = end_time - start_time

print("\n--- Performance Metrics ---")
print(f"Model Loading Time: {end_loading_time - start_loading_time:.2f} seconds")
print(f"Inference Time: {inference_duration:.2f} seconds")
print(f"Total Execution Time: {total_duration:.2f} seconds")
print(f"GPU Memory After Model Loading: {after_loading_gpu_memory:.2f} MB")
print(f"Final GPU Memory: {final_gpu_memory:.2f} MB")
print(f"GPU Memory Before Inference: {before_inference_gpu_memory:.2f} MB")
print(f"GPU Memory After Inference: {final_gpu_memory:.2f} MB")
print(f"CPU Memory After Model Loading: {after_loading_cpu_memory:.2f} MB")
print(f"CPU Memory Before Inference: {before_inference_cpu_memory:.2f} MB")
print(f"CPU Memory After Inference: {after_inference_cpu_memory:.2f} MB")
print(f"Final CPU Memory: {final_cpu_memory:.2f} MB")
# plt.imshow(image)
# plt.axis('off')  
# plt.show()