# doc-scan
This project optimizes the minicpmv2_6 model using INT 2 quantization for efficient text generation from document images. It reduces inference time and memory usage, enabling real-time performance on both CPU and GPU. The model is optimized for low latency and high accuracy in document text extraction . Model Info

    Model Name: openbmb/MiniCPM-V-2_6-int4

    Type: Vision-Language Model (VLM)

    Format: Hugging Face Transformers (uses .chat())

    Quantization: int4 (lightweight, ~6.8GB)

    Tokenizer: AutoTokenizer with trust_remote_code=True

    Requires: Authorization token from Hugging Face (set via use_auth_token)

🖥️ System Requirements

    Python ≥ 3.8

    GPU (recommended) or CPU

    CUDA installed for GPU acceleration

    nvidia-smi available in PATH (for GPU memory monitoring)

    Memory: ~8–12GB RAM

📦 Dependencies

Install required packages:

pip install torch torchvision transformers psutil pillow matplotlib

🗂 File Structure

.
├── script.py # This script
├── IMG-20230930-WA0003.jpg # Sample image (or replace with your own)
├── README.md # This file

🔧 How It Works

    Loads the MiniCPM-V2.6 model and tokenizer

    Measures memory usage (GPU/CPU) before/after model loading and inference

    Resizes and processes an image

    Sends an OCR prompt via .chat() using the model

    Receives structured JSON output

    Logs memory usage and performance stats

📸 Supported Input

    Identity card image (JPEG, JPG, PNG formats)

    Automatically resizes image to width 560 pixels (maintains aspect ratio)

    Can handle tilted or rotated document images

✅ Sample Prompt

"You are a document ID API that extracts and sends details from driving licenses in JSON format. Extract the following details: Document Type, Issuing Authority, Name, Date of Birth, Gender, Address or Location (Location should be separate), Document Number, Expiry Date, Nationality, Other Identifiers. If the image is upside down, turned left or right, fix it. If a field is missing, return an empty string. If not an ID, return {}. Reply only in JSON."

📤 Output Example

{
"Document Type": "Driving License",
"Issuing Authority": "Govt of India",
"Name": "John Doe",
"Date of Birth": "1995-02-18",
"Gender": "Male",
"Address": "123 Street",
"Location": "Chennai",
"Document Number": "DL-1234-5678",
"Expiry Date": "2030-01-01",
"Nationality": "Indian",
"Other Identifiers": ""
}

▶️ How to Run

Save the script as script.py, then run:

python script.py

Make sure the image path IMG-20230930-WA0003.jpg is correct or update image_path = "your_image.jpg".

📈 Logs

    Model Loading Time

    Inference Time

    Total Execution Time

    GPU/CPU memory before and after each stage

🛡️ Notes

    Do not share your Hugging Face token publicly.

    This script is synchronous and not meant for API usage. For REST API, use FastAPI instead.

    If running on CPU, expect longer inference time.

    For GGUF models, use llama.cpp instead (not covered in this version).

📎 License

MIT License — for educational and research purposes. Check model license on Hugging Face.
