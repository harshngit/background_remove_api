import base64
import uuid
import os
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np
import onnxruntime as ort

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output folder
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

# Download and cache the model
MODEL_PATH = "u2net.onnx"
MODEL_URL = "https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx"

def download_model():
    """Download the background removal model if not present."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model downloaded successfully")

# Initialize model
try:
    download_model()
    session = ort.InferenceSession(MODEL_PATH, providers=['CPUExecutionProvider'])
except Exception as e:
    print(f"Model initialization error: {e}")
    session = None

class ImageRequest(BaseModel):
    base64_image: str | None = None
    image_url: str | None = None


def normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """Normalize image for model input."""
    img = img / 255.0
    img = (img - mean) / std
    return img


def remove_background(image: Image.Image) -> Image.Image:
    """Remove background using U2-Net model."""
    if session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # Prepare image
    original_size = image.size
    image = image.convert('RGB')
    image = image.resize((320, 320))
    
    # Convert to array and normalize
    img_array = np.array(image).astype(np.float32)
    img_array = normalize(img_array)
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, 0)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: img_array})[0][0][0]
    
    # Process mask
    mask = Image.fromarray((output * 255).astype(np.uint8))
    mask = mask.resize(original_size, Image.Resampling.LANCZOS)
    
    # Apply mask
    image_original = image.resize(original_size)
    image_original = image_original.convert('RGBA')
    mask_array = np.array(mask)
    image_array = np.array(image_original)
    image_array[:, :, 3] = mask_array
    
    return Image.fromarray(image_array)


def process_image(image_bytes: bytes) -> Image.Image:
    """Process image and remove background."""
    try:
        input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        return remove_background(input_image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


@app.post("/remove-bg")
async def remove_bg(data: ImageRequest):
    if not data.base64_image and not data.image_url:
        raise HTTPException(status_code=400, detail="Provide base64_image or image_url")

    try:
        # ----- BASE64 INPUT -----
        if data.base64_image:
            if "," in data.base64_image:
                data.base64_image = data.base64_image.split(",")[1]
            image_bytes = base64.b64decode(data.base64_image)

        # ----- IMAGE URL INPUT -----
        elif data.image_url:
            response = requests.get(data.image_url, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download image")
            image_bytes = response.content

        # Process image
        output_image = process_image(image_bytes)

        # Save with unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")
        output_image.save(file_path, format="PNG")

        # Construct public URL
        base_url = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost:8000")
        if not base_url.startswith("http"):
            base_url = f"https://{base_url}"

        public_url = f"{base_url}/output/{file_id}.png"

        return JSONResponse(content={
            "success": True,
            "image_url": public_url,
            "file_id": file_id
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/")
def home():
    return {
        "message": "Background Remover API is running",
        "version": "1.0.0",
        "status": "model_loaded" if session else "model_not_loaded"
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model_ready": session is not None}