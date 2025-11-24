import base64
import uuid
import os
import requests
from io import BytesIO
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from PIL import Image
import numpy as np

app = FastAPI(title="Background Remover API")

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output directory
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")


class ImageRequest(BaseModel):
    base64_image: str | None = None
    image_url: str | None = None
    threshold: int = 240  # Adjust for background removal sensitivity


def simple_bg_remove(image: Image.Image, threshold: int = 240) -> Image.Image:
    """
    Simple background removal based on edge detection and color threshold.
    Works best with solid/white backgrounds.
    """
    # Convert to RGBA
    img = image.convert("RGBA")
    data = np.array(img)
    
    # Get RGB channels
    r, g, b, a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    
    # Create mask: identify near-white pixels
    mask = (r > threshold) & (g > threshold) & (b > threshold)
    
    # Set alpha to 0 for background pixels
    data[:,:,3] = np.where(mask, 0, 255)
    
    return Image.fromarray(data)


def get_image_bytes(data: ImageRequest) -> bytes:
    """Download or decode image from request."""
    if data.base64_image:
        # Handle data URL format
        if "," in data.base64_image:
            data.base64_image = data.base64_image.split(",")[1]
        return base64.b64decode(data.base64_image)
    
    elif data.image_url:
        response = requests.get(data.image_url, timeout=15)
        if response.status_code != 200:
            raise HTTPException(
                status_code=400, 
                detail=f"Failed to download image. Status: {response.status_code}"
            )
        return response.content
    
    else:
        raise HTTPException(
            status_code=400, 
            detail="Must provide either base64_image or image_url"
        )


@app.post("/remove-bg")
async def remove_background_endpoint(data: ImageRequest):
    """
    Remove background from image.
    
    - **base64_image**: Base64 encoded image string
    - **image_url**: URL to image
    - **threshold**: Background detection threshold (200-255, default 240)
    """
    try:
        # Get image bytes
        image_bytes = get_image_bytes(data)
        
        # Open image
        input_image = Image.open(BytesIO(image_bytes))
        
        # Remove background
        output_image = simple_bg_remove(input_image, data.threshold)
        
        # Save with unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")
        output_image.save(file_path, format="PNG", optimize=True)
        
        # Build public URL
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN")
        railway_static = os.getenv("RAILWAY_STATIC_URL")
        
        if railway_domain:
            base_url = f"https://{railway_domain}"
        elif railway_static:
            base_url = railway_static
        else:
            base_url = "http://localhost:8000"
        
        public_url = f"{base_url}/output/{file_id}.png"
        
        return JSONResponse(content={
            "success": True,
            "message": "Background removed successfully",
            "image_url": public_url,
            "file_id": file_id,
            "tip": "Adjust 'threshold' (200-255) for better results"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Processing error: {str(e)}"
        )


@app.post("/upload")
async def upload_file(file: UploadFile = File(...), threshold: int = 240):
    """
    Upload an image file directly for background removal.
    """
    try:
        # Read uploaded file
        contents = await file.read()
        input_image = Image.open(BytesIO(contents))
        
        # Remove background
        output_image = simple_bg_remove(input_image, threshold)
        
        # Save
        file_id = str(uuid.uuid4())
        file_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")
        output_image.save(file_path, format="PNG", optimize=True)
        
        # Build URL
        railway_domain = os.getenv("RAILWAY_PUBLIC_DOMAIN", "localhost:8000")
        base_url = f"https://{railway_domain}" if not railway_domain.startswith("http") else railway_domain
        public_url = f"{base_url}/output/{file_id}.png"
        
        return JSONResponse(content={
            "success": True,
            "image_url": public_url,
            "file_id": file_id
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {
        "name": "Background Remover API",
        "version": "2.0",
        "status": "running",
        "endpoints": {
            "POST /remove-bg": "Remove background (JSON with base64_image or image_url)",
            "POST /upload": "Upload file directly",
            "GET /health": "Health check"
        },
        "example": {
            "base64_image": "data:image/png;base64,iVBORw0KG...",
            "image_url": "https://example.com/image.jpg",
            "threshold": 240
        }
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "output_dir_exists": os.path.exists(OUTPUT_DIR),
        "files_count": len(os.listdir(OUTPUT_DIR))
    }


@app.get("/output/{file_id}")
async def get_output_file(file_id: str):
    """Direct file access endpoint."""
    file_path = os.path.join(OUTPUT_DIR, file_id)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(file_path)
