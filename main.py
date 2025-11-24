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
from rembg import remove
from PIL import Image
import numpy as np

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create output folder if missing
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Mount static files
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

class ImageRequest(BaseModel):
    base64_image: str | None = None
    image_url: str | None = None


def process_image(image_bytes: bytes) -> Image.Image:
    """Remove background and return PIL image."""
    try:
        input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
        output_np = remove(np.array(input_image))
        return Image.fromarray(output_np)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Image processing failed: {str(e)}")


@app.post("/remove-bg")
async def remove_bg(data: ImageRequest):
    if not data.base64_image and not data.image_url:
        raise HTTPException(status_code=400, detail="Provide base64_image or image_url")

    try:
        # ----- BASE64 INPUT -----
        if data.base64_image:
            # Remove data URL prefix if present
            if "," in data.base64_image:
                data.base64_image = data.base64_image.split(",")[1]
            image_bytes = base64.b64decode(data.base64_image)

        # ----- IMAGE URL INPUT -----
        elif data.image_url:
            response = requests.get(data.image_url, timeout=10)
            if response.status_code != 200:
                raise HTTPException(status_code=400, detail="Failed to download image from URL")
            image_bytes = response.content

        # Process image
        output_image = process_image(image_bytes)

        # Create unique filename
        file_id = str(uuid.uuid4())
        file_path = os.path.join(OUTPUT_DIR, f"{file_id}.png")

        # Save final image
        output_image.save(file_path, format="PNG")

        # Get the base URL from environment or construct it
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
        "endpoints": {
            "POST /remove-bg": "Remove background from image",
            "GET /": "API info"
        }
    }


@app.get("/health")
def health():
    return {"status": "healthy"}
```

## 4. **Procfile** (for Railway deployment)
```
web: uvicorn main:app --host 0.0.0.0 --port $PORT