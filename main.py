import base64
import uuid
import os
import requests
from io import BytesIO
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from rembg import remove
from PIL import Image
import numpy as np

app = FastAPI()

# Create output folder if missing
os.makedirs("output", exist_ok=True)

class ImageRequest(BaseModel):
    base64_image: str | None = None
    image_url: str | None = None


def process_image(image_bytes: bytes):
    """Remove background and return PIL image."""
    input_image = Image.open(BytesIO(image_bytes)).convert("RGBA")
    output_np = remove(np.array(input_image))
    return Image.fromarray(output_np)


@app.post("/remove-bg")
async def remove_bg(data: ImageRequest):

    if not data.base64_image and not data.image_url:
        return {"error": "Provide base64_image or image_url"}

    try:
        # ----- BASE64 INPUT -----
        if data.base64_image:
            image_bytes = base64.b64decode(data.base64_image)

        # ----- IMAGE URL INPUT -----
        elif data.image_url:
            response = requests.get(data.image_url)
            if response.status_code != 200:
                return {"error": "Failed to download image from URL"}
            image_bytes = response.content

        # Process image
        output_image = process_image(image_bytes)

        # Create unique filename
        file_id = str(uuid.uuid4())
        file_path = f"output/{file_id}.png"

        # Save final image
        output_image.save(file_path, format="PNG")

        # Public URL (Railway auto-exposes /output if static enabled)
        public_url = f"https://YOUR-RAILWAY-DOMAIN/output/{file_id}.png"

        return {"image_url": public_url}

    except Exception as e:
        return {"error": str(e)}


@app.get("/")
def home():
    return {"message": "Background Remover API is running"}
