from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import StreamingResponse
from typing import Optional
from rembg import remove
from PIL import Image
import io
import base64
import requests
import uvicorn

app = FastAPI(title="Background Removal API", version="3.0.0")

@app.post("/remove-background")
async def remove_background(
    url: Optional[str] = Form(None),
    base64_image: Optional[str] = Form(None)
):
    """
    Remove background from:
    - Image URL
    - Base64 image string
    """

    # Validate single input
    if not url and not base64_image:
        raise HTTPException(status_code=400, detail="Provide url OR base64_image")

    if url and base64_image:
        raise HTTPException(status_code=400, detail="Send ONLY ONE: url OR base64_image")

    try:
        # -------------------------------------------------------
        # Process URL input
        # -------------------------------------------------------
        if url:
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            if not response.headers.get("content-type", "").startswith("image/"):
                raise HTTPException(status_code=400, detail="URL must point to an image")

            input_image = Image.open(io.BytesIO(response.content))

            # Extract filename
            name = url.split("/")[-1].split("?")[0]
            filename = (name.rsplit(".", 1)[0] if "." in name else "image") + ".png"

        # -------------------------------------------------------
        # Process Base64 input
        # -------------------------------------------------------
        elif base64_image:
            # Remove prefix like "data:image/png;base64,"
            if "," in base64_image:
                base64_image = base64_image.split(",")[1]

            try:
                decoded = base64.b64decode(base64_image)
                input_image = Image.open(io.BytesIO(decoded))
                filename = "image.png"
            except:
                raise HTTPException(status_code=400, detail="Invalid base64 image")

        # -------------------------------------------------------
        # REMOVE BACKGROUND
        # -------------------------------------------------------
        output = remove(input_image)

        buffer = io.BytesIO()
        output.save(buffer, format="PNG")
        buffer.seek(0)

        return StreamingResponse(
            buffer,
            media_type="image/png",
            headers={
                "Content-Disposition": f"attachment; filename=removed_bg_{filename}"
            }
        )

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"URL fetch error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
