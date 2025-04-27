from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import Response
from PIL import Image
import io
from ultralytics import FastSAM
from ultralytics.models.fastsam import FastSAMPredictor
import numpy as np

app = FastAPI(title="FastSAM Image Segmentation API")

# Initialize the FastSAM predictor
MODEL_PATH = "../weights/fastsam.pt"
try:
    # Check if model file exists
    import os
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        predictor = None
    else:
        # Create FastSAMPredictor with configuration
        overrides = dict(
            conf=0.4,
            iou=0.9,
            task="segment",
            mode="predict",
            model=MODEL_PATH,
            save=False,
            imgsz=1024,
            retina_masks=True
        )
        predictor = FastSAMPredictor(overrides=overrides)
except Exception as e:
    print("Error loading FastSAM model:")
    print(f"Error details: {e}")
    predictor = None


def create_segmented_output(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create a transparent PNG with only the segmented object.
    The white part in the mask represents the object we want to keep."""

    # Ensure mask is binary (0 or 1)
    binary_mask = mask.astype(np.uint8)

    # No need to invert since white (1) already represents the object we want to keep
    # Convert to 255 scale for alpha
    alpha_mask = binary_mask * 255

    # Create alpha channel
    alpha = Image.fromarray(alpha_mask)

    # Convert original image to RGBA
    image_rgba = image.convert('RGBA')

    # Apply the mask as alpha channel
    image_rgba.putalpha(alpha)

    return image_rgba


@app.post("/segment")
async def segment_image(
    file: UploadFile,
    x_center: float = Form(...),  # Normalized center x (0-1)
    y_center: float = Form(...),  # Normalized center y (0-1)
    width: float = Form(...),     # Normalized width (0-1)
    height: float = Form(...)     # Normalized height (0-1)
):
    if not predictor:
        print("Predictor not working")
        raise HTTPException(status_code=500, detail="Model not loaded")

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Read image from request
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        # Convert to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get image dimensions
        img_width, img_height = image.size

        # Convert YOLOv8 normalized coordinates to absolute coordinates
        # YOLO format: [x_center, y_center, width, height] (normalized 0-1)
        # Convert to [x1, y1, x2, y2] format
        x1 = (x_center - width/2) * img_width
        y1 = (y_center - height/2) * img_height
        x2 = (x_center + width/2) * img_width
        y2 = (y_center + height/2) * img_height

        # Create bounding box in format [x1, y1, x2, y2]
        bbox = [[x1, y1, x2, y2]]

        # First get everything results
        everything_results = predictor(image)

        # Then use bbox prompt
        prompt_results = predictor.prompt(everything_results, bboxes=bbox)

        # Get the first result's masks
        if not prompt_results or len(prompt_results) == 0:
            raise HTTPException(
                status_code=404, detail="No object detected in the specified bounding box")

        result = prompt_results[0]
        if not hasattr(result, 'masks') or result.masks is None or len(result.masks.data) == 0:
            raise HTTPException(
                status_code=404, detail="No masks found in the detection results")

        # Get the first mask (most confident one for the bbox)
        mask = result.masks.data[0].cpu().numpy()

        # Create segmented output
        result_image = create_segmented_output(image, mask)

        # Save to bytes
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        return Response(content=img_byte_arr, media_type="image/png")

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
