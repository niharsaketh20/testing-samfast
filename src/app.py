from fastapi import FastAPI, UploadFile, HTTPException, Form
from fastapi.responses import Response
from PIL import Image
import io
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

app = FastAPI(title="SegFormer Image Segmentation API")

# Initialize the SegFormer model and processor
MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
try:
    processor = SegformerImageProcessor.from_pretrained(MODEL_NAME)
    model = SegformerForSemanticSegmentation.from_pretrained(MODEL_NAME)
    if torch.cuda.is_available():
        model = model.to("cuda")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model loaded successfully on {device}")
except Exception as e:
    print("Error loading SegFormer model:")
    print(f"Error details: {e}")
    processor = None
    model = None


def create_segmented_output(image: Image.Image, mask: np.ndarray) -> Image.Image:
    """Create a transparent PNG with only the segmented object.
    The mask represents the segmentation map where each pixel value corresponds to a class."""

    # Convert mask to binary (foreground vs background)
    # Assuming 0 is background class, everything else is foreground
    binary_mask = (mask > 0).astype(np.uint8)

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
    if not processor or not model:
        print("Model not loaded")
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

        # Convert normalized coordinates to absolute coordinates
        x1 = int((x_center - width/2) * img_width)
        y1 = int((y_center - height/2) * img_height)
        x2 = int((x_center + width/2) * img_width)
        y2 = int((y_center + height/2) * img_height)

        # Crop the image to the region of interest
        roi_image = image.crop((x1, y1, x2, y2))

        # Prepare the image for the model
        inputs = processor(images=roi_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        # Get the predicted segmentation mask
        upsampled_logits = torch.nn.functional.interpolate(
            logits,
            size=roi_image.size[::-1],  # (height, width)
            mode="bilinear",
            align_corners=False,
        )
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        # Create a full-size mask with the ROI segmentation
        full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = pred_mask

        # Create segmented output
        result_image = create_segmented_output(image, full_mask)

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
