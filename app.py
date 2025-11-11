import io
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np

app = FastAPI()

# Allow the frontend (served locally) to call this API. Adjust origins if you serve frontend on another port.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:8001", "http://localhost:8001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def load_model():
    # Load the YOLOv8 model once at startup.
    # This uses the ultralytics package. Ensure `yolov8n.pt` is in the same folder as this file.
    global model
    try:
        from ultralytics import YOLO

        model = YOLO("yolov8n.pt")
        print("Model loaded: yolov8n.pt")
    except Exception as e:
        print("Failed to load ultralytics YOLO model:", e)
        model = None


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an uploaded image, runs YOLO inference and returns a JSON with a base64 PNG of the annotated image and the raw predictions."""
    if model is None:
        return JSONResponse({"error": "Model not loaded on server."}, status_code=500)

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return JSONResponse({"error": f"Unable to read image: {e}"}, status_code=400)

    arr = np.array(image)

    # Run inference. Pass the numpy array (H,W,3)
    results = model(arr)
    if len(results) == 0:
        return {"image": None, "predictions": []}

    r = results[0]

    # Annotated image as numpy array
    try:
        annotated = r.plot()  # ultralytics result.plot()
    except Exception:
        # fallback: return original image
        annotated = arr

    # Convert annotated (numpy) to PNG bytes
    if annotated.dtype != np.uint8:
        annotated = (annotated * 255).astype(np.uint8)

    annotated_pil = Image.fromarray(annotated)
    buf = io.BytesIO()
    annotated_pil.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")

    # Extract simple box predictions if available
    preds = []
    try:
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            xyxy = getattr(boxes, "xyxy", None)
            conf = getattr(boxes, "conf", None)
            cls = getattr(boxes, "cls", None)
            # xyxy, conf, cls may be tensors; convert to lists
            if xyxy is not None:
                xyxy_list = xyxy.cpu().numpy().tolist()
            else:
                xyxy_list = []
            if conf is not None:
                conf_list = conf.cpu().numpy().tolist()
            else:
                conf_list = []
            if cls is not None:
                cls_list = cls.cpu().numpy().tolist()
            else:
                cls_list = []

            for i, box in enumerate(xyxy_list):
                c = int(cls_list[i]) if i < len(cls_list) else None
                cf = float(conf_list[i]) if i < len(conf_list) else None
                preds.append({"xyxy": box, "class": c, "conf": cf})
    except Exception:
        preds = []

    return {"image": img_b64, "predictions": preds}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
