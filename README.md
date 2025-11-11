# Local backend for YOLOv8 inference (FastAPI)

This workspace contains a simple FastAPI backend that loads a YOLOv8 model (`yolov8n.pt`) and an updated `main.html` with a file upload UI that calls the backend and displays an annotated image.

Quick steps (Windows, cmd.exe):

1. Create and activate a virtual environment (recommended):

```cmd
python -m venv .venv
.venv\Scripts\activate
```

2. Install dependencies. Note: `ultralytics` requires `torch`. If you don't have torch installed, install a compatible wheel from PyTorch first (CPU or GPU build):

```cmd
pip install -r requirements.txt
# If ultralytics fails because torch isn't present, install torch first. Example (CPU):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install ultralytics
```

3. Run the backend:

```cmd
python app.py
```

The backend runs on http://127.0.0.1:8000 by default.

4. Serve the frontend (important to avoid file:// CORS issues). From the same folder run a simple HTTP server on port 8001:

```cmd
# from the project folder
python -m http.server 8001
```

Then open the frontend in your browser:

http://127.0.0.1:8001/main.html

5. Use the "Choose Image" control on the right to upload an image and click "Upload & Predict". The page will POST the image to the backend and display the annotated image and predictions.

Notes & tips:

- The backend uses `ultralytics.YOLO` to load `yolov8n.pt`. Make sure `yolov8n.pt` is in the same folder as `app.py`.
- If you're using GPU, install the GPU build of `torch` before installing `ultralytics`.
- For production, consider using streaming endpoints, batching, Docker, or a task queue for large loads.
