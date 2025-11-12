# Construction Site Safety – Full Stack (FastAPI + YOLOv8 + WebSocket + DB)

This project provides a local safety monitoring system with live camera ingest, real‑time detection overlays, event storage, zones/rules, alerts, RBAC auth, analytics, and optional Docker deployment.

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

5. Use the Upload & Predict, or enter a camera source (0 or RTSP/HTTP URL) and click Start to see live annotated frames. Use Live Cameras (Dynamic) to add multiple sources.

Features
- Live RTSP/USB streaming via WebSocket, multi‑camera tiles
- YOLOv8 inference and annotated frames
- Tracking with ByteTrack (fallback to IOU)
- Events persisted (SQLite by default, PostgreSQL if POSTGRES_DSN set)
- Zones (no_go/info) with dwell‑time rule, PPE rules (helmet/vest) using configured class IDs
- Alerts via webhook/email/SMS (Twilio) with rate limiting and escalation
- RBAC auth (JWT), users & roles (admin/operator), audit logs
- Analytics endpoints, CSV/PDF reports
- Privacy: face blurring, data retention
- Prometheus metrics (/metrics), optional Docker Compose with Postgres/Prometheus/Grafana
- Model mgmt: configurable confidence threshold; health endpoint; hot‑reload weights

Environment (optional)
- API_KEY: legacy API key (not required for JWT flow)
- JWT_SECRET: secret for signing tokens
- POSTGRES_DSN: e.g. postgresql://user:pass@host:5432/safetydb
- ALERT_WEBHOOK_URL: URL to send JSON alerts
- SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/ALERT_EMAIL_TO: email alerts
- TWILIO_ACCOUNT_SID/TWILIO_AUTH_TOKEN/TWILIO_FROM/TWILIO_TO/TWILIO_ESCALATE_TO: SMS alerts
- ADMIN_PASSWORD: default admin user password (username: admin)

Docker (optional)

```bash
docker compose up --build
```

Services:
- api: FastAPI (http://localhost:8000)
- frontend: static HTML (http://localhost:8001/main.html)
- postgres: database (default: postgres/postgres, db safetydb)
- prometheus: metrics (http://localhost:9090)
- grafana: dashboards (http://localhost:3000)

Key endpoints (selection)
- POST /predict (image upload -> annotated image + predictions)
- WS /ws (send {"source":"0"|"rtsp://..."} -> base64 jpeg frames)
- GET /events (list latest events)
- GET /zones, POST /zones, DELETE /zones/{id} (admin JWT required for mutations)
- GET /config, POST /config (admin JWT required for update)
- GET /model (model status), POST /model/reload (admin; {"weights":"path/to/best.pt"})
- GET /health (liveness/status)
- POST /auth/login (returns JWT), POST /auth/register (admin)
- GET /analytics/summary, GET /reports/events.csv, GET /reports/summary.pdf

Notes
- Default model is `yolov8n.pt` (COCO). PPE helmet/vest rules require a PPE‑trained model; set numeric class IDs via POST /config:
  {"ppe_classes":{"person":0,"helmet":<id>,"vest":<id>},"rules":{"require_helmet":true,"require_vest":true}}
  You can still demo other rules (zones/dwell) with COCO.

Security & licensing
- YOLOv8 is AGPL‑3.0. For closed‑source/commercial usage consider a commercial license.
- JWT auth is provided for basic RBAC. Rotate secrets and configure TLS in production.

Production accuracy workflow (summary)
1) Collect and label PPE data (helmet, vest, gloves, boots) across your sites. Ensure varied lighting, angles, PPE colors, motion blur, occlusion. Split into train/val/test.
2) Train:
   ```bash
   # dataset.yaml in Ultralytics format with train/val/test image lists and class names
   python train_ppe.py --data path/to/dataset.yaml --model yolov8m.pt --epochs 150 --imgsz 640 --batch 32 --project runs/ppe --name exp1
   ```
   Track mAP, precision/recall. Tune anchors, LR, augmentation; consider larger backbones (yolov8m/l) if needed.
3) Evaluate on held‑out test set; perform ablations (image size, conf threshold). Target mAP@50 and class‑wise PR curves that meet your SLA.
4) Deploy the best weights:
   ```bash
   # from project root
   curl -X POST http://127.0.0.1:8000/model/reload -H "Authorization: Bearer <ADMIN_TOKEN>" -H "Content-Type: application/json" -d "{\"weights\":\"runs/ppe/exp1/weights/best.pt\"}"
   ```
   Set class IDs in /config and thresholds:
   ```bash
   curl -X POST http://127.0.0.1:8000/config -H "Authorization: Bearer <ADMIN_TOKEN>" -H "Content-Type: application/json" -d "{\"ppe_classes\":{\"person\":0,\"helmet\":1,\"vest\":2},\"rules\":{\"require_helmet\":true,\"require_vest\":true},\"thresholds\":{\"conf\":0.6}}"
   ```
5) Monitor metrics via /analytics/summary and Prometheus; iterate on data collection where errors concentrate.

Notes & tips:

- The backend uses `ultralytics.YOLO` to load `yolov8n.pt`. Make sure `yolov8n.pt` is in the same folder as `app.py`.
- If you're using GPU, install the GPU build of `torch` before installing `ultralytics`.
- In Docker, GPU acceleration and TensorRT/Jetson are optional and not enabled by default.
