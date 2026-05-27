import io

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.responses import StreamingResponse
from ultralytics import YOLO
import cv2
import numpy as np

# 1. Initialize the FastAPI App
app = FastAPI(title="Touchless HCI Hand Tracker API", version="1.0")

# 2. Load the optimized ONNX model
print("Loading ONNX model into API memory...")
model = YOLO('best.onnx', task='detect')


# ──────────────────────────────────────────────
# HELPER: generate MJPEG frames from webcam
# ──────────────────────────────────────────────
def generate_webcam_frames(conf: float = 0.5):
    # Use cv2.CAP_DSHOW to prevent freezing on Windows
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Set dimensions for stream speed and stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            success, frame = cap.read()
            if not success:
                continue

            # Run YOLO inference
            results = model(frame, conf=conf, verbose=False)
            annotated_frame = results[0].plot()

            # Draw hands-detected counter
            num_hands = len(results[0].boxes)
            cv2.putText(
                annotated_frame,
                f"Hands Detected: {num_hands}",
                (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3
            )

            # Encode to JPEG
            _, buffer = cv2.imencode('.jpg', annotated_frame)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes() +
                b"\r\n"
            )
    finally:
        cap.release()


# ══════════════════════════════════════════════
# 1. Original Endpoints (Predict & Draw)
# ══════════════════════════════════════════════

@app.get("/")
def home():
    return {"message": "Touchless HCI Hand Detection API is live!"}


@app.post("/predict/")
async def predict_hand(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        results = model(img, verbose=False)
        boxes = results[0].boxes

        predictions = []
        for box in boxes:
            coords = box.xyxy[0].tolist()
            conf = box.conf[0].item()
            predictions.append({
                "class": "hand",
                "confidence": round(conf * 100, 2),
                "bounding_box": {
                    "x1": int(coords[0]),
                    "y1": int(coords[1]),
                    "x2": int(coords[2]),
                    "y2": int(coords[3])
                }
            })

        return {"hands_detected": len(predictions), "predictions": predictions}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/predict_and_draw/")
async def predict_and_draw(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        results = model(img, verbose=False)
        annotated_img = results[0].plot()

        _, encoded_img = cv2.imencode('.jpg', annotated_img)
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


# ══════════════════════════════════════════════
# 2. Clean New Endpoints without Parameters
# ══════════════════════════════════════════════

@app.get("/webcam", response_class=HTMLResponse)
def webcam_interface():
    """
    Display a quick link button for the live stream (without complex options)
    """
    button_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            .stream-btn {
                background-color: #00ff88;
                color: #111;
                font-size: 16px;
                font-weight: bold;
                padding: 12px 24px;
                border: none;
                border-radius: 6px;
                cursor: pointer;
                text-decoration: none;
                display: inline-block;
                transition: background 0.3s ease;
                font-family: sans-serif;
            }
            .stream-btn:hover {
                background-color: #00cc6e;
            }
        </style>
    </head>
    <body>
        <p style="color: #666; font-family: sans-serif;">Click the button below to open the live stream in a new window:</p>
        <a href="/webcam/stream" target="_blank" class="stream-btn">🖐 Open Live Stream</a>
    </body>
    </html>
    """
    return HTMLResponse(content=button_html)


@app.get("/webcam/stream", include_in_schema=False)
def webcam_stream():
    """
    The actual stream URL (hidden from Swagger UI to keep it clean)
    """
    try:
        return StreamingResponse(
            generate_webcam_frames(conf=0.5),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})