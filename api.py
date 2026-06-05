import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.responses import StreamingResponse
from ultralytics import YOLO

# --- API and Model Initialization ---

# Initialize the FastAPI server instance
app = FastAPI(title="Touchless HCI Hand Tracker API", version="1.0")

# Load the optimized ONNX model into memory exactly once when the server starts.
# This prevents the system from having to reload the heavy weights on every individual request.
print("Loading ONNX model into API memory...")
model = YOLO('best.onnx', task='detect')


# --- Streaming Helper Function ---

def generate_webcam_frames(conf: float = 0.5):
    """
    Continuously captures frames from the local webcam, runs the YOLO inference,
    and yields an MJPEG (Motion JPEG) stream to be displayed in the browser.
    """
    # Initialize standard webcam capture (device 0 is usually the built-in webcam)
    cap = cv2.VideoCapture(0)

    # Check if the camera hardware is successfully accessible
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    # Lock the resolution to 640x480. Lower resolution ensures higher FPS
    # and stable stream latency for real-time tracking on edge hardware.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            # Read a single frame from the camera
            success, frame = cap.read()
            if not success:
                continue  # Skip this loop iteration if the frame drops

            # Run the frame through the ONNX YOLO model.
            # conf=0.5 means it only accepts detections with a 50%+ confidence score.
            results = model(frame, conf=conf, verbose=False)

            # The .plot() method automatically draws the YOLO bounding boxes onto the frame
            annotated_frame = results[0].plot()

            # Count how many hands are currently on the screen
            num_hands = len(results[0].boxes)

            # Draw the active hand counter in the top-left corner
            cv2.putText(
                annotated_frame,
                f"Hands Detected: {num_hands}",
                (10, 35),  # (x, y) coordinates for the text
                cv2.FONT_HERSHEY_SIMPLEX,  # Font style
                1.1,  # Font scale
                (0, 255, 0),  # Color in BGR format (Green)
                3  # Line thickness
            )

            # Compress the raw numpy array into a JPEG format for web transmission
            _, buffer = cv2.imencode('.jpg', annotated_frame)

            # Yield the frame in the multipart format required by web browsers for live video streaming
            yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )
    finally:
        # Always release the camera hardware when the stream stops to prevent system freezing
        cap.release()


# --- REST API Endpoints ---

@app.get("/")
def home():
    """Simple health check endpoint to verify the server is running."""
    return {"message": "Touchless HCI Hand Detection API is live!"}


@app.post("/predict/")
async def predict_hand(file: UploadFile = File(...)):
    """
    Receives an uploaded image, runs YOLO, and returns the raw bounding box
    coordinates in JSON format. Ideal for machine-to-machine communication (like AR/VR apps).
    """
    try:
        # Read the uploaded image file into raw memory bytes
        image_bytes = await file.read()

        # Convert the raw bytes into a numpy array, then decode it into an OpenCV color image
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        # Safety check to ensure the uploaded file was actually a valid image
        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        # Run inference (verbose=False keeps the terminal clean)
        results = model(img, verbose=False)
        boxes = results[0].boxes

        # Format the output data
        predictions = []
        for box in boxes:
            # Extract spatial coordinates [x_min, y_min, x_max, y_max]
            coords = box.xyxy[0].tolist()

            # Extract the confidence score (e.g., 0.95)
            conf = box.conf[0].item()

            # Append to our JSON dictionary
            predictions.append({
                "class": "hand",
                "confidence": round(conf * 100, 2),  # Convert to percentage
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
    """
    Receives an uploaded image, physically draws the bounding boxes directly on it,
    and sends the modified JPEG file back to the user. Useful for visual debugging in Swagger UI.
    """
    try:
        # Decode the uploaded image
        image_bytes = await file.read()
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"error": "Invalid image file format."})

        # Run inference and immediately draw the boxes on the image array
        results = model(img, verbose=False)
        annotated_img = results[0].plot()

        # Encode the drawn image back into JPEG format
        _, encoded_img = cv2.imencode('.jpg', annotated_img)

        # Stream the file back to the client as an image file
        return StreamingResponse(io.BytesIO(encoded_img.tobytes()), media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/webcam", response_class=HTMLResponse)
def webcam_interface():
    """
    Serves a lightweight HTML webpage with a UI button that triggers the live stream.
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
            .stream-btn:hover { background-color: #00cc6e; }
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
    The actual endpoint that handles the MJPEG video feed.
    'include_in_schema=False' hides this from the /docs Swagger UI to prevent clutter.
    """
    try:
        # Stream the continuous frame generator back to the browser
        return StreamingResponse(
            generate_webcam_frames(conf=0.5),
            media_type="multipart/x-mixed-replace; boundary=frame"
        )
    except RuntimeError as e:
        return JSONResponse(status_code=500, content={"error": str(e)})