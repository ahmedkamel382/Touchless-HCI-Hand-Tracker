import io
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, HTMLResponse
from starlette.responses import StreamingResponse
from ultralytics import YOLO

# --- API and Model Initialization ---

app = FastAPI(title="Touchless HCI Hand Tracker API", version="1.0")

print("Loading ONNX model into API memory...")
model = YOLO('best.onnx', task='detect')


# --- REST API Endpoints ---

@app.get("/")
def home():
    """Simple health check endpoint to verify the server is running."""
    return {"message": "Touchless HCI Hand Detection API is live!"}


@app.post("/predict/")
async def predict_hand(file: UploadFile = File(...)):
    """
    Receives an uploaded image from the browser, runs YOLO, and returns
    the raw bounding box coordinates in JSON format.
    """
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
    """
    Standard visual debugging endpoint (useful for Swagger UI testing).
    """
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


# --- Client-Side Web Interface ---

@app.get("/webcam", response_class=HTMLResponse)
def webcam_interface():
    """
    Serves the advanced Frontend HTML/JS application.
    It accesses the viewer's local webcam and communicates with the /predict/ endpoint.
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Live Hand Tracking (Client-Side)</title>
        <style>
    body { 
        background-color: #111; 
        color: white; 
        font-family: sans-serif; 
        text-align: center; 
    }
    #video-container { 
        position: relative; 
        display: inline-block; 
        margin-top: 20px; 
    }
    /* ONLY mirror the video so it acts like a mirror */
    video { 
        border-radius: 8px; 
        transform: scaleX(-1); 
        background: #000;
    }
    /* Keep the canvas normal so text renders left-to-right */
    canvas { 
        position: absolute; 
        top: 0; 
        left: 0; 
        border-radius: 8px;
    }
    #status { margin-top: 15px; color: #00ff88; font-weight: bold; }
</style>
    </head>
    <body>
        <h2>🖐 Remote Touchless HCI Tracking</h2>
        <p>This securely uses your device's camera and streams frames to the Edge AI server.</p>

        <div id="video-container">
            <video id="video" width="640" height="480" autoplay playsinline></video>
            <canvas id="canvas" width="640" height="480"></canvas>
        </div>

        <p id="status">Waiting for camera permissions...</p>

        <script>
            const video = document.getElementById('video');
            const canvas = document.getElementById('canvas');
            const ctx = canvas.getContext('2d');
            const status = document.getElementById('status');

            // A hidden canvas just to take "photos" of the video feed
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = 640;
            captureCanvas.height = 480;
            const captureCtx = captureCanvas.getContext('2d');

            // 1. Turn on the Viewer's Camera
            async function startCamera() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
                    video.srcObject = stream;
                    status.innerText = "Camera active. Streaming data to AI Server...";

                    // Start the infinite processing loop once the video is playing
                    video.onloadeddata = () => processFrame();
                } catch (err) {
                    status.innerText = "Error accessing camera. Please allow permissions.";
                    console.error(err);
                }
            }

            // 2. The Main AI Loop
            function processFrame() {
                if (video.readyState !== video.HAVE_ENOUGH_DATA) {
                    requestAnimationFrame(processFrame);
                    return;
                }

                // Copy the current video frame to the hidden canvas
                captureCtx.drawImage(video, 0, 0, 640, 480);

                // Compress the image to a tiny JPEG to save network bandwidth
                captureCanvas.toBlob(async (blob) => {
                    const formData = new FormData();
                    formData.append('file', blob, 'frame.jpg');

                    try {
                        // Send the image to your FastAPI Python server
                        const response = await fetch('/predict/', {
                            method: 'POST',
                            body: formData
                        });

                        const data = await response.json();

                        // Draw the JSON coordinates on the screen
                        drawBoxes(data.predictions);
                    } catch (err) {
                        console.error("Inference Error:", err);
                    }

                    // Instantly trigger the next frame
                    requestAnimationFrame(processFrame);

                }, 'image/jpeg', 0.6); // 60% JPEG quality is the sweet spot for YOLO speed
            }

            // 3. Draw the Bounding Boxes
function drawBoxes(predictions) {
    // Clear the previous frame's boxes
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Setup the styling
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#00ff88";
    ctx.fillStyle = "#00ff88";
    ctx.font = "bold 18px sans-serif";

    if (predictions && predictions.length > 0) {
        predictions.forEach(p => {
            const box = p.bounding_box;
            const width = box.x2 - box.x1;
            const height = box.y2 - box.y1;
            
            // MATH FIX: Flip the X coordinate so the box aligns with the CSS-flipped video
            // We subtract the right edge (x2) from the canvas width to find the new left edge
            const mirroredX = canvas.width - box.x2;
            
            // Draw the rectangle using the mirrored X
            ctx.strokeRect(mirroredX, box.y1, width, height);
            
            // Draw the text background and label normally (left-to-right)
            ctx.fillRect(mirroredX, box.y1 - 25, 120, 25);
            ctx.fillStyle = "#111"; // Make text dark for contrast
            ctx.fillText(`Hand ${p.confidence}%`, mirroredX + 5, box.y1 - 7);
            ctx.fillStyle = "#00ff88"; // Reset for the next box
        });
    }
}

            // Initialize
            startCamera();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)