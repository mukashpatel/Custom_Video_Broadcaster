import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from stream_utils import Streaming
import threading
import cv2
import os

app = FastAPI()

# Mount static files (e.g., your HTML + assets)
app.mount("/static", StaticFiles(directory="static"), name="static")
stream_thread =None
# Global instance of your streamer
streaming = Streaming()


@app.get("/")
def serve_ui():
    #  return FileResponse("static/3d_background.html")
     return FileResponse("static/index.html")


@app.get("/start")
def start_stream(
    source: str = Query("0"),
    fps: int = Query(15),
    blur_strength: int = Query(21),
    background: str = Query("none")
):
    global stream_thread
    if streaming.running:
        return JSONResponse(content={"message": "Stream already running"}, status_code=400)

    # Safely parse source as int if possible
    try:
        source_val = int(source)
    except ValueError:
        source_val = source

    # Update config with query params
    streaming.update_stream_config(
        in_source=source_val,
        fps=fps,
        blur_strength=blur_strength,
        background=background
    )

    # Start stream in background thread
    stream_thread = threading.Thread(target=streaming.stream_video, args=())
    stream_thread.start()

    return {
        "message": f"Stream started with source: {source}, fps: {fps}, blur_strength: {blur_strength}, background: {background}"
    }



@app.get("/stop")
def stop_stream():
    return streaming.update_running_status()


@app.get("/devices")
def devices():
    return streaming.list_available_devices()


@app.get("/3d-background")
async def get_3d_background():
    return FileResponse("static/3d_background.html")

@app.get("/favicon.ico")
async def favicon():
    favicon_path = os.path.join("static", "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return FileResponse("static/IMG-20250705-WA0004.jpg")  # fallback or return 204/empty if you prefer


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
