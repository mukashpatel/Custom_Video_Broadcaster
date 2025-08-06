import cv2
from typing import Optional, List, Dict
import pyvirtualcam
from engine import CustomerSegmentationWithYOLO
import torch


class Streaming(CustomerSegmentationWithYOLO):
    def __init__(
        self,
        in_source: Optional[int] = None,
        out_source: Optional[str] = None,
        fps: Optional[int] = None,
        blur_strength: Optional[int] = None,
        cam_fps: int = 15,
        background: str = "none"
    ):
        super().__init__()
        self.input_source = in_source
        self.out_source = out_source
        self.fps = fps
        self.blur_strength = blur_strength
        self.background = background
        self.running = False
        self.original_fps = fps

        self.device = (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if torch.backends.mps.is_available()
            else torch.device("cpu")
        )


    def update_stream_config(
        self,
        in_source: Optional[int] = None,
        out_source: Optional[str] = None,
        fps: Optional[int] = None,
        blur_strength: Optional[int] = None,
        background: Optional[str] = None
    ):
        """Update the streaming configuration dynamically."""
        if in_source is not None:
            self.input_source = in_source
        if out_source is not None:
            self.out_source = out_source
        if fps is not None:
            self.fps = fps
        if blur_strength is not None:
            self.blur_strength = blur_strength
        if background is not None:
            self.background = background

    def list_available_devices(self) -> List[Dict[str, str]]:
        """Detect and return available video input devices."""
        devices = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                devices.append({"id": str(i), "name": f"Camera ({i})"})
                cap.release()
        return devices
    

    def update_running_status(self, running_status=False):
        self.running = running_status
        return {"message": "Streaming stopped" if not running_status else "Streaming started"}

    def stream_video(self):
        self.running = True
        cap = cv2.VideoCapture(int(self.input_source))

        frame_idx = 0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            self.original_fps = int(cap.get(cv2.CAP_PROP_FPS))
        except Exception as e:
            print(f"Webcam({self.input_source}), live fps not available. Set the fps accordingly. Exception: {e}")
            self.original_fps = 30  # fallback fps

        if self.fps:
            if self.fps > self.original_fps:
                self.fps = self.original_fps
                frame_interval = int(self.original_fps / self.fps)
            else:
                frame_interval = int(self.original_fps / self.fps)
        else:
            frame_interval = 1

        with pyvirtualcam.Camera(width=width, height=height, fps=self.fps or self.original_fps) as cam:
            print(f"virtual camera running at {width}x{height} at {self.fps or self.original_fps} fps")

            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_interval == 0:
                    results = self.model.predict(source=frame, save=False, stream=True, verbose=False, device=self.device)
                    mask = self.generate_mask_from_result(results)

                    if mask is not None:
                        if self.background == "blur":
                            result_frame = self.apply_blur_with_mask(frame, mask, blur_strength=self.blur_strength)
                        elif self.background == "none":
                            result_frame = self.apply_black_background(frame, mask)
                        elif self.background == "default":
                            result_frame = self.apply_custom_background(frame, mask)
                        elif self.background == "virtual":
                            result_frame = self.apply_virtual_background(frame, mask)
                        else:
                            result_frame = frame  # fallback
                    else:
                        result_frame = frame  # no mask available, use raw frame

                    # Send to virtual camera
                    cam.send(cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB))
                    cam.sleep_until_next_frame()

                frame_idx += 1

        cap.release()






