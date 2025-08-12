import cv2
import numpy as np
from ultralytics import YOLO
import threading
import time
import logging
from typing import Optional, Tuple, Dict, Any
import pyvirtualcam
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import base64
import os
import json
from PIL import Image, ImageFilter
import subprocess
import psutil

logger = logging.getLogger(__name__)

class VideoEngine:
    def __init__(self):
        self.cap: Optional[cv2.VideoCapture] = None
        self.virtual_cam: Optional[pyvirtualcam.Camera] = None
        self.model = None
        self.is_streaming = False
        self.current_frame = None
        self.background_frame = None
        self.device_id = 0
        
        # 3D Background settings
        self.use_3d_background = False
        self.current_scene = "corporate"
        self.scene_settings = {"speed": 1.0, "particles": 200, "intensity": 1.0}
        self.driver: Optional[webdriver.Chrome] = None
        self.background_thread: Optional[threading.Thread] = None
        self.frame_lock = threading.Lock()
        
        # Background effect settings
        self.effect_type = "blur"  # blur, black, image, 3d
        self.background_image = None
        self.blur_strength = 15
        
        # Performance settings
        self.target_fps = 30
        self.frame_skip = 0
        self.processing_resolution = (640, 480)
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model for person segmentation"""
        try:
            self.model = YOLO('yolov8n-seg.pt')
            logger.info("YOLO model initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YOLO model: {e}")
    
    def get_available_devices(self) -> list:
        """Get list of available camera devices"""
        devices = []
        for i in range(10):  # Check first 10 devices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    devices.append({
                        "id": i,
                        "name": f"Camera {i}",
                        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                    })
                cap.release()
        return devices
    
    def select_device(self, device_id: int) -> bool:
        """Select camera device"""
        try:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(device_id)
            if self.cap.isOpened():
                # Set optimal settings
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                self.device_id = device_id
                logger.info(f"Selected camera device {device_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to select device {device_id}: {e}")
            return False
    
    def apply_3d_background(self, frame, person_mask):
    """Apply 3D animated background to video frame"""
    try:
        h, w = frame.shape[:2]
        
        # Generate 3D background
        bg_3d = bg_3d_generator.generate_background(w, h)
        
        # Apply background where person is not detected
        result = frame.copy()
        result[person_mask == 0] = bg_3d[person_mask == 0]
        
        return result
    except Exception as e:
        print(f"Error applying 3D background: {e}")
        return frame
        
    def _setup_3d_background(self):
        """Setup headless browser for 3D background rendering"""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-extensions")
            chrome_options.add_argument("--disable-plugins")
            chrome_options.add_argument("--disable-images")
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            
            self.driver = webdriver.Chrome(options=chrome_options)
            
            # Load the 3D background page
            background_url = "http://localhost:8002/3d-background"
            self.driver.get(background_url)
            
            # Wait for page to load
            time.sleep(3)
            
            # Set initial scene and settings
            self._update_3d_scene()
            
            logger.info("3D background setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup 3D background: {e}")
            return False
    
    def _update_3d_scene(self):
        """Update 3D scene settings in the browser"""
        if not self.driver:
            return
        
        try:
            # Execute JavaScript to update scene
            script = f"""
            if (typeof updateScene === 'function') {{
                updateScene('{self.current_scene}', {json.dumps(self.scene_settings)});
            }}
            """
            self.driver.execute_script(script)
        except Exception as e:
            logger.error(f"Failed to update 3D scene: {e}")
    
    def _capture_3d_background(self) -> Optional[np.ndarray]:
        """Capture 3D background frame from browser"""
        if not self.driver:
            return None
        
        try:
            # Get screenshot as base64
            screenshot_b64 = self.driver.get_screenshot_as_base64()
            
            # Decode and convert to OpenCV format
            screenshot_data = base64.b64decode(screenshot_b64)
            nparr = np.frombuffer(screenshot_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            return frame
            
        except Exception as e:
            logger.error(f"Failed to capture 3D background: {e}")
            return None
    
    def _generate_procedural_background(self, scene_name: str, width: int, height: int) -> np.ndarray:
        """Generate procedural background when browser method fails"""
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        if scene_name == "corporate":
            # Blue gradient with grid
            for y in range(height):
                intensity = int(255 * (y / height) * 0.3)
                frame[y, :] = [intensity + 30, intensity + 60, intensity + 120]
            
            # Add grid lines
            for i in range(0, width, 50):
                cv2.line(frame, (i, 0), (i, height), (100, 100, 150), 1)
            for i in range(0, height, 50):
                cv2.line(frame, (0, i), (width, i), (100, 100, 150), 1)
        
        elif scene_name == "cyber":
            # Matrix-style background
            frame[:] = [0, 20, 0]  # Dark green base
            
            # Add random "code" lines
            for i in range(0, width, 20):
                for j in range(0, height, 10):
                    if np.random.random() > 0.95:
                        cv2.circle(frame, (i, j), 2, (0, 255, 100), -1)
        
        elif scene_name == "space":
            # Starfield
            frame[:] = [5, 5, 20]  # Dark blue base
            
            # Add stars
            for _ in range(200):
                x, y = np.random.randint(0, width), np.random.randint(0, height)
                brightness = np.random.randint(100, 255)
                cv2.circle(frame, (x, y), 1, (brightness, brightness, brightness), -1)
        
        elif scene_name == "waves":
            # Ocean waves
            for y in range(height):
                wave = int(30 * np.sin(2 * np.pi * y / 100))
                for x in range(width):
                    blue_intensity = 100 + wave + int(50 * np.sin(2 * np.pi * x / 200))
                    frame[y, x] = [blue_intensity, blue_intensity//2, 30]
        
        else:
            # Default gradient
            for y in range(height):
                intensity = int(255 * (y / height) * 0.5)
                frame[y, :] = [intensity, intensity//2, intensity//3]
        
        return frame
    
    def _segment_person(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Segment person from frame using YOLO"""
        if self.model is None:
            return None
        
        try:
            results = self.model(frame, classes=[0], verbose=False)  # class 0 is person
            
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'masks') and result.masks is not None:
                    # Get the largest mask (main person)
                    masks = result.masks.data.cpu().numpy()
                    if len(masks) > 0:
                        # Combine all person masks
                        combined_mask = np.any(masks, axis=0).astype(np.uint8) * 255
                        
                        # Resize mask to match frame
                        mask_resized = cv2.resize(combined_mask, (frame.shape[1], frame.shape[0]))
                        
                        # Apply morphological operations to smooth mask
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        mask_resized = cv2.morphologyEx(mask_resized, cv2.MORPH_CLOSE, kernel)
                        mask_resized = cv2.GaussianBlur(mask_resized, (5, 5), 0)
                        
                        return mask_resized
            
            return None
        except Exception as e:
            logger.error(f"Person segmentation failed: {e}")
            return None
    
    def _apply_background_effect(self, frame: np.ndarray) -> np.ndarray:
        """Apply selected background effect"""
        mask = self._segment_person(frame)
        
        if mask is None:
            return frame
        
        # Create background
        if self.effect_type == "3d" and self.use_3d_background:
            background = self._capture_3d_background()
            if background is None:
                background = self._generate_procedural_background(
                    self.current_scene, frame.shape[1], frame.shape[0]
                )
        elif self.effect_type == "blur":
            background = cv2.GaussianBlur(frame, (self.blur_strength, self.blur_strength), 0)
        elif self.effect_type == "black":
            background = np.zeros_like(frame)
        elif self.effect_type == "image" and self.background_image is not None:
            background = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))
        else:
            background = self._generate_procedural_background(
                "corporate", frame.shape[1], frame.shape[0]
            )
        
        # Ensure background matches frame size
        if background.shape[:2] != frame.shape[:2]:
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
        
        # Create 3-channel mask
        mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR) / 255.0
        
        # Composite person over background
        result = (frame * mask_3ch + background * (1 - mask_3ch)).astype(np.uint8)
        
        return result
    
    def configure_background(self, effect_type: str, background_image: Optional[str] = None,
                           blur_strength: int = 15, use_3d_background: bool = False,
                           scene_name: str = "corporate"):
        """Configure background settings"""
        self.effect_type = effect_type
        self.blur_strength = blur_strength
        self.use_3d_background = use_3d_background
        self.current_scene = scene_name
        
        if background_image and os.path.exists(background_image):
            self.background_image = cv2.imread(background_image)
        
        if use_3d_background and not self.driver:
            self._setup_3d_background()
        elif use_3d_background:
            self._update_3d_scene()
    
    def start_streaming(self, use_3d_background: bool = False, 
                       scene_name: str = "corporate",
                       scene_settings: dict = None):
        """Start video streaming with optional 3D background"""
        if self.is_streaming:
            return False
        
        if not self.cap:
            logger.error("No camera selected")
            return False
        
        self.use_3d_background = use_3d_background
        self.current_scene = scene_name
        if scene_settings:
            self.scene_settings.update(scene_settings)
        
        if use_3d_background:
            success = self._setup_3d_background()
            if not success:
                logger.warning("3D background setup failed, using procedural background")
        
        try:
            # Initialize virtual camera
            self.virtual_cam = pyvirtualcam.Camera(
                width=1280, height=720, fps=self.target_fps, fmt=pyvirtualcam.PixelFormat.BGR
            )
            
            self.is_streaming = True
            self.background_thread = threading.Thread(target=self._streaming_loop)
            self.background_thread.start()
            
            logger.info("Video streaming started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {e}")
            self.is_streaming = False
            return False
    
    def _streaming_loop(self):
        """Main streaming loop"""
        frame_count = 0
        last_time = time.time()
        
        while self.is_streaming:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Skip frames for performance if needed
                frame_count += 1
                if frame_count % (self.frame_skip + 1) != 0:
                    continue
                
                # Resize for processing
                original_size = frame.shape[:2][::-1]  # (width, height)
                if original_size != self.processing_resolution:
                    frame = cv2.resize(frame, self.processing_resolution)
                
                # Apply background effect
                processed_frame = self._apply_background_effect(frame)
                
                # Resize back to output resolution
                if processed_frame.shape[:2][::-1] != (1280, 720):
                    processed_frame = cv2.resize(processed_frame, (1280, 720))
                
                # Update current frame for preview
                with self.frame_lock:
                    self.current_frame = processed_frame.copy()
                
                # Send to virtual camera
                if self.virtual_cam:
                    self.virtual_cam.send(processed_frame)
                
                # FPS control
                current_time = time.time()
                elapsed = current_time - last_time
                target_delay = 1.0 / self.target_fps
                
                if elapsed < target_delay:
                    time.sleep(target_delay - elapsed)
                
                last_time = current_time
                
            except Exception as e:
                logger.error(f"Streaming loop error: {e}")
                time.sleep(0.1)
    
    def stop_streaming(self):
        """Stop video streaming"""
        self.is_streaming = False
        
        if self.background_thread:
            self.background_thread.join(timeout=5)
        
        if self.virtual_cam:
            self.virtual_cam.close()
            self.virtual_cam = None
        
        if self.driver:
            self.driver.quit()
            self.driver = None
        
        logger.info("Video streaming stopped")
    
    def get_current_frame(self) -> Optional[np.ndarray]:
        """Get current processed frame"""
        with self.frame_lock:
            return self.current_frame.copy() if self.current_frame is not None else None
    
    def update_scene_settings(self, scene_name: str, settings: dict):
        """Update 3D scene settings"""
        self.current_scene = scene_name
        self.scene_settings.update(settings)
        if self.driver:
            self._update_3d_scene()
    
    def capture_screenshot(self, path: str) -> bool:
        """Capture current frame as screenshot"""
        try:
            frame = self.get_current_frame()
            if frame is not None:
                cv2.imwrite(path, frame)
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to capture screenshot: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop_streaming()
        
        if self.cap:
            self.cap.release()
            self.cap = None