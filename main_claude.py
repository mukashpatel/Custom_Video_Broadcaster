import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from stream_utils import Streaming
import threading
import cv2
import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import cv2
from typing import Optional, Dict, Any
import math
import time

app = FastAPI()

# Mount static files (e.g., your HTML + assets)
app.mount("/static", StaticFiles(directory="static"), name="static")
stream_thread =None
# Global instance of your streamer
streaming = Streaming()


@app.get("/")
def serve_ui():
      return FileResponse("static/3d_background.html")
    # return FileResponse("static/index.html")


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




class Background3DGenerator:
    """3D Background Generator for video streaming"""
    
    def __init__(self):
        self.current_settings = {
            'scene_type': 'particles',
            'animation_speed': 1.0,
            'object_count': 100,
            'primary_color': (78, 205, 196),  # RGB values
            'secondary_color': (255, 107, 107),
            'brightness': 1.0,
            'bloom': 0.5,
            'camera_distance': 50,
            'auto_rotate': True,
            'fov': 75
        }
        self.frame_time = 0
    
    def generate_particle_background(self, width: int, height: int) -> np.ndarray:
        """Generate animated particle background"""
        # Create base image
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        # Animate particles
        self.frame_time += 0.1 * self.current_settings['animation_speed']
        
        for i in range(self.current_settings['object_count']):
            # Create particle positions with movement
            x = (math.sin(self.frame_time + i * 0.1) * 0.3 + 0.5) * width
            y = (math.cos(self.frame_time * 0.7 + i * 0.15) * 0.3 + 0.5) * height
            
            # Particle size based on depth simulation
            size = int(5 + 3 * math.sin(self.frame_time + i * 0.2))
            
            # Alternate colors
            color = self.current_settings['primary_color'] if i % 2 == 0 else self.current_settings['secondary_color']
            
            # Apply brightness
            color = tuple(min(255, int(c * self.current_settings['brightness'])) for c in color)
            
            # Draw particle with glow effect
            for glow in range(3, 0, -1):
                alpha = 50 // glow
                glow_color = tuple(c // (4 - glow) for c in color)
                draw.ellipse([x - size - glow, y - size - glow, 
                            x + size + glow, y + size + glow], 
                           fill=glow_color)
        
        # Apply bloom effect if enabled
        if self.current_settings['bloom'] > 0:
            img = img.filter(ImageFilter.GaussianBlur(radius=self.current_settings['bloom'] * 2))
        
        return np.array(img)
    
    def generate_geometry_background(self, width: int, height: int) -> np.ndarray:
        """Generate animated geometric shapes background"""
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        self.frame_time += 0.05 * self.current_settings['animation_speed']
        
        for i in range(min(self.current_settings['object_count'] // 5, 50)):
            # Position with rotation
            center_x = width // 2 + math.cos(self.frame_time + i) * width * 0.3
            center_y = height // 2 + math.sin(self.frame_time * 0.7 + i) * height * 0.3
            
            # Size with pulsing effect
            size = 20 + 15 * math.sin(self.frame_time * 2 + i)
            
            # Rotation
            rotation = self.frame_time + i * 0.5
            
            # Color selection
            color = self.current_settings['primary_color'] if i % 2 == 0 else self.current_settings['secondary_color']
            color = tuple(min(255, int(c * self.current_settings['brightness'])) for c in color)
            
            # Draw different shapes
            shape_type = i % 4
            if shape_type == 0:  # Rectangle
                points = self._rotate_rectangle(center_x, center_y, size, rotation)
                draw.polygon(points, fill=color)
            elif shape_type == 1:  # Triangle
                points = self._rotate_triangle(center_x, center_y, size, rotation)
                draw.polygon(points, fill=color)
            elif shape_type == 2:  # Circle
                draw.ellipse([center_x - size, center_y - size, 
                            center_x + size, center_y + size], fill=color)
            else:  # Diamond
                points = self._rotate_diamond(center_x, center_y, size, rotation)
                draw.polygon(points, fill=color)
        
        return np.array(img)
    
    def generate_wave_background(self, width: int, height: int) -> np.ndarray:
        """Generate animated wave background"""
        img = Image.new('RGB', (width, height), (0, 0, 0))
        pixels = img.load()
        
        self.frame_time += 0.05 * self.current_settings['animation_speed']
        
        for y in range(height):
            for x in range(width):
                # Create wave patterns
                wave1 = math.sin((x * 0.01) + self.frame_time) * 0.5 + 0.5
                wave2 = math.cos((y * 0.008) + self.frame_time * 0.7) * 0.5 + 0.5
                wave3 = math.sin(((x + y) * 0.005) + self.frame_time * 1.2) * 0.5 + 0.5
                
                # Combine waves
                intensity = (wave1 + wave2 + wave3) / 3
                
                # Color interpolation
                color1 = np.array(self.current_settings['primary_color'])
                color2 = np.array(self.current_settings['secondary_color'])
                color = color1 * intensity + color2 * (1 - intensity)
                color = color * self.current_settings['brightness']
                color = np.clip(color, 0, 255).astype(int)
                
                pixels[x, y] = tuple(color)
        
        return np.array(img)
    
    def generate_tunnel_background(self, width: int, height: int) -> np.ndarray:
        """Generate space tunnel background"""
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        self.frame_time += 0.1 * self.current_settings['animation_speed']
        
        center_x, center_y = width // 2, height // 2
        
        # Draw concentric rings moving toward viewer
        for i in range(20):
            # Ring position and size
            ring_distance = (i * 50 + self.frame_time * 100) % 1000
            ring_size = min(width, height) * (ring_distance / 1000)
            
            if ring_size > 0:
                # Color based on distance
                color_intensity = 1 - (ring_distance / 1000)
                color = self.current_settings['primary_color'] if i % 2 == 0 else self.current_settings['secondary_color']
                color = tuple(int(c * color_intensity * self.current_settings['brightness']) for c in color)
                
                # Draw ring
                ring_width = max(1, int(ring_size * 0.05))
                draw.ellipse([center_x - ring_size, center_y - ring_size,
                            center_x + ring_size, center_y + ring_size],
                           outline=color, width=ring_width)
        
        return np.array(img)
    
    def generate_crystal_background(self, width: int, height: int) -> np.ndarray:
        """Generate crystal field background"""
        img = Image.new('RGB', (width, height), (0, 0, 0))
        draw = ImageDraw.Draw(img)
        
        self.frame_time += 0.03 * self.current_settings['animation_speed']
        
        for i in range(min(self.current_settings['object_count'] // 3, 60)):
            # Crystal position with slow movement
            x = (width * 0.1) + (i * 137.5) % (width * 0.8)  # Golden ratio distribution
            y = (height * 0.1) + (i * 219.7) % (height * 0.8)
            
            # Add movement
            x += math.sin(self.frame_time + i * 0.1) * 20
            y += math.cos(self.frame_time * 0.7 + i * 0.15) * 15
            
            # Crystal size with pulsing
            size = 15 + 10 * math.sin(self.frame_time * 2 + i * 0.3)
            height_mult = 1.5 + 0.5 * math.cos(self.frame_time + i * 0.2)
            
            # Color and brightness
            color = self.current_settings['primary_color'] if i % 3 != 0 else self.current_settings['secondary_color']
            brightness = self.current_settings['brightness'] * (0.7 + 0.3 * math.sin(self.frame_time + i))
            color = tuple(min(255, int(c * brightness)) for c in color)
            
            # Draw crystal as diamond shape
            points = [
                (x, y - size * height_mult),  # Top
                (x - size, y),                # Left
                (x, y + size * height_mult),  # Bottom
                (x + size, y)                 # Right
            ]
            draw.polygon(points, fill=color, outline=tuple(min(255, c + 50) for c in color))
        
        return np.array(img)
    
    def _rotate_rectangle(self, cx: float, cy: float, size: float, rotation: float) -> list:
        """Helper function to create rotated rectangle points"""
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        points = [(-size, -size), (size, -size), (size, size), (-size, size)]
        return [(cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r) for x, y in points]
    
    def _rotate_triangle(self, cx: float, cy: float, size: float, rotation: float) -> list:
        """Helper function to create rotated triangle points"""
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        points = [(0, -size), (-size * 0.866, size * 0.5), (size * 0.866, size * 0.5)]
        return [(cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r) for x, y in points]
    
    def _rotate_diamond(self, cx: float, cy: float, size: float, rotation: float) -> list:
        """Helper function to create rotated diamond points"""
        cos_r, sin_r = math.cos(rotation), math.sin(rotation)
        points = [(0, -size), (size, 0), (0, size), (-size, 0)]
        return [(cx + x * cos_r - y * sin_r, cy + x * sin_r + y * cos_r) for x, y in points]
    
    def generate_background(self, width: int, height: int) -> np.ndarray:
        """Main function to generate background based on current settings"""
        scene_type = self.current_settings['scene_type']
        
        if scene_type == 'particles':
            return self.generate_particle_background(width, height)
        elif scene_type == 'geometry':
            return self.generate_geometry_background(width, height)
        elif scene_type == 'waves':
            return self.generate_wave_background(width, height)
        elif scene_type == 'tunnel':
            return self.generate_tunnel_background(width, height)
        elif scene_type == 'crystals':
            return self.generate_crystal_background(width, height)
        else:
            return self.generate_particle_background(width, height)
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update 3D background settings"""
        self.current_settings.update(settings)
    
    def get_settings(self) -> Dict[str, Any]:
        """Get current 3D background settings"""
        return self.current_settings.copy()


# Add this to your existing FastAPI routes in main.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Initialize the 3D background generator
bg_3d_generator = Background3DGenerator()

class Background3DSettings(BaseModel):
    scene_type: Optional[str] = None
    animation_speed: Optional[float] = None
    object_count: Optional[int] = None
    primary_color: Optional[tuple] = None
    secondary_color: Optional[tuple] = None
    brightness: Optional[float] = None
    bloom: Optional[float] = None
    camera_distance: Optional[int] = None
    auto_rotate: Optional[bool] = None
    fov: Optional[int] = None

@app.post("/api/background-3d/settings")
async def update_3d_background_settings(settings: Background3DSettings):
    """Update 3D background settings"""
    try:
        # Convert model to dict and filter out None values
        settings_dict = {k: v for k, v in settings.dict().items() if v is not None}
        bg_3d_generator.update_settings(settings_dict)
        return {"status": "success", "message": "3D background settings updated"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update settings: {str(e)}")

@app.get("/api/background-3d/settings")
async def get_3d_background_settings():
    """Get current 3D background settings"""
    try:
        return bg_3d_generator.get_settings()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")

@app.post("/api/background-3d/preview")
async def generate_3d_background_preview():
    """Generate a preview frame of the current 3D background"""
    try:
        # Generate preview at standard resolution
        preview_bg = bg_3d_generator.generate_background(640, 480)
        
        # Convert to base64 for web display
        import base64
        from io import BytesIO
        
        img = Image.fromarray(preview_bg)
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_str = base64.b64encode(buffer.getvalue()).decode()
        
        return {"preview": f"data:image/png;base64,{img_str}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate preview: {str(e)}")


# Update your existing engine.py to include 3D background support

# Add this method to your existing VideoEngine class in engine.py:

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

# Update your process_frame method to include 3D background option:
# (Add this case to your existing background processing logic)

elif self.background_mode == "3d":
    frame = self.apply_3d_background(frame, person_mask)


# Add this to requirements.txt:
"""
Pillow>=9.0.0
numpy>=1.21.0
"""