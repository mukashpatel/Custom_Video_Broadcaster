from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum

class EffectType(str, Enum):
    """Background effect types"""
    BLUR = "blur"
    BLACK = "black"
    IMAGE = "image"
    THREED = "3d"

class SceneType(str, Enum):
    """3D Scene types"""
    CORPORATE = "corporate"
    CYBER = "cyber"
    PARTICLES = "particles"
    GEOMETRIC = "geometric"
    WAVES = "waves"
    SPACE = "space"
    MATRIX = "matrix"
    NEURAL = "neural"
    CRYSTAL = "crystal"
    FIRE = "fire"

class BackgroundConfig(BaseModel):
    """Background configuration model"""
    effect_type: EffectType = EffectType.BLUR
    background_image: Optional[str] = None
    blur_strength: int = Field(default=15, ge=1, le=50)
    use_3d_background: bool = False
    scene_name: SceneType = SceneType.CORPORATE

class SceneConfig(BaseModel):
    """3D Scene configuration model"""
    speed: float = Field(default=1.0, ge=0.1, le=5.0)
    particles: int = Field(default=200, ge=50, le=1000)
    intensity: float = Field(default=1.0, ge=0.1, le=2.0)
    auto_transition: bool = False
    transition_interval: int = Field(default=30, ge=5, le=300)
    custom_colors: Optional[List[str]] = None
    complexity: Optional[str] = "medium"

class CameraDevice(BaseModel):
    """Camera device model"""
    id: int
    name: str
    resolution: str
    is_available: bool = True

class StreamStatus(BaseModel):
    """Streaming status model"""
    is_streaming: bool
    current_scene: SceneType
    device_id: Optional[int] = None
    fps: float = 30.0
    resolution: str = "1280x720"
    uptime_seconds: int = 0

class ScenePreset(BaseModel):
    """Scene preset model"""
    name: str
    description: str
    scene_type: SceneType
    settings: SceneConfig
    thumbnail_url: Optional[str] = None
    tags: List[str] = []

class SystemSettings(BaseModel):
    """System configuration model"""
    target_fps: int = Field(default=30, ge=15, le=60)
    processing_resolution: str = "640x480"
    output_resolution: str = "1280x720"
    enable_gpu_acceleration: bool = True
    log_level: str = "INFO"
    auto_start_streaming: bool = False

class PerformanceMetrics(BaseModel):
    """Performance metrics model"""
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None
    fps_actual: float
    frame_processing_time: float
    frames_dropped: int = 0

class APIResponse(BaseModel):
    """Generic API response model"""
    status: str
    message: Optional[str] = None
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class SceneExport(BaseModel):
    """Scene export model"""
    scene_name: str
    config: SceneConfig
    preset_name: Optional[str] = None
    created_at: str
    version: str = "2.0"