import cv2 
from ultralytics import YOLO
import numpy as np
import torch
import math
import random

class CustomerSegmentationWithYOLO():
    def __init__(self, erode_size=5, erode_intensity=2, background_path='static/IMG-20250705-WA0006.jpg'):
        self.model = YOLO('yolov8n-seg.pt')
        self.erode_size = erode_size
        self.erode_intensity = erode_intensity
        self.background_image = cv2.imread(background_path)
        if self.background_image is None:
            print("[Warning] Background image not found, using black background.")
            self.background_image = np.zeros((480, 640, 3), dtype=np.uint8)
        # Starfield state used for 3D background rendering
        self._starfield_state = None

    def generate_mask_from_result(self, results):
        for result in results:
            if result.masks is not None:
                # Get masks and bounding boxes
                masks = result.masks.data  # shape: (N, H, W)
                boxes = result.boxes.data  # shape: (N, 6)

                # Extract class labels
                clss = boxes[:, 5].int()  # COCO class IDs

                # Find indices for class 0 (person)
                people_indices = torch.where(clss == 0)[0]

                if len(people_indices) == 0:
                    return None

                # Get only the masks for people
                people_masks = masks[people_indices]

                # Merge all people masks into one binary mask
                people_mask = torch.any(people_masks, dim=0).to(torch.uint8) * 255

                # Erode the mask
                kernel = np.ones((self.erode_size, self.erode_size), np.uint8)
                eroded_mask = cv2.erode(people_mask.cpu().numpy(), kernel, iterations=self.erode_intensity)

                if eroded_mask.sum() < 100:  # low pixel count means weak detection
                    return None

                return eroded_mask  # Already NumPy

        return None

    def apply_blur_with_mask(self, frame, mask, blur_strength=21):
        # If blur_strength is too low, skip blurring and return the original frame
        if isinstance(blur_strength, tuple):
            strength_val = blur_strength[0]
        else:
            strength_val = blur_strength
        if strength_val <= 5:
            return frame

        # Ensure blur_strength is odd
        if strength_val % 2 == 0:
            strength_val += 1
        blur_kernel = (strength_val, strength_val)

        # Apply Gaussian blur to the whole frame
        blurred_frame = cv2.GaussianBlur(frame, blur_kernel, 0)

        # Ensure mask is binary (0 or 255)
        mask = (mask > 0).astype(np.uint8) * 255

        # Expand to 3 channels
        mask_3d = cv2.merge([mask, mask, mask])

        # Combine using the mask (keep original where mask is 255, blurred elsewhere)
        result_frame = np.where(mask_3d == 255, frame, blurred_frame)

        return result_frame

    def apply_black_background(self, frame, mask):
        # Ensure mask is binary (0 or 255)
        mask = (mask > 0).astype(np.uint8) * 255

        # Expand mask to 3 channels
        mask_3d = cv2.merge([mask, mask, mask])

        # Create a black background
        black_background = np.zeros_like(frame)

        # Combine: keep frame where mask is 255, otherwise black
        result_frame = np.where(mask_3d == 255, frame, black_background)

        return result_frame

    def apply_custom_background(self, frame, mask):
        # Resize background to match frame dimensions
        background_image = cv2.resize(self.background_image, (frame.shape[1], frame.shape[0]))

        # Ensure binary mask (0 or 255)
        mask = (mask > 0).astype(np.uint8) * 255

        # Expand mask to 3 channels
        mask_3d = cv2.merge([mask, mask, mask])

        # Combine: show frame where mask is 255, else show background
        result_frame = np.where(mask_3d == 255, frame, background_image)

        return result_frame

    def apply_virtual_background(self, frame, mask):
        # Just re-use the custom background logic
        return self.apply_custom_background(frame, mask)

    def apply_dynamic_background(self, frame, mask, background_frame):
        """Composite a dynamically generated background frame behind the subject mask."""
        bg_resized = cv2.resize(background_frame, (frame.shape[1], frame.shape[0]))
        mask_bin = (mask > 0).astype(np.uint8) * 255
        mask_3d = cv2.merge([mask_bin, mask_bin, mask_bin])
        return np.where(mask_3d == 255, frame, bg_resized)

    def render_3d_grid_background(self, width: int, height: int, t: float = 0.0) -> np.ndarray:
        """Render a retro neon perspective grid with a moving floor as a faux-3D background."""
        img = np.zeros((height, width, 3), dtype=np.uint8)

        # Gradient sky
        top_color = np.array([60, 0, 80], dtype=np.float32)   # BGR
        bottom_color = np.array([10, 10, 20], dtype=np.float32)
        for y in range(height):
            alpha = y / height
            color = (1 - alpha) * top_color + alpha * bottom_color
            img[y, :, :] = color

        cx, cy = width // 2, int(height * 0.38)
        horizon_y = cy

        # Animate parameters
        num_h_lines = 22
        num_v_lines = 24
        scroll_speed = 0.6  # lines per second
        phase = (t * scroll_speed) % 1.0

        # Colors
        line_color = (200, 60, 255)  # neon purple (BGR)
        line_thickness = 1

        # Horizontal lines with perspective spacing
        for k in range(1, num_h_lines + 1):
            # Quadratic spacing plus animated phase
            p = (k + phase) / (num_h_lines + 1)
            y = int(horizon_y + (height - horizon_y) * (p ** 2))
            cv2.line(img, (0, y), (width, y), line_color, line_thickness, cv2.LINE_AA)

        # Vertical lines radiating from vanishing point
        max_angle = math.radians(55)
        for i in range(-num_v_lines // 2, num_v_lines // 2 + 1):
            a = (i / (num_v_lines // 2)) * max_angle
            # Animate slight sway
            a += 0.06 * math.sin(t * 1.2 + i * 0.3)
            x_end = int(cx + math.tan(a) * (height - horizon_y))
            cv2.line(img, (cx, horizon_y), (x_end, height), line_color, line_thickness, cv2.LINE_AA)

        # Glow effect via slight blur
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=1.0)
        return img

    def _init_starfield(self, width: int, height: int, num_stars: int = 600):
        rng = np.random.default_rng()
        # x, y in [-1, 1], z in (0.1, 1]
        self._starfield_state = {
            "width": width,
            "height": height,
            "x": rng.uniform(-1.0, 1.0, size=(num_stars,)).astype(np.float32),
            "y": rng.uniform(-1.0, 1.0, size=(num_stars,)).astype(np.float32),
            "z": rng.uniform(0.15, 1.0, size=(num_stars,)).astype(np.float32),
        }

    def render_starfield_background(self, width: int, height: int, dz_per_frame: float = 0.025) -> np.ndarray:
        """Render a simple moving starfield using a manual perspective projection."""
        if self._starfield_state is None \
           or self._starfield_state.get("width") != width \
           or self._starfield_state.get("height") != height:
            self._init_starfield(width, height)

        state = self._starfield_state
        x = state["x"]
        y = state["y"]
        z = state["z"]

        # Move stars towards the viewer
        z -= dz_per_frame
        # Reset stars that passed the camera
        reset_mask = z <= 0.05
        if np.any(reset_mask):
            rng = np.random.default_rng()
            x[reset_mask] = rng.uniform(-1.0, 1.0, size=reset_mask.sum()).astype(np.float32)
            y[reset_mask] = rng.uniform(-1.0, 1.0, size=reset_mask.sum()).astype(np.float32)
            z[reset_mask] = rng.uniform(0.6, 1.0, size=reset_mask.sum()).astype(np.float32)
        state["z"] = z

        # Perspective projection
        cx, cy = width // 2, height // 2
        f = 0.9 * min(width, height)
        inv_z = 1.0 / z
        x2d = (x * f * inv_z) + cx
        y2d = (y * f * inv_z) + cy

        # Prepare background (dark blue/purple gradient)
        img = np.zeros((height, width, 3), dtype=np.uint8)
        top_color = np.array([20, 0, 30], dtype=np.float32)
        bottom_color = np.array([5, 5, 15], dtype=np.float32)
        for row in range(height):
            alpha = row / height
            color = (1 - alpha) * top_color + alpha * bottom_color
            img[row, :, :] = color

        # Draw stars
        x_int = np.clip(x2d.astype(np.int32), 0, width - 1)
        y_int = np.clip(y2d.astype(np.int32), 0, height - 1)
        # Star size and brightness scale with depth
        size = np.clip(((1.2 - z) * 2.2), 0.5, 3.0)
        brightness = np.clip(((1.1 - z) * 255), 80, 255).astype(np.uint8)
        for xi, yi, si, br in zip(x_int, y_int, size, brightness):
            cv2.circle(img, (int(xi), int(yi)), int(si), (int(br), int(br), int(br)), -1, lineType=cv2.LINE_AA)

        # Slight glow
        img = cv2.GaussianBlur(img, (0, 0), sigmaX=0.8)
        return img
