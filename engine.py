import cv2 
from ultralytics import YOLO
import numpy as np
import torch

class CustomerSegmentationWithYOLO():
    def __init__(self, erode_size=5, erode_intensity=2, background_path='static/IMG-20250705-WA0006.jpg'):
        self.model = YOLO('yolov8n-seg.pt')
        self.erode_size = erode_size
        self.erode_intensity = erode_intensity
        self.background_image = cv2.imread(background_path)
        if self.background_image is None:
            print("[Warning] Background image not found, using black background.")
            self.background_image = np.zeros((480, 640, 3), dtype=np.uint8)

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
