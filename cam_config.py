import cv2
import numpy as np
import threading
import queue
import time

def rotate_intrinsics(rotation, K, image_size):
    """Rotate camera intrinsics for rotated image.
    
    Args:
        rotation: cv2.ROTATE_* constant
        K: 3x3 intrinsic matrix
        image_size: (height, width) of original image before rotation
        
    Returns:
        K_rotated: Rotated intrinsic matrix
    """
    h, w = image_size
    K_rot = K.copy()
    
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        # (h, w) -> (w, h)
        # (x, y) -> (h - y, x)
        K_rot[0, 2] = h - K[1, 2]  # new_cx = h - old_cy
        K_rot[1, 2] = K[0, 2]       # new_cy = old_cx
        
    elif rotation == cv2.ROTATE_90_COUNTERCLOCKWISE:
        # (h, w) -> (w, h)
        # (x, y) -> (y, w - x)
        K_rot[0, 2] = K[1, 2]       # new_cx = old_cy
        K_rot[1, 2] = w - K[0, 2]  # new_cy = w - old_cx
        
    elif rotation == cv2.ROTATE_180:
        # (h, w) -> (h, w)
        # (x, y) -> (w - x, h - y)
        K_rot[0, 2] = w - K[0, 2]  # new_cx = w - old_cx
        K_rot[1, 2] = h - K[1, 2]  # new_cy = h - old_cy
    
    return K_rot

# TODO: for cams such as droidcam, where we might need to rotate the frame, we should specify the unrotated intrinsics, then if user requires rotation, we can compute the new intrinsics accordingly.
class Camera:
    """Camera configuration with intrinsics and frame acquisition."""
    
    def __init__(self, K, D, frame_getter, rotation=None, image_shape_hw=None):
        """Initialize camera.
        
        Args:
            K: Camera intrinsic matrix
            D: Distortion coefficients
            frame_getter: Callable that returns a frame
            rotation: Optional cv2.ROTATE_* constant to apply to frames
            image_shape_hw: Optional (height, width) tuple. Required if rotation is specified.
        """
        # Validate that rotation and image_shape_hw are provided together
        if (rotation is None) != (image_shape_hw is None):
            raise ValueError("rotation and image_shape_hw must both be provided or both be None")
        
        self.frame_getter = frame_getter
        self.rotation = rotation
        self.D = D  # Distortion coefficients don't change (radially symmetric)
        self._released = False
        
        # Compute rotated intrinsics if rotation specified
        if rotation is not None:
            self.K = rotate_intrinsics(rotation, K, image_shape_hw)
        else:
            self.K = K
    
    def get_frame(self):
        """Get frame from frame_getter."""
        if self._released:
            return None

        frame = self.frame_getter()
        if frame is not None and self.rotation is not None:
            frame = cv2.rotate(frame, self.rotation)
        return frame

    def read(self):
        """Return a frame using the cv2.VideoCapture interface."""
        frame = self.get_frame()
        return frame is not None, frame

    def isOpened(self):
        """Return whether this camera has not been released."""
        return not self._released

    def release(self):
        """Release the underlying source when it supports release()."""
        configured_source = getattr(self.frame_getter, "source", None)
        if configured_source is not None and hasattr(configured_source, "release"):
            configured_source.release()
        else:
            source = getattr(self.frame_getter, "cap", None)
            if source is not None and hasattr(source, "release"):
                source.release()
        self._released = True
