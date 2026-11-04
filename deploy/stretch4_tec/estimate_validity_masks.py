#!/usr/bin/env python3
import sys
import os
import cv2
import numpy as np
from pathlib import Path

# Ensure the root directory is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stretch4_emulated_rgbd.shared_utils import get_arg_parser, generate_rgb_vignetting_mask, generate_depth_validity_mask
from stretch4_emulated_rgbd.api import get_emulated_rgbd_stream

def estimate_masks(args):
    # We force the resolution to the maximum available for the OAK-D cameras (800p)
    # This ensures that we estimate the mask at the highest possible fidelity. 
    # The ValidityMaskManager will automatically downscale these high-res masks if 
    # downstream applications request a lower resolution (e.g., 400p).
    args.resolution = 800
    
    streamer, generator = get_emulated_rgbd_stream(
        use_left=(args.camera in ["left", "left_right", "all"]),
        use_right=(args.camera in ["right", "left_right", "all"]),
        use_center=(args.camera in ["center", "all"]),
        use_left_right=(args.camera == "left_right"),
        use_left_right_center=(args.camera == "all"),
        use_left_lidar=(args.lidar in ["left", "both"]),
        use_right_lidar=(args.lidar in ["right", "both"]),
        emulated_rgbd_fps=args.emulated_rgbd_fps,
        camera_fps=args.camera_fps,
        resolution_height=args.resolution,
        compress=not args.disable_compression,
        oak_buffer_size=args.oak_buffer_size,
        calibration=None,
        merge_lidars=args.merge_lidars
    )
    if streamer is None:
        print("Failed to initialize streamer.")
        return

    # We capture multiple frames over time (default 30) instead of just one.
    # Why? The LiDAR data is sparse and the RGB data has noise. By accumulating 
    # data over a couple of seconds while the robot might be moving slightly or 
    # observing different scenes, we can generate a much more robust average 
    # representation of the true static hardware vignetting and LiDAR field-of-view bounds.
    frames_to_capture = 30
    camera_data = {}
    
    print(f"Capturing {frames_to_capture} frames to estimate validity masks...")
    
    try:
        for frame_data in generator:
            if frame_data is None: continue
            
            # frame_data could be a single frame or a synced_frame
            frames = []
            if hasattr(frame_data, "left") or hasattr(frame_data, "right") or hasattr(frame_data, "center"):
                if getattr(frame_data, "left", None): frames.append(frame_data.left)
                if getattr(frame_data, "right", None): frames.append(frame_data.right)
                if getattr(frame_data, "center", None): frames.append(frame_data.center)
            else:
                frames.append(frame_data)

                
            done = True
            for f in frames:
                c_name = f.camera_type
                if c_name not in camera_data:
                    camera_data[c_name] = {"rgb": [], "depth": []}
                
                if len(camera_data[c_name]["rgb"]) < frames_to_capture:
                    # Convert RGB to grayscale for vignetting mask
                    gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
                    camera_data[c_name]["rgb"].append(gray)
                    camera_data[c_name]["depth"].append(f.depth_image.copy())
                    print(f"Captured frame {len(camera_data[c_name]['rgb'])}/{frames_to_capture} for {c_name} camera")
                    
                if len(camera_data[c_name]["rgb"]) < frames_to_capture:
                    done = False
                    
            if done and camera_data:
                break
    except KeyboardInterrupt:
        print("Interrupted by user. Using collected frames.")
    finally:
        streamer.stop()

    # Create masks dir
    fleet_path = os.environ.get("HELLO_FLEET_PATH", "")
    fleet_id = os.environ.get("HELLO_FLEET_ID", "")
    if fleet_path and fleet_id:
        masks_dir = Path(os.path.join(fleet_path, fleet_id, "calibration_cameras"))
    else:
        masks_dir = Path("data/validity_masks")
    masks_dir.mkdir(parents=True, exist_ok=True)
    
    lidar_str = "no_lidar"
    if args.lidar == "both":
        lidar_str = "both_lidar"
    elif args.lidar == "left":
        lidar_str = "left_lidar"
    elif args.lidar == "right":
        lidar_str = "right_lidar"
    
    for c_name, data in camera_data.items():
        if len(data["rgb"]) == 0:
            continue
            
        print(f"\nProcessing {c_name} camera...")
        
        # 1. Generate RGB Vignetting Mask
        # This identifies the dark corners of the fisheye lens housing, allowing us
        # to explicitly zero them out so computer vision models don't process the housing rim.
        print("Generating RGB vignetting mask...")
        rgb_vignette_mask = generate_rgb_vignetting_mask(data["rgb"])
        
        # 2. Generate Depth Validity Mask
        # This identifies the true spatial bounds of the physical LiDAR laser coverage.
        # It prevents our dense depth interpolator from wildly extrapolating depth 
        # values into regions of the camera frame that the LiDAR cannot physically see.
        print("Generating Depth validity mask...")
        depth_valid_mask = generate_depth_validity_mask(data["depth"])
        
        if rgb_vignette_mask is not None:
            out_rgb_path = masks_dir / f"rgb_vignette_mask_{c_name}_camera.png"
            cv2.imwrite(str(out_rgb_path), rgb_vignette_mask)
            print(f"Saved RGB vignetting mask to: {out_rgb_path}")
            
        if depth_valid_mask is not None:
            out_depth_path = masks_dir / f"depth_valid_mask_{c_name}_camera_{lidar_str}.png"
            cv2.imwrite(str(out_depth_path), depth_valid_mask)
            print(f"Saved Depth valid mask to: {out_depth_path}")

def main():
    parser = get_arg_parser("Capture and estimate validity masks for Emulated RGB-D cameras.")
    args = parser.parse_args()
    
    # We only want live streaming for mask estimation, prevent replay args
    args.data_path = None
    args.opt_yaml = None
    
    estimate_masks(args)
    os._exit(0)

if __name__ == "__main__":
    main()
