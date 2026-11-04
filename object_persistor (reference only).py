import sys
import os
# Add hybrid_control directory to path (assumes it's a sibling directory)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'hybrid_control'))
import hybrid_control as hc
import normalized_velocity_control as nvc
from get_cam_feeds import stop_all_cameras
from cam_info import HEAD_CAMERA, WRIST_CAMERA
from object_finder import ArucoFinder, MultiFinder, get_largest
from arucube_finder import ArucubeFinder
from yolo_finder import YoloFinder
from object_locator import ObjectLocator, LocatedObject
from robot_transforms import RobotTransforms
from visual_servoing import look_at_object
import stretch_body.robot as rb
import cv2
import numpy as np
import time
from scipy.optimize import linear_sum_assignment

def reliability_to_color(normalized_reliability):
    """Convert normalized reliability to BGR color.
    
    Args:
        normalized_reliability: Reliability value in [0, 1] range
    
    Returns:
        BGR color tuple (B, G, R) interpolated from red (low) to yellow to green (high)
    """
    # Interpolate color: red (0, 0, 255) -> yellow (0, 255, 255) -> green (0, 255, 0)
    if normalized_reliability > 0.5:
        # Upper half: yellow to green
        t = (normalized_reliability - 0.5) * 2  # 0 to 1
        color = (0, 255, int(255 * (1 - t)))
    else:
        # Lower half: red to yellow
        t = normalized_reliability * 2  # 0 to 1
        color = (0, int(255 * t), 255)
    
    return color

def draw_tracked_object_marker(drawing_frame, object_id, pixel_x, pixel_y, color):
    """Draw object ID and marker with specified color.
    
    Args:
        drawing_frame: Image to draw on
        object_id: Object ID number to display
        pixel_x, pixel_y: Pixel coordinates for marker
        color: BGR color tuple
    """
    # Draw ID with specified color
    text = f"{object_id}"
    cv2.putText(drawing_frame, text, (int(pixel_x), int(pixel_y) - 10), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Draw circle at centroid
    cv2.circle(drawing_frame, (int(pixel_x), int(pixel_y)), 4, color, -1)

def match_objects_by_identity_and_position(
    new_objects,           # List of any object type
    old_objects,           # List of any object type  
    get_identity_fn,       # Function: obj -> identity object for comparison
    get_position_fn,       # Function: obj -> np.array([x, y, z]) or None
    are_identical_fn,      # Function: (identity1, identity2) -> bool
    position_threshold     # float
):
    """Match new objects to old objects using identity grouping and position-based Hungarian algorithm.
    
    Returns:
        Tuple of (matched_new, matched_old, match_pairs) where:
        - matched_new: set of matched new object indices
        - matched_old: set of matched old object indices
        - match_pairs: list of (new_idx, old_idx) tuples
    """
    
    def group_by_identity(objects):
        """Group objects by identity."""
        groups = []
        used = set()
        
        for i, obj in enumerate(objects):
            if i in used:
                continue
            
            obj_identity = get_identity_fn(obj)
            group = [(i, obj)]
            used.add(i)
            
            # Find all other objects identical to this one
            for j, other_obj in enumerate(objects):
                if j <= i or j in used:
                    continue
                
                other_identity = get_identity_fn(other_obj)
                if are_identical_fn(obj_identity, other_identity):
                    group.append((j, other_obj))
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    # Group new and old objects by identity
    new_groups = group_by_identity(new_objects)
    old_groups = group_by_identity(old_objects)
    
    matched_new = set()
    matched_old = set()
    match_pairs = []
    
    # For each group of new objects, find the matching group of old objects
    for new_group in new_groups:
        if not new_group:
            continue
        
        # Get a representative object from the new group
        _, new_rep = new_group[0]
        new_rep_identity = get_identity_fn(new_rep)
        
        # Find the matching old group (if any)
        matching_old_group = None
        for old_group in old_groups:
            if not old_group:
                continue
            _, old_rep = old_group[0]
            old_rep_identity = get_identity_fn(old_rep)
            
            if are_identical_fn(new_rep_identity, old_rep_identity):
                matching_old_group = old_group
                break
        
        if matching_old_group is None:
            # No matching old group, these are all new objects
            continue
        
        # Build cost matrix for this group only
        group_cost_matrix = np.zeros((len(new_group), len(matching_old_group)))
        
        for i, (new_idx, new_obj) in enumerate(new_group):
            new_pos = get_position_fn(new_obj)
            if new_pos is None:
                group_cost_matrix[i, :] = np.inf
                continue
            
            for j, (old_idx, old_obj) in enumerate(matching_old_group):
                old_pos = get_position_fn(old_obj)
                if old_pos is None:
                    group_cost_matrix[i, j] = np.inf
                    continue
                group_cost_matrix[i, j] = np.linalg.norm(new_pos - old_pos)
        
        # Skip Hungarian algorithm if matrix is degenerate or has no valid matches
        if group_cost_matrix.size == 0:
            # Empty matrix
            print(f"DEBUG: Skipping empty cost matrix")
            continue
        if np.all(np.isinf(group_cost_matrix)):
            # All costs are infinite - no valid positions
            print(f"DEBUG: Skipping all-infinite cost matrix, shape: {group_cost_matrix.shape}")
            continue
        
        # Check if there are any feasible matches (costs below threshold)
        feasible_mask = group_cost_matrix < position_threshold
        if not np.any(feasible_mask):
            # No feasible matches in this group, skip Hungarian algorithm
            print(f"DEBUG: No feasible matches below threshold {position_threshold}, min cost: {np.min(group_cost_matrix[~np.isinf(group_cost_matrix)]) if np.any(~np.isinf(group_cost_matrix)) else 'all inf'}")
            continue
        
        # Debug: Check matrix state before Hungarian
        print(f"DEBUG: Running Hungarian on matrix shape {group_cost_matrix.shape}, finite values: {np.sum(np.isfinite(group_cost_matrix))}, min: {np.min(group_cost_matrix)}, max: {np.max(group_cost_matrix[np.isfinite(group_cost_matrix)])}")
        
        # Run Hungarian algorithm on this group
        row_ind, col_ind = linear_sum_assignment(group_cost_matrix)
        
        for new_i, old_i in zip(row_ind, col_ind):
            if group_cost_matrix[new_i, old_i] < position_threshold:
                new_idx, _ = new_group[new_i]
                old_idx, _ = matching_old_group[old_i]
                
                matched_new.add(new_idx)
                matched_old.add(old_idx)
                match_pairs.append((new_idx, old_idx))
    
    return matched_new, matched_old, match_pairs

class PersistentObject:
    _next_id = 0
    
    def __init__(self, located_object, persistor, id=None):
        self.located_object = located_object
        self.persistor = persistor
        self.visible = True
        self.last_seen_timestamp = located_object.image_object.timestamp
        self.last_base2world_T = located_object.base2world_T
        self.probation = 0.0  # 0.0 = starting probation, 1.0 = passed probation, -1.0 = failed probation
        self.first_seen_timestamp = located_object.image_object.timestamp  # Track when probation started
        
        if id is None:
            self.id = PersistentObject._next_id
            PersistentObject._next_id += 1
        else:
            self.id = id
    
    def update(self, new_located_object):
        """Update with new detection (called by ObjectPersistor on match).
        
        Args:
            new_located_object: LocatedObject if matched this frame, None if not seen
        """
        if new_located_object is not None:
            self.located_object = new_located_object
            self.visible = True
            self.last_seen_timestamp = new_located_object.image_object.timestamp
            self.last_base2world_T = new_located_object.base2world_T
            
            # Update probation if still in probation period
            if 0.0 <= self.probation < 1.0:
                time_elapsed = self.last_seen_timestamp - self.first_seen_timestamp
                probation_progress = time_elapsed / self.persistor.probation_period
                self.probation = min(1.0, probation_progress)
        else:
            self.visible = False
            
            # Fail probation if not seen during probation period
            if 0.0 <= self.probation < 1.0:
                self.probation = -1.0
    
    def get_reliability(self, time_decay_halflife=5.0):
        """Compute reliability factor (0.0-1.0) based on visibility and time.
        
        Args:
            time_decay_halflife: Half-life in seconds for time-based decay when not visible
        
        Returns:
            Reliability factor between 0.0 (unreliable) and 1.0 (fully reliable)
        """
        if self.visible:
            return 1.0
        
        # Time-based exponential decay when not visible
        time_elapsed = time.time() - self.last_seen_timestamp
        time_decay = np.exp(-time_elapsed * np.log(2) / time_decay_halflife)
        
        return time_decay
    
    def get_normalized_reliability(self):
        """Get reliability normalized to [0, 1] range relative to persistor's threshold."""
        reliability = self.get_reliability(self.persistor.time_decay_halflife)
        normalized = (reliability - self.persistor.reliability_threshold) / (1.0 - self.persistor.reliability_threshold)
        return np.clip(normalized, 0.0, 1.0)

class ObjectPersistor:
    def __init__(self, object_locator, position_threshold=0.2, reliability_threshold=0.5, time_decay_halflife=5.0, probation_period=1.0):
        self.object_locator = object_locator
        self.tracked_objects = []
        self.position_threshold = position_threshold
        self.reliability_threshold = reliability_threshold
        self.time_decay_halflife = time_decay_halflife
        self.probation_period = probation_period
    
    def _group_by_identity(self, located_objects):
        """Group objects by identity. Objects that are identical to each other go in the same group.
        
        Returns:
            List of groups, where each group is a list of (index, located_object) tuples
        """
        groups = []
        used = set()
        
        for i, obj in enumerate(located_objects):
            if i in used:
                continue
            
            # Start a new group with this object
            group = [(i, obj)]
            used.add(i)
            
            # Find all other objects identical to this one
            for j, other_obj in enumerate(located_objects):
                if j <= i or j in used:
                    continue
                
                identical = obj.image_object.is_identical_to(other_obj.image_object)
                if identical is True:
                    group.append((j, other_obj))
                    used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _match_objects_grouped(self, new_located_objects):
        """Match new detections to tracked objects by grouping identical objects first."""
        # Define accessor functions for the generic matching algorithm
        def get_identity(obj):
            # Handle both LocatedObject and PersistentObject
            if hasattr(obj, 'image_object'):
                return obj.image_object
            else:  # PersistentObject
                return obj.located_object.image_object
        
        def get_position(obj):
            # Handle both LocatedObject and PersistentObject
            if hasattr(obj, 'obj2world_T'):
                return obj.obj2world_T[:3, 3] if obj.obj2world_T is not None else None
            else:  # PersistentObject
                obj2world_T = obj.located_object.obj2world_T
                return obj2world_T[:3, 3] if obj2world_T is not None else None
        
        def are_identical(identity1, identity2):
            return identity1.is_identical_to(identity2) is True
        
        # Use generic matching algorithm
        matched_new, matched_old, match_pairs = match_objects_by_identity_and_position(
            new_objects=new_located_objects,
            old_objects=self.tracked_objects,
            get_identity_fn=get_identity,
            get_position_fn=get_position,
            are_identical_fn=are_identical,
            position_threshold=self.position_threshold
        )
        
        # Update matched tracked objects with new detections
        for new_idx, old_idx in match_pairs:
            self.tracked_objects[old_idx].update(new_located_objects[new_idx])
        
        # Update unmatched tracked objects with None (they weren't seen this frame)
        for i, persistent_obj in enumerate(self.tracked_objects):
            if i not in matched_old:
                persistent_obj.update(None)
        
        return matched_new, matched_old
    
    def _add_new_objects(self, new_located_objects, matched_new):
        """Create persistent objects for unmatched new detections."""
        for i, located_obj in enumerate(new_located_objects):
            if i not in matched_new:
                persistent_obj = PersistentObject(located_obj, self)
                self.tracked_objects.append(persistent_obj)
    
    def _remove_unreliable_objects(self):
        """Remove objects with low reliability (exponential decay for lost objects) and failed probation."""
        self.tracked_objects = [
            obj for obj in self.tracked_objects 
            if obj.get_reliability(self.time_decay_halflife) >= self.reliability_threshold
            and obj.probation != -1.0  # Remove objects that failed probation
        ]
    
    def _draw_tracked_objects(self, drawing_frame):
        """Draw tracked objects with reliability-based color coding."""
        if drawing_frame is None:
            return
        
        for persistent_obj in self.tracked_objects:
            obj = persistent_obj.located_object
            
            # Get object pose accounting for robot motion
            updated_obj2base_T, updated_obj2cam_T = obj.account_for_motion()
            
            # Check if pose is valid for drawing
            if updated_obj2cam_T is None:
                continue
            
            # Get object position on image
            pitch, yaw, distance = ObjectLocator.pose_to_polar(updated_obj2cam_T)
            pixel_x, pixel_y = self.object_locator.depth_cam_info.object_angles_to_pixel(yaw, pitch)
            
            if persistent_obj.probation >= 1.0:
                # Draw approved objects with reliability-based color coding
                normalized_reliability = persistent_obj.get_normalized_reliability()
                color = reliability_to_color(normalized_reliability)
                draw_tracked_object_marker(drawing_frame, persistent_obj.id, pixel_x, pixel_y, color)
            else:
                # Draw unapproved objects (still in probation) with yellow question mark
                color = (0, 255, 255)  # Yellow in BGR
                
                # Draw question mark
                text = "?"
                cv2.putText(drawing_frame, text, (int(pixel_x), int(pixel_y) - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw small circle at centroid
                cv2.circle(drawing_frame, (int(pixel_x), int(pixel_y)), 4, color, -1)
    
    def update(self, rgb_frame, depth_frame, drawing_frame=None):
        """Detect, locate, and track objects using optimal bipartite matching."""
        # Get current camera transform and detect objects
        cam_T = self.object_locator.robot_transforms.get_cam_T( self.object_locator.depth_cam_info )
        new_located_objects = self.object_locator.locate_all(rgb_frame, depth_frame, cam_T, None)
        
        if not new_located_objects:
            # Update all tracked objects with None (no detections this frame)
            for persistent_obj in self.tracked_objects:
                persistent_obj.update(None)
            self._remove_unreliable_objects()
            self._draw_tracked_objects(drawing_frame)
            return
        
        if not self.tracked_objects:
            # No existing objects, create new ones for all detections
            for located_obj in new_located_objects:
                self.tracked_objects.append(PersistentObject(located_obj, self))
            return
        
        # Match new detections to existing tracked objects
        matched_new, matched_old = self._match_objects_grouped(new_located_objects)
        
        # Add new objects and remove unreliable ones
        self._add_new_objects(new_located_objects, matched_new)
        self._remove_unreliable_objects()
        
        # Draw tracked objects with reliability-based colors
        self._draw_tracked_objects(drawing_frame)
    
    def get_visible_objects(self):
        """Return list of currently visible objects that have passed probation."""
        return [obj for obj in self.tracked_objects if obj.visible and obj.probation >= 1.0]
    
    def get_approved_objects(self):
        """Return all tracked objects that have passed probation."""
        return [obj for obj in self.tracked_objects if obj.probation >= 1.0]
    
    def get_all_objects(self):
        """Return all tracked objects (including those in probation)."""
        return self.tracked_objects

if __name__ == '__main__':
    try:
        # Initialize robot
        robot = rb.Robot()
        robot.startup()
        controller = nvc.NormalizedVelocityControl(robot)

        # Set up object persistor
        CAMERA = WRIST_CAMERA
        robot_transforms = RobotTransforms(robot)
        finder = MultiFinder([
            ArucubeFinder(ArucoFinder(dictionary=cv2.aruco.DICT_4X4_50, ids=[0, 1, 2])),
            YoloFinder(weights="tennis.pt", conf=0.4),
        ])
        locator = ObjectLocator(finder, CAMERA, robot_transforms)
        persistor = ObjectPersistor(locator, position_threshold=0.2)

        while True:
            # Get camera feeds
            rgb_frame, depth_frame = CAMERA.get_frames()
            drawing_frame = rgb_frame.copy()
            
            # Update persistent object tracking (includes drawing)
            persistor.update(rgb_frame, depth_frame, drawing_frame)
            
            # Debug info about tracking status
            total_tracked = len(persistor.get_all_objects())
            established = len(persistor.get_approved_objects())
            in_probation = len([obj for obj in persistor.tracked_objects if 0.0 <= obj.probation < 1.0])
            print(f"Tracking: {established} established, {in_probation} in probation ({total_tracked} total)")

            # Display feeds
            display_frame = cv2.resize(drawing_frame, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow('Color Camera', display_frame)
            
            # Track object with lowest ID (only approved objects)
            approved_objects = persistor.get_approved_objects()
            if approved_objects:
                target_obj = min(approved_objects, key=lambda obj: obj.id)
                head_cmd = look_at_object(target_obj.located_object, CAMERA)
            else:
                head_cmd = {}
            
            # Allow control
            cmd = hc.hybridize(head_cmd) # Human override
            controller.set_command(cmd)

            # Exit on ESC key or if plot window is closed
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key or plot closed
                break
        
    finally:
        cv2.destroyAllWindows()
        stop_all_cameras()
        robot.stop()