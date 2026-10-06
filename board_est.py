import threading
import numpy as np
import cv2
import math

def vecs_to_matrix(rvec, tvec):
    """Convert rvec, tvec to a 4x4 transformation matrix."""
    rvec = np.asarray(rvec, dtype=np.float32)
    tvec = np.asarray(tvec, dtype=np.float32)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def matrix_to_vecs(T):
    """Convert a 4x4 transformation matrix to rvec, tvec."""
    R = T[:3, :3]
    tvec = T[:3, 3]
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten(), tvec.flatten()

class PnpResult:
    def __init__(self, obj_pts, img_pts, tvec, rvec):
        """
        obj_pts: array of shape (N, 1, 3) or (N, 3) containing 3D object‐space coordinates
                 (X, Y, Z) of detected Charuco corners (Z is usually 0).
        img_pts: array of shape (N, 1, 2) or (N, 2) containing 2D image‐space coordinates (u, v).
        tvec, rvec: the usual solvePnP outputs (not used in project_point).
        """
        # Convert obj_pts to shape (N, 2) by flattening and taking X, Y only
        obj = np.asarray(obj_pts, dtype=np.float32)
        if obj.ndim == 3 and obj.shape[1] == 1 and obj.shape[2] == 3:
            obj = obj.reshape(-1, 3)
        elif obj.ndim == 2 and obj.shape[1] == 3:
            pass
        else:
            raise ValueError(f"Unexpected obj_pts shape {obj.shape}, expected (N,1,3) or (N,3)")

        # Only keep X, Y columns
        self.obj_pts = obj[:, :2].copy()  # shape (N, 2)

        # Convert img_pts to shape (N, 2)
        img = np.asarray(img_pts, dtype=np.float32)
        if img.ndim == 3 and img.shape[1] == 1 and img.shape[2] == 2:
            img = img.reshape(-1, 2)
        elif img.ndim == 2 and img.shape[1] == 2:
            pass
        else:
            raise ValueError(f"Unexpected img_pts shape {img.shape}, expected (N,1,2) or (N,2)")

        self.img_pts = img.copy()  # shape (N, 2)

        self.tvec = tvec
        self.rvec = rvec

    def get_ref_T(self):
        """Get the 4x4 transformation matrix from reference to camera coordinates.
        
        Returns:
            4x4 numpy array representing the board pose
        """
        return vecs_to_matrix(self.rvec, self.tvec)

    def get_quad_corners(self):
        """
        Selects four corners from obj_pts/img_pts that correspond to the board's
        outer quadrilateral. Returns (quad_obj_pts, quad_img_pts), each shape (4, 2).
        """
        N = self.obj_pts.shape[0]
        if N < 4:
            raise ValueError("Need at least 4 points to form a quadrilateral")

        xs = self.obj_pts[:, 0]
        ys = self.obj_pts[:, 1]
        min_x, max_x = float(xs.min()), float(xs.max())
        min_y, max_y = float(ys.min()), float(ys.max())

        # Define the four ideal corner positions in object space:
        targets = [
            (min_x, min_y),  # top-left
            (max_x, min_y),  # top-right
            (max_x, max_y),  # bottom-right
            (min_x, max_y),  # bottom-left
        ]

        quad_obj = []
        quad_img = []
        used_indices = set()

        for tx, ty in targets:
            diffs = self.obj_pts - np.array([tx, ty], dtype=np.float32)
            d2 = np.sum(diffs**2, axis=1)  # squared distance to each obj_pt
            idx = int(np.argmin(d2))

            if idx in used_indices:
                # If already used, pick the next closest unused
                sorted_idxs = np.argsort(d2)
                for candidate in sorted_idxs:
                    if candidate not in used_indices:
                        idx = int(candidate)
                        break

            used_indices.add(idx)
            quad_obj.append(self.obj_pts[idx])
            quad_img.append(self.img_pts[idx])

        quad_obj = np.array(quad_obj, dtype=np.float32)  # shape (4,2)
        quad_img = np.array(quad_img, dtype=np.float32)  # shape (4,2)
        return quad_obj, quad_img

    def project_point(self, point, z=0.0):
        quad_obj, quad_img = self.get_quad_corners()
        H = cv2.getPerspectiveTransform(quad_img, quad_obj)
        pts = np.array([[[point[0], point[1]]]], dtype=np.float32)  # shape (1,1,2)
        projected = cv2.perspectiveTransform(pts, H)  # shape (1,1,2)
        X = float(projected[0, 0, 0])
        Y = float(projected[0, 0, 1])
        Z = z

        # Get camera position in board coordinates
        board_T = self.get_ref_T()
        # board_T transforms from board to camera, so invert to get camera pose in board frame
        cam_T_in_board = np.linalg.inv(board_T)
        cam_pos = cam_T_in_board[:3, 3]  # Camera position in board coordinates
        
        # Calculate angle between camera and point
        # Vector from camera to point (not point to camera)
        delta_x = X - cam_pos[0]
        delta_y = Y - cam_pos[1]
        delta_z = Z - cam_pos[2]
        
        # Angle in X axis: angle between projection onto YZ plane
        # atan2(delta_x, delta_z) gives angle from Z axis toward X
        angle_x = math.atan2(delta_x, delta_z)
        
        # Angle in Y axis: angle between projection onto XZ plane
        angle_y = math.atan2(delta_y, delta_z)
        
        # Apply parallax correction based on object height and viewing angles
        # The homography projects to Z=0, but object is at height Z
        # Offset is Z * tan(angle) in each axis
        offset_x = Z * math.tan(angle_x)
        offset_y = Z * math.tan(angle_y)
        
        # Correct the position
        X_corrected = X - offset_x
        Y_corrected = Y - offset_y

        return (X_corrected, Y_corrected)

class BoardEstimator:
    def __init__(self, board_config, K, D=None, rotate_180=True):
        """Initialize BoardEstimator.
        
        Args:
            board_config: Board configuration object
            K: Camera intrinsic matrix
            D: Distortion coefficients (default: zeros)
            rotate_180: Whether to rotate input frame 180° before processing (default: True)
        """
        self.config = board_config
        self.board = board_config.board
        self.detector = board_config.detector
        self.K = K
        self.D = D if D is not None else np.zeros(5)
        self.rotate_180 = rotate_180

    def get_board_transform(self, frame, drawing_frame=None):
        # Detect markers/corners using config's detect method (on original frame for drawing)
        corners, ids = self.config.detect_corners(frame, drawing_frame=drawing_frame)
        if ids is None:
            return None
        
        # For pose estimation, use the rotated corners if rotation is enabled
        if self.rotate_180:
            # Rotate corner coordinates 180 degrees around image center
            h, w = frame.shape[:2]
            cx, cy = w / 2, h / 2
            rotated_corners = []
            for corner_set in corners:
                # Each corner_set is shape (1, N, 2) where N is number of corners per marker
                rotated_set = corner_set.copy()
                for i in range(rotated_set.shape[1]):
                    x, y = rotated_set[0, i]
                    # Rotate 180 degrees around center
                    rotated_set[0, i, 0] = 2 * cx - x
                    rotated_set[0, i, 1] = 2 * cy - y
                rotated_corners.append(rotated_set)
            corners_for_pnp = rotated_corners
        else:
            corners_for_pnp = corners
        
        # Get board pose with centering offset applied before PnP
        res = get_board_pose(self.board, self.K, self.D, corners_for_pnp, ids, offset=self.config.center)
        if res is None:
            return None
        
        # Convert rvec, tvec to a 4x4 transformation matrix
        board_T = vecs_to_matrix(res.rvec, res.tvec)
        
        # Apply 180-degree rotation around X-axis to fix coordinate convention
        if self.rotate_180:
            R_x_180 = np.array([
                [1,  0,  0,  0],
                [0, -1,  0,  0],
                [0,  0, -1,  0],
                [0,  0,  0,  1]
            ], dtype=np.float64)
            board_T = board_T @ R_x_180

        # Display on the drawing frame
        if drawing_frame is not None:
            rvec, tvec = matrix_to_vecs(board_T)
            rvec_string = ', '.join([str(round(math.degrees(x), 3)) for x in rvec])
            tvec_string = ', '.join([str(round(float(x), 3)) for x in tvec])
            cv2.putText(drawing_frame, f"R: {rvec_string}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.putText(drawing_frame, f"T: {tvec_string}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        return board_T, res
    
    def project_point_to_board(self, pnp_result, image_point, frame_shape, z=0.0):
        """Project image point to board coordinates, handling rotation if enabled.
        
        Args:
            pnp_result: PnpResult from get_board_transform
            image_point: (x, y) tuple in original image coordinates
            frame_shape: (height, width) or (height, width, channels) of the frame
            
        Returns:
            (X, Y) in board coordinates
        """
        if self.rotate_180:
            h, w = frame_shape[:2]
            cx, cy = w / 2, h / 2
            x, y = image_point
            rotated_point = (2 * cx - x, 2 * cy - y)
            board_x, board_y = pnp_result.project_point(rotated_point, z=z)
            # Invert Y to match board coordinate convention (Y up vs image Y down)
            return (board_x, -board_y)
        else:
            return pnp_result.project_point(image_point, z=z)

def get_board_pose(
    board: cv2.aruco.Board,
    K: np.ndarray,
    D: np.ndarray,
    detected_corners: np.ndarray,
    detected_ids: np.ndarray,
    offset: np.ndarray = None
) -> PnpResult:
    """
    Estimate board pose for either CharucoBoard or GridBoard.
    
    Args:
        offset: Optional 3D offset to apply to object points before solving PnP.
                Use this to recenter the coordinate system (e.g., board center).
    """
    obj_pts, img_pts = board.matchImagePoints(detected_corners, detected_ids)
    if obj_pts is None or obj_pts.shape[0] < 6:
        return None

    # Apply offset to object points before solving PnP
    if offset is not None:
        obj_pts = obj_pts - offset

    success, rvec, tvec = cv2.solvePnP(
        obj_pts,
        img_pts,
        K,
        D,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None

    return PnpResult(obj_pts=obj_pts, img_pts=img_pts, tvec=tvec.flatten(), rvec=rvec.flatten())

def get_cam_T(ref_T: np.ndarray) -> np.ndarray:
    # Invert to get camera-to-reference
    cam_T = np.linalg.inv(ref_T)

    return cam_T

class BoardPlotter3D:
    """Real-time 3D visualization of board pose relative to camera.
    
    Can display either:
    - Camera at origin, board moving (camera_at_origin=True, default)
    - Board at origin, camera moving (camera_at_origin=False)
    """
    
    def __init__(self, board_config, axis_limit=1.0, update_interval=10, camera_at_origin=True):
        """Initialize 3D plotter.
        
        Args:
            board_config: BoardConfig instance to get board dimensions
            axis_limit: Axis limits in meters (default: 1.0m cube)
            update_interval: Update plot every N frames (default: 10)
            camera_at_origin: If True, camera at origin and board moves.
                            If False, board at origin and camera moves (default: True)
        """
        # Lazy import matplotlib only when plotter is created
        import matplotlib
        matplotlib.use('TkAgg')  # Use non-threaded backend
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        
        self.plt = plt
        self.Poly3DCollection = Poly3DCollection
        
        self.board_config = board_config
        self.axis_limit = axis_limit
        self.update_interval = update_interval
        self.camera_at_origin = camera_at_origin
        self.frame_count = 0
        
        # Get board dimensions
        self.board_width, self.board_height = board_config.get_board_dimensions()
        
        # Setup plot
        plt.ion()
        self.fig = plt.figure(figsize=(8, 6))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Initialize artists (will be updated)
        self.board_poly = None
        self.board_quivers = []
        self.camera_artists = []
        
        self._setup_plot()
        
        # Draw fixed reference frame based on mode
        if self.camera_at_origin:
            self._draw_camera_frame()
        else:
            self._draw_board_frame()
        
        # Initialize moving object artists
        self.board_poly = None
        self.board_quivers = []
        
        # Show initially
        self.plt.show(block=False)
        self.plt.pause(0.001)
        
    def _setup_plot(self):
        """Configure 3D axes and labels."""
        self.ax.set_xlim([-self.axis_limit, self.axis_limit])
        self.ax.set_ylim([-self.axis_limit, self.axis_limit])
        self.ax.set_zlim([0, 2 * self.axis_limit])
        
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('Board Pose Estimation')
        
        # Set viewing angle
        self.ax.view_init(elev=20, azim=45)
        
    def _draw_camera_frame(self):
        """Draw camera coordinate frame at origin (only once)."""
        axis_length = 0.2
        
        # Camera coordinate axes at origin
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2)
        )
    
    def _draw_board_frame(self):
        """Draw board coordinate frame and plane at origin (only once)."""
        axis_length = 0.2
        
        # Board coordinate axes at origin
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, axis_length, 0, 0, color='r', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, axis_length, 0, color='g', arrow_length_ratio=0.3, linewidth=2)
        )
        self.camera_artists.append(
            self.ax.quiver(0, 0, 0, 0, 0, axis_length, color='b', arrow_length_ratio=0.3, linewidth=2)
        )
        
        # Draw board plane at origin
        corners = self._get_board_corners()
        verts = [corners]
        board_poly = self.Poly3DCollection(verts, alpha=0.3, facecolor='gray', edgecolor='black', linewidth=2)
        self.ax.add_collection3d(board_poly)
        self.camera_artists.append(board_poly)
        
    def _get_board_corners(self):
        """Get board corners in board's local frame.
        
        Returns:
            np.ndarray: 4x3 array of corner positions (centered at origin)
        """
        w, h = self.board_width, self.board_height
        
        # Corners centered at origin, in XY plane (Z=0)
        corners = np.array([
            [-w/2, -h/2, 0],  # Bottom-left
            [ w/2, -h/2, 0],  # Bottom-right
            [ w/2,  h/2, 0],  # Top-right
            [-w/2,  h/2, 0],  # Top-left
        ])
        
        return corners
    
    def _transform_points(self, points, T):
        """Transform points by homogeneous transformation matrix.
        
        Args:
            points: Nx3 array of points
            T: 4x4 transformation matrix
            
        Returns:
            Nx3 array of transformed points
        """
        # Convert to homogeneous coordinates
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        
        # Apply transformation
        points_transformed = (T @ points_h.T).T
        
        # Convert back to 3D
        return points_transformed[:, :3]
    
    def update(self, board_T):
        """Update visualization with new board pose.
        
        Args:
            board_T: 4x4 homogeneous transformation matrix from camera to board
        """
        # Only update every N frames to reduce lag
        self.frame_count += 1
        if self.frame_count % self.update_interval != 0:
            return
        
        # If board is at origin, invert the transform to show camera moving
        if not self.camera_at_origin:
            board_T = np.linalg.inv(board_T)
        
        # Remove old visualization
        if self.board_poly is not None:
            self.board_poly.remove()
        for quiver in self.board_quivers:
            quiver.remove()
        self.board_quivers.clear()
        
        # Draw based on mode
        if self.camera_at_origin:
            # Draw board plane and axes at transformed position
            self._draw_moving_board(board_T)
        else:
            # Draw camera axes only (no plane) at transformed position
            self._draw_moving_camera(board_T)
        
        # Refresh display (non-blocking)
        self.fig.canvas.flush_events()
    
    def _draw_moving_board(self, board_T):
        """Draw board plane and coordinate frame at given transform."""
        # Get board corners and transform to camera frame
        corners_local = self._get_board_corners()
        corners_camera = self._transform_points(corners_local, board_T)
        
        # Draw board as filled polygon
        verts = [corners_camera]
        self.board_poly = self.Poly3DCollection(verts, alpha=0.5, facecolor='cyan', edgecolor='darkblue', linewidth=2)
        self.ax.add_collection3d(self.board_poly)
        
        # Draw board coordinate frame
        board_origin = board_T[:3, 3]
        axis_length = 0.15
        
        # Extract rotation axes from transformation matrix
        x_axis = board_T[:3, 0] * axis_length
        y_axis = board_T[:3, 1] * axis_length
        z_axis = board_T[:3, 2] * axis_length
        
        # Draw axes
        self.board_quivers.append(
            self.ax.quiver(board_origin[0], board_origin[1], board_origin[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(board_origin[0], board_origin[1], board_origin[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(board_origin[0], board_origin[1], board_origin[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
    
    def _draw_moving_camera(self, cam_T):
        """Draw camera coordinate frame (axes only) at given transform."""
        camera_origin = cam_T[:3, 3]
        axis_length = 0.15
        
        # Extract rotation axes from transformation matrix
        x_axis = cam_T[:3, 0] * axis_length
        y_axis = cam_T[:3, 1] * axis_length
        z_axis = cam_T[:3, 2] * axis_length
        
        # Draw axes
        self.board_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          x_axis[0], x_axis[1], x_axis[2],
                          color='r', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          y_axis[0], y_axis[1], y_axis[2],
                          color='g', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
        self.board_quivers.append(
            self.ax.quiver(camera_origin[0], camera_origin[1], camera_origin[2],
                          z_axis[0], z_axis[1], z_axis[2],
                          color='b', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.7)
        )
    
    def close(self):
        """Close the plot window."""
        self.plt.close(self.fig)

# if __name__ == "__main__":
#     import cv2
#     from board_config import gridboard_letter, gridboard_letter
#     from cam_config import global_cam, webcam
    
#     # board_config = global_board_config
#     board_config = gridboard_letter
#     active_cam = webcam

#     be = BoardEstimator(
#         board_config=board_config,
#         K=active_cam.K,
#         D=active_cam.D,
#         # rotate_180=False,
#     )
    
#     plotter = BoardPlotter3D(
#         board_config,
#         axis_limit=0.5,
#         # camera_at_origin=True,
#         camera_at_origin=False,
#     )
    
#     while True:
#         if cv2.waitKey(1) & 0xFF == 27:
#             break
    
#         # Get frame
#         frame = active_cam.get_frame()
#         drawing_frame = frame.copy()
    
#         # Estimate
#         res = be.get_board_transform(frame, drawing_frame=drawing_frame)
    
#         if res is not None:
#             board_T, _ = res
#             plotter.update(board_T)
    
#         # Display
#         cv2.imshow("Camera", drawing_frame)
#         cv2.setWindowProperty("Camera", cv2.WND_PROP_TOPMOST, 1)
