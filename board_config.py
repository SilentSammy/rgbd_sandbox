import cv2
import numpy as np

def image_to_pdf(img, filepath, physical_width):
    """Convert an image to PDF with exact physical dimensions.
    
    
    Args:
        img: OpenCV image (BGR format) or PIL Image
        filepath: Output PDF file path
        physical_width: Physical width in meters (including margins)
        margin: Margin in meters on each side (default: 0.0)
        
    Returns:
        str: Path to generated PDF file
    """

    from reportlab.pdfgen import canvas
    from reportlab.lib.units import cm
    from PIL import Image
    from reportlab.lib.utils import ImageReader
    
    margin = 0.0 # Deprecated parameter, always set to 0.0

    # Convert to PIL Image if it's an OpenCV image
    if isinstance(img, np.ndarray):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
    else:
        pil_img = img
    
    # Calculate physical height from image aspect ratio
    img_width, img_height = pil_img.size
    aspect_ratio = img_height / img_width
    physical_height = physical_width * aspect_ratio
    
    # Calculate image dimensions (subtract margins from both sides)
    img_physical_width = physical_width - (2 * margin)
    img_physical_height = img_physical_width * aspect_ratio
    
    # Calculate vertical margin to center the image
    # The image maintains its aspect ratio but is smaller, so we need to center it vertically
    vertical_margin = (physical_height - img_physical_height) / 2
    
    # Create PDF canvas sized to exact physical dimensions
    # Convert meters to cm for ReportLab
    width_cm = physical_width * 100
    height_cm = physical_height * 100
    margin_cm = margin * 100
    vertical_margin_cm = vertical_margin * 100
    
    c = canvas.Canvas(filepath, pagesize=(width_cm * cm, height_cm * cm))
    
    # Use ImageReader to wrap PIL image for ReportLab
    img_reader = ImageReader(pil_img)
    
    # Draw image with margins (offset by margin horizontally, centered vertically)
    c.drawImage(img_reader,
               x=margin_cm * cm,
               y=vertical_margin_cm * cm,
               width=img_physical_width * 100 * cm,
               height=img_physical_height * 100 * cm)
    
    c.showPage()
    c.save()
    
    return filepath

class BoardConfig:
    """Base class for board configurations."""
    
    def __init__(self, dictionary, board_width, print_width=None, filename="board"):
        """Initialize board configuration.
        
        Args:
            dictionary: ArUco dictionary
            board_width: Physical width of board content in meters
            print_width: Total width for printing in meters (includes margins). 
                        If None, defaults to board_width (no margins)
            filename: Base filename for saving/loading (no extension)
        """
        self.dictionary = dictionary
        self.board_width = board_width
        self.print_width = print_width if print_width is not None else board_width
        self.filename = filename
        self.board: cv2.aruco.Board = self._create_board()
        self.center = self._calculate_center()
        self.board_marker_ids = []  # Override in subclasses
    
    @property
    def image_path(self):
        """Get image filepath with .png extension."""
        return f"{self.filename}.png"
    
    @property
    def pdf_path(self):
        """Get PDF filepath with .pdf extension."""
        return f"{self.filename}.pdf"
    
    def _create_board(self):
        """Create and return the board object. Override in subclasses."""
        pass
    
    def _calculate_center(self):
        """Calculate and return the board's geometric center. Override in subclasses."""
        pass
    
    def detect_corners(self, frame, drawing_frame=None):
        """Detect corners/markers in frame. Override in subclasses."""
        pass
    
    def get_board_dimensions(self):
        """Return (width, height) of board in physical units. Override in subclasses."""
        pass
    
    def get_print_dimensions(self):
        """Return (width, height) for printing including margins.
        
        Returns:
            tuple: (print_width, print_height) in meters
        """
        board_width, board_height = self.get_board_dimensions()
        
        # Calculate margins in meters
        margin_m = (self.print_width - board_width) / 2
        
        # Total print height includes board height plus vertical margins
        print_height = board_height + (2 * margin_m)
        
        return self.print_width, print_height
    
    def generate_image(self, filepath=None, width_px=2160, marker_margin_px=0):
        """Generate board image in pixels with automatic margin calculation.
        
        Args:
            width_px: Width of the board content in pixels (not including margin)
            filepath: Optional path to save the image. If None, image is not saved.
            marker_margin_px: Transparent margin around markers in pixels (default: 4)
            
        Returns:
            Board image with margins based on print_width (BGRA with transparent background)
        """
        # Get board dimensions from subclass
        actual_width, actual_height = self.get_board_dimensions()
        
        # Calculate aspect ratio
        aspect_ratio = actual_height / actual_width
        height_px = int(width_px * aspect_ratio)
        
        # Calculate margin in meters and convert to pixels
        margin_m = (self.print_width - actual_width) / 2
        margin_px = int(margin_m * (width_px / actual_width))

        # Increase generation size so that after margin is applied,
        # the board content is exactly width_px × height_px
        adjusted_width = width_px + (2 * margin_px)
        adjusted_height = height_px + (2 * margin_px)
        
        # Generate board image with margin built-in
        img = self.board.generateImage((adjusted_width, adjusted_height), marginSize=margin_px, borderBits=1)
        
        # Create transparent PNG by masking out the white background
        # Check if image is already grayscale
        if len(img.shape) == 2 or img.shape[2] == 1:
            gray = img
        else:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Threshold to find black pixels (marker borders)
        _, black_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours of marker regions
        contours, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create alpha mask by filling all marker contours
        alpha = np.zeros_like(gray)
        cv2.drawContours(alpha, contours, -1, 255, -1)  # Fill all contours
        
        # Optionally dilate to add margin around markers
        if marker_margin_px > 0:
            # Each iteration expands by ~1-2 pixels, so use half the desired margin
            iterations = max(1, marker_margin_px // 2)
            kernel = np.ones((3, 3), np.uint8)
            alpha = cv2.dilate(alpha, kernel, iterations=iterations)
        
        # Add alpha channel to image
        if len(img.shape) == 2:
            # Grayscale image, convert to BGRA
            img_rgba = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        else:
            img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        img_rgba[:, :, 3] = alpha
        img = img_rgba
        
        # Save if filepath provided
        if filepath is not None:
            cv2.imwrite(filepath, img)
        
        return img

    def generate_pdf(self, img, filepath):
        """Generate PDF with board at exact physical dimensions.
        
        Args:
            img: Board image (from generate_image)
            filepath: Output PDF file path
            
        Returns:
            str: Path to generated PDF file
        """
        # Use helper function to create PDF (uses print_width for total page size)
        return image_to_pdf(img, filepath, self.print_width)

class GridboardConfig(BoardConfig):
    """Configuration for ArUco GridBoard with automatic marker separation calculation."""
    def __init__(self, dictionary, size, marker_length, board_width, print_width=None, filename="gridboard"):
        """Initialize GridBoard configuration with automatic marker separation.
        
        Args:
            dictionary: ArUco dictionary
            size: Grid size (cols, rows)
            marker_length: Length of each marker in meters
            board_width: Physical width of board content in meters
            print_width: Total width for printing in meters (includes margins)
            filename: Base filename for saving/loading (no extension)
        """
        self.size = size
        self.marker_length = marker_length
        
        # Calculate marker separation automatically
        # board_width = (n_markers * marker_length) + ((n_markers - 1) * separation)
        # separation = (board_width - n_markers * marker_length) / (n_markers - 1)
        cols, rows = size
        total_marker_width = cols * marker_length
        if cols > 1:
            self.marker_separation = (board_width - total_marker_width) / (cols - 1)
        else:
            self.marker_separation = 0.0
        
        # Call parent constructor
        super().__init__(dictionary, board_width, print_width, filename)
        
        # Set board marker IDs
        self.board_marker_ids = list(range(cols * rows))
        
        # Create detector (specific to GridBoard)
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)
    
    def _create_board(self):
        """Create and return the GridBoard."""
        return cv2.aruco.GridBoard(
            size=self.size,
            markerLength=self.marker_length,
            markerSeparation=self.marker_separation,
            dictionary=self.dictionary
        )
    
    def _calculate_center(self):
        """Calculate and return the board's geometric center."""
        cols, rows = self.size
        total_x = (cols * self.marker_length) + ((cols - 1) * self.marker_separation)
        total_y = (rows * self.marker_length) + ((rows - 1) * self.marker_separation)
        return np.array([total_x / 2.0, total_y / 2.0, 0.0], dtype=np.float64)
    
    def get_board_dimensions(self):
        """Return (width, height) of board in physical units."""
        cols, rows = self.size
        actual_width = cols * self.marker_length + (cols - 1) * self.marker_separation
        actual_height = rows * self.marker_length + (rows - 1) * self.marker_separation
        return actual_width, actual_height
    
    def detect_corners(self, frame, drawing_frame=None):
        """Detect markers in a frame and return corners and IDs.
        
        Args:
            frame: Input image/frame to detect markers in
            drawing_frame: Optional frame to draw detected markers on
            
        Returns:
            tuple: (corners, ids) where corners is a list of detected marker corners
                   and ids is an array of corresponding marker IDs
        """
        corners, ids, _ = self.detector.detectMarkers(frame)
        
        if drawing_frame is not None and ids is not None:
            cv2.aruco.drawDetectedMarkers(drawing_frame, corners, ids)
        
        return corners, ids

# TODO: Add CharucoBoardConfig similarly if needed

if __name__ == "__main__":
    gridboard_letter = GridboardConfig(
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
        size=(3, 4),
        marker_length=0.025,
        board_width=0.15,      # board content width (fits Letter height)
        print_width=0.2159,    # Letter width is 8.5" = 21.59cm
        filename="resources/gridboard_letter"
    )
    # Example usage: save board image and PDF
    img = gridboard_letter.generate_image(filepath=gridboard_letter.image_path, width_px=2160*4)
    gridboard_letter.generate_pdf(img, gridboard_letter.pdf_path)
    print(gridboard_letter.get_print_dimensions())  # Print dimensions including margins
