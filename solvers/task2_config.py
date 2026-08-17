"""MaixCAM Pro 拼图视觉参数。现场主要调整阈值和面积比例。"""

# Vendored from superiwan/2026_vision_v1@39c94d8 for task2_white.py.

# Camera / display
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 60
CAMERA_BUFFERS = 3
CAMERA_SKIP_FRAMES = 30
# Run one hands-free validation after deployment.  Set to False after the
# mounted work-mat calibration has been accepted for normal touch operation.
AUTO_START_ON_BOOT = True

# Green A4 detection is implemented by legacy_2026_new.py using green-channel
# dominance, saturation, and A4 geometry. These values document the shared
# field calibration used by that runtime.
PAPER_GREEN_MIN_G_MINUS_R = 12
PAPER_GREEN_MIN_G_MINUS_B = 6
PAPER_GREEN_MIN_SATURATION = 25
PAPER_ROI_MARGIN_X_RATIO = 0.04
PAPER_ROI_MARGIN_Y_RATIO = 0.01
PAPER_MIN_AREA_RATIO = 0.18
PAPER_MAX_AREA_RATIO = 0.75
PAPER_MIN_FILL_RATIO = 0.78
PAPER_MIN_OPPOSITE_SIMILARITY = 0.65
PAPER_MAX_CORNER_COSINE = 0.40
PAPER_ROI_BORDER_MARGIN_PX = 3
PAPER_CLOSE_KERNEL = 7
PAPER_CLOSE_ITERATIONS = 2
PAPER_OPEN_KERNEL = 3
PAPER_OPEN_ITERATIONS = 1
PAPER_BLUR_KERNEL = 5
PAPER_APPROX_EPSILON_RATIOS = (0.015, 0.02, 0.025, 0.03, 0.04, 0.05)
PAPER_A4_ASPECT_RATIO = 1.41421356
PAPER_ASPECT_REL_TOLERANCE = 0.35
PAPER_EDGE_REFINE_BAND_RATIO = 0.025
PAPER_EDGE_SAMPLE_OFFSET_RATIO = 0.006
PAPER_EDGE_MIN_CONTRAST = 12
PAPER_EDGE_MIN_POINTS_RATIO = 0.05
PAPER_EDGE_CANNY_LOW = 35
PAPER_EDGE_CANNY_HIGH = 110
PAPER_EDGE_REFINE_MIN_SHORT_SIDE = 720
PAPER_STABLE_FRAMES = 1
PAPER_STABLE_MAX_CORNER_SHIFT_PX = 8.0
PAPER_SEARCH_MAX_FRAMES = 1
PAPER_GUIDE_LANDSCAPE = True

# The competition rig uses a fixed overhead camera and a fixed green A4.
# Its corners are expressed as (x / frame_width, y / frame_height), ordered
# exactly as OpenCV's canonical plane: top-left, top-right, bottom-right,
# bottom-left. The active merged runtime uses automatic green-paper detection;
# these ratios remain available to the fixed-camera fallback code.
PAPER_FIXED_QUAD_RATIOS = (
    # Calibrated against the MaixCAM Pro's 640x480 preview.  The order is
    # top-left, top-right, bottom-right, bottom-left.
    (0.100, 0.531),
    (0.475, 0.531),
    (0.608, 0.867),
    (0.072, 0.854),
)
PAPER_FIXED_QUAD_AFTER_ATTEMPTS = 1
PAPER_FIXED_QUAD_MAX_BORDER_GRAY = 185
PAPER_FIXED_QUAD_MIN_DARK_BORDER_RATIO = 0.70

# Canonical portrait A4 plane: exactly 2 pixels/mm.
A4_WARP_WIDTH = 420
A4_WARP_HEIGHT = 594
A4_MM_PER_PIXEL = 0.5
A4_BORDER_INSET = 5

# White piece segmentation and polygon refinement, aligned with D:\26_new.
PIECE_GRAY_MIN = 165
PIECE_MIN_AREA_RATIO = 0.001
PIECE_MAX_AREA_RATIO = 0.25
PIECE_BORDER_REJECT_PX = 0
PIECE_APPROX_EPSILON_RATIOS = (0.012, 0.018, 0.025, 0.035, 0.05, 0.07)
PIECE_VERTEX_PENALTY = 0.0
MORPH_KERNEL = 3

# Edge matching / assembly. Algorithm 1 is the upstream v2.1 multi-topology
# graph solver (full edges, T-junction partial edges, concave pieces and equal
# rectangles). Algorithm 2 keeps the older iterative contour merger available
# for field comparison.
SOLVER_ALGORITHM = 1
# Shared contest deadline for all Task 2 and Task 3 search paths.
SOLVE_TIMEOUT_SECONDS = 80.0
EDGE_LENGTH_TOLERANCE = 0.15
MAX_EDGE_CANDIDATES = 80
PARTIAL_EDGE_MIN_RATIO = 0.22
PARTIAL_EDGE_MAX_RATIO = 0.88
PARTIAL_EDGE_PENALTY = 0.15
# The field-provided target rectangle may range from 9 x 5 cm to 12 x 9 cm.
TARGET_SHORT_SIDE_MM_RANGE = (50.0, 90.0)
TARGET_LONG_SIDE_MM_RANGE = (90.0, 120.0)
TARGET_DIMENSION_TOLERANCE_MM = 5.0
EQUAL_RECTANGLE_SIZE_TOLERANCE = 0.12
EQUAL_RECTANGLE_MIN_FILL = 0.985
FAST_SEARCH_FULL_CANDIDATES = 8
FAST_SEARCH_PARTIAL_CANDIDATES = 80
FAST_STANDARD_MAX_ANCHORS = 1
FAST_STANDARD_MAX_TOPOLOGIES = 2500
FAST_MULTI_PARTIAL_MAX_TOPOLOGIES = 3400
# Accept a candidate only after the final overlap/dimension gates below pass.
FAST_SEARCH_ACCEPT_FILL = 0.90
FAST_SEARCH_MIN_ROUGH_FILL = 0.85
MIN_RECTANGLE_FILL = 0.85
# Reject visually packed or heavily overlapping false assemblies.
MAX_RECTANGLE_OVERLAP_RATIO = 0.05
# Bounded fallback for small perspective/segmentation size errors. Each piece
# remains similar to its detected polygon and is scaled about its own centre.
DYNAMIC_SCALE_MIN = 0.95
DYNAMIC_SCALE_MAX = 1.05
DYNAMIC_SCALE_FULL_CANDIDATES = 12
DYNAMIC_SCALE_STANDARD_MAX_TOPOLOGIES = 2500
DYNAMIC_SCALE_MULTI_PARTIAL_MAX_TOPOLOGIES = 3400

# Algorithm 2: iterative contour merging.
MERGE_ANGLE_TOLERANCE_DEG = 12.0
MERGE_COLLINEAR_TOLERANCE_DEG = 12.0
MERGE_RECTANGLE_ANGLE_TOLERANCE_DEG = 15.0
MERGE_ENDPOINT_TOLERANCE_PX = 8.0
MERGE_MAX_AREA_ERROR_RATIO = 0.10
MERGE_MAX_OVERLAP_RATIO = 0.05
MERGE_MAX_STATES = 512

# Recovered rectangle position, relative to the green A4 bounding box.
TARGET_CENTER_X_RATIO = 0.50
TARGET_MARGIN_RATIO = 0.03
# Keep the assembled target close to, but fully below, the horizontal axis.
TARGET_AXIS_CLEARANCE_MM = 25.0
# Leave a physical safety gap between every placed piece.  This affects both
# the preview transforms and the UART place coordinates.
TARGET_PIECE_GAP_MM = 10

# Runtime UI and terminal telemetry
UI_HEADER_HEIGHT = 54
UI_BUTTON_HEIGHT = 64
UI_MARGIN = 16
PRINT_TIMING_EVERY_N_FRAMES = 60

# Overlay colours (BGR)
PAPER_COLOR = (255, 180, 0)
PIECE_COLORS = (
    (0, 220, 255),
    (80, 220, 80),
    (255, 140, 60),
    (220, 80, 220),
)
TARGET_COLOR = (0, 255, 0)
ERROR_COLOR = (0, 0, 255)
