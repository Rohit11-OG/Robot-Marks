"""
Configuration settings for Robot Movement Monitor
Adjust these values to tune the movement detection sensitivity.
"""

# Camera Settings
CAMERA_ID = 0  # Default webcam (change to 1, 2, etc. for other cameras)
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 30

# ArUco Marker Settings
# Available: DICT_4X4_50, DICT_4X4_100, DICT_5X5_50, DICT_6X6_250, etc.
MARKER_DICT = "DICT_4X4_50"

# Movement Detection Thresholds
MOVEMENT_THRESHOLD = 5.0      # Minimum pixel movement to register as motion
VELOCITY_THRESHOLD = 15.0     # Pixels/second above this = definitely moving
STATIONARY_TIME = 1.5         # Seconds without movement to confirm stationary

# Position Tracking
HISTORY_SIZE = 60             # Number of positions to keep in history (for trail)
SMOOTHING_WINDOW = 5          # Frames to average for position smoothing

# Visual Display
SHOW_TRAIL = True             # Draw movement trail behind marker
TRAIL_LENGTH = 30             # Number of points in trail
SHOW_VELOCITY = True          # Display velocity on screen
SHOW_COORDINATES = True       # Display marker coordinates

# Logging
LOG_TO_FILE = False           # Save movement logs to file
LOG_DIR = "logs"              # Directory for log files

# Colors (BGR format)
COLOR_MOVING = (0, 165, 255)      # Orange
COLOR_STATIONARY = (0, 255, 0)    # Green
COLOR_LOST = (0, 0, 255)          # Red
COLOR_TRAIL = (255, 200, 100)     # Light blue
