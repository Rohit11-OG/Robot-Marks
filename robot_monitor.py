#!/usr/bin/env python3
"""
Robot Movement Monitor
Continuously monitors a robot's movement using ArUco marker tracking.

The system classifies the robot's state as:
- MOVING: Robot is actively moving
- STATIONARY: Robot has been still for a defined period
- MARKER LOST: Cannot detect the ArUco marker

Usage:
    python robot_monitor.py              # Start monitoring
    python robot_monitor.py --camera 1   # Use camera ID 1
    
Controls:
    q - Quit
    r - Reset tracking history
    s - Save current frame
"""

import argparse
import time
import os
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, List

import cv2
import cv2.aruco as aruco
import numpy as np

# Try to import RealSense
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False

import config


class RobotState(Enum):
    """Robot movement states."""
    MOVING = "MOVING"
    STATIONARY = "STATIONARY"
    MARKER_LOST = "MARKER LOST"


@dataclass
class MarkerPosition:
    """Stores marker position data."""
    center: Tuple[float, float]
    corners: np.ndarray
    timestamp: float
    marker_id: int


class MovementTracker:
    """Tracks marker positions and calculates movement metrics."""
    
    def __init__(
        self,
        velocity_threshold: float = config.VELOCITY_THRESHOLD,
        stationary_time: float = config.STATIONARY_TIME,
        history_size: int = config.HISTORY_SIZE,
        smoothing_window: int = config.SMOOTHING_WINDOW
    ):
        self.velocity_threshold = velocity_threshold
        self.stationary_time = stationary_time
        self.history_size = history_size
        self.smoothing_window = smoothing_window
        
        self.position_history: deque = deque(maxlen=history_size)
        self.last_moving_time: float = time.time()
        self.current_state: RobotState = RobotState.MARKER_LOST
        self.current_velocity: float = 0.0
        self.time_stationary: float = 0.0
        
    def update(self, position: Optional[MarkerPosition]) -> RobotState:
        """
        Update tracker with new position and determine movement state.
        
        Args:
            position: Current marker position or None if not detected
            
        Returns:
            Current RobotState
        """
        current_time = time.time()
        
        if position is None:
            self.current_state = RobotState.MARKER_LOST
            self.current_velocity = 0.0
            return self.current_state
        
        self.position_history.append(position)
        
        # Need at least 2 positions to calculate velocity
        if len(self.position_history) < 2:
            self.current_state = RobotState.STATIONARY
            return self.current_state
        
        # Calculate smoothed velocity using recent positions
        velocities = []
        positions = list(self.position_history)
        
        for i in range(max(0, len(positions) - self.smoothing_window), len(positions) - 1):
            p1 = positions[i]
            p2 = positions[i + 1]
            
            dx = p2.center[0] - p1.center[0]
            dy = p2.center[1] - p1.center[1]
            distance = np.sqrt(dx**2 + dy**2)
            
            dt = p2.timestamp - p1.timestamp
            if dt > 0:
                velocities.append(distance / dt)
        
        if velocities:
            self.current_velocity = np.mean(velocities)
        else:
            self.current_velocity = 0.0
        
        # Determine state based on velocity
        if self.current_velocity > self.velocity_threshold:
            self.last_moving_time = current_time
            self.time_stationary = 0.0
            self.current_state = RobotState.MOVING
        else:
            self.time_stationary = current_time - self.last_moving_time
            if self.time_stationary >= self.stationary_time:
                self.current_state = RobotState.STATIONARY
            else:
                # Transitioning - still considered moving
                self.current_state = RobotState.MOVING
        
        return self.current_state
    
    def get_trail(self, length: int = config.TRAIL_LENGTH) -> List[Tuple[int, int]]:
        """Get recent positions for drawing movement trail."""
        positions = list(self.position_history)[-length:]
        return [(int(p.center[0]), int(p.center[1])) for p in positions]
    
    def reset(self):
        """Reset tracking history."""
        self.position_history.clear()
        self.last_moving_time = time.time()
        self.current_state = RobotState.MARKER_LOST
        self.current_velocity = 0.0
        self.time_stationary = 0.0


class RobotMonitor:
    """Main robot monitoring system using ArUco markers."""
    
    def __init__(self, camera_id: int = config.CAMERA_ID, use_realsense: bool = True):
        self.camera_id = camera_id
        self.use_realsense = use_realsense and REALSENSE_AVAILABLE
        self.cap: Optional[cv2.VideoCapture] = None
        self.pipeline = None  # RealSense pipeline
        self.tracker = MovementTracker()
        
        # Initialize ArUco detector (modern OpenCV 4.7+ API)
        dict_name = getattr(aruco, config.MARKER_DICT)
        self.aruco_dict = aruco.getPredefinedDictionary(dict_name)
        self.detector_params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, self.detector_params)
        
        # For FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        
        # Logging
        if config.LOG_TO_FILE:
            os.makedirs(config.LOG_DIR, exist_ok=True)
            self.log_file = open(
                os.path.join(config.LOG_DIR, f"monitor_{int(time.time())}.log"),
                "w"
            )
        else:
            self.log_file = None
    
    def initialize_camera(self) -> bool:
        """Initialize RealSense camera (RealSense only - no webcam fallback)."""
        
        if not REALSENSE_AVAILABLE:
            print("Error: pyrealsense2 not installed!")
            print("Install with: pip install pyrealsense2")
            return False
        
        try:
            self.pipeline = rs.pipeline()
            rs_config = rs.config()
            
            # Enable color stream - RealSense D435 supports 60fps at 640x480
            rs_config.enable_stream(
                rs.stream.color, 
                config.CAMERA_WIDTH, 
                config.CAMERA_HEIGHT, 
                rs.format.bgr8, 
                config.CAMERA_FPS
            )
            
            # Start streaming
            profile = self.pipeline.start(rs_config)
            
            # Get actual resolution
            color_profile = profile.get_stream(rs.stream.color)
            intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            
            print(f"✓ RealSense camera initialized: {intrinsics.width}x{intrinsics.height} @ {config.CAMERA_FPS}fps")
            return True
            
        except Exception as e:
            print(f"Error: RealSense initialization failed: {e}")
            print("\nTroubleshooting:")
            print("  1. Check USB connection (use USB 3.0 port)")
            print("  2. Run: realsense-viewer to verify camera works")
            print("  3. Try: sudo dmesg | tail -20  to check for USB errors")
            return False
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get a frame from RealSense camera."""
        if not self.pipeline:
            return False, None
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=5000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            frame = np.asanyarray(color_frame.get_data())
            return True, frame
        except Exception as e:
            print(f"RealSense frame error: {e}")
            return False, None
    
    def detect_marker(self, frame: np.ndarray) -> Optional[MarkerPosition]:
        """
        Detect ArUco markers in the frame.
        
        Returns the first detected marker's position, or None if not found.
        """
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        if ids is None or len(ids) == 0:
            return None
        
        # Use the first detected marker
        marker_corners = corners[0][0]
        marker_id = ids[0][0]
        
        # Calculate center point
        center_x = np.mean(marker_corners[:, 0])
        center_y = np.mean(marker_corners[:, 1])
        
        return MarkerPosition(
            center=(center_x, center_y),
            corners=marker_corners,
            timestamp=time.time(),
            marker_id=marker_id
        )
    
    def draw_overlay(
        self,
        frame: np.ndarray,
        position: Optional[MarkerPosition],
        state: RobotState
    ) -> np.ndarray:
        """Draw visual overlay on the frame."""
        overlay = frame.copy()
        
        # Select color based on state
        if state == RobotState.MOVING:
            color = config.COLOR_MOVING
            status_text = "MOVING"
        elif state == RobotState.STATIONARY:
            color = config.COLOR_STATIONARY
            status_text = "STATIONARY"
        else:
            color = config.COLOR_LOST
            status_text = "MARKER LOST"
        
        # Draw status box at top
        cv2.rectangle(overlay, (0, 0), (350, 120), (40, 40, 40), -1)
        cv2.rectangle(overlay, (0, 0), (350, 120), color, 2)
        
        # Status text
        cv2.putText(
            overlay, status_text, (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3
        )
        
        # Velocity
        if config.SHOW_VELOCITY:
            vel_text = f"Velocity: {self.tracker.current_velocity:.1f} px/s"
            cv2.putText(
                overlay, vel_text, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
            )
        
        # FPS
        cv2.putText(
            overlay, f"FPS: {self.current_fps:.1f}", (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        if position is not None:
            # Draw marker bounding box
            corners_int = position.corners.astype(int)
            cv2.polylines(overlay, [corners_int], True, color, 3)
            
            # Draw center point
            center = (int(position.center[0]), int(position.center[1]))
            cv2.circle(overlay, center, 8, color, -1)
            cv2.circle(overlay, center, 12, (255, 255, 255), 2)
            
            # Draw marker ID
            cv2.putText(
                overlay, f"ID: {position.marker_id}",
                (corners_int[0][0], corners_int[0][1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )
            
            # Draw coordinates
            if config.SHOW_COORDINATES:
                coord_text = f"({int(position.center[0])}, {int(position.center[1])})"
                cv2.putText(
                    overlay, coord_text,
                    (center[0] + 20, center[1] + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
                )
            
            # Draw movement trail
            if config.SHOW_TRAIL:
                trail = self.tracker.get_trail()
                for i in range(1, len(trail)):
                    alpha = i / len(trail)
                    thickness = max(1, int(3 * alpha))
                    cv2.line(
                        overlay, trail[i-1], trail[i],
                        config.COLOR_TRAIL, thickness
                    )
        
        # Add instruction text at bottom
        h = overlay.shape[0]
        cv2.putText(
            overlay, "Press 'q' to quit | 'r' to reset | 's' to save frame",
            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        return overlay
    
    def log_state(self, position: Optional[MarkerPosition], state: RobotState):
        """Log the current state."""
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if position:
            msg = f"[{timestamp}] {state.value} | ID:{position.marker_id} | " \
                  f"Pos:({int(position.center[0])},{int(position.center[1])}) | " \
                  f"Vel:{self.tracker.current_velocity:.1f}px/s"
        else:
            msg = f"[{timestamp}] {state.value}"
        
        # Only log state changes or every 30 frames
        if self.frame_count % 30 == 0 or state == RobotState.MARKER_LOST:
            print(msg)
            
            if self.log_file:
                self.log_file.write(msg + "\n")
                self.log_file.flush()
    
    def update_fps(self):
        """Update FPS calculation."""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Main monitoring loop."""
        if not self.initialize_camera():
            return
        
        print("\n" + "="*50)
        print("   ROBOT MOVEMENT MONITOR")
        print("   ArUco Marker Tracking System")
        print("="*50)
        print("\nLooking for ArUco markers (DICT_6X6_250)...")
        print("Show a marker to the camera to begin tracking.\n")
        
        try:
            while True:
                ret, frame = self.get_frame()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Detect marker
                position = self.detect_marker(frame)
                
                # Update tracker and get state
                state = self.tracker.update(position)
                
                # Draw visualization
                display_frame = self.draw_overlay(frame, position, state)
                
                # Log state
                self.log_state(position, state)
                
                # Update FPS
                self.update_fps()
                
                # Show frame
                cv2.imshow("Robot Movement Monitor", display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nExiting...")
                    break
                elif key == ord('r'):
                    self.tracker.reset()
                    print("Tracking history reset")
                elif key == ord('s'):
                    filename = f"frame_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(f"Saved: {filename}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.pipeline:
            self.pipeline.stop()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        
        if self.log_file:
            self.log_file.close()
        
        print("Monitor stopped.")


def main():
    parser = argparse.ArgumentParser(
        description="Robot Movement Monitor using ArUco markers"
    )
    parser.add_argument(
        "--camera", type=int, default=config.CAMERA_ID,
        help=f"Camera ID to use (default: {config.CAMERA_ID})"
    )
    
    args = parser.parse_args()
    
    monitor = RobotMonitor(camera_id=args.camera)
    monitor.run()


if __name__ == "__main__":
    main()
