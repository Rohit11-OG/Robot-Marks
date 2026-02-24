#!/usr/bin/env python3
"""
ArUco Marker Generator
Generates printable ArUco markers for robot tracking.

Usage:
    python generate_marker.py                    # Generate marker ID 0
    python generate_marker.py --id 5             # Generate marker ID 5
    python generate_marker.py --id 10 --size 300 # Generate 300x300 pixel marker
"""

import argparse
import os
import cv2
import cv2.aruco as aruco


def generate_marker(marker_id: int, size: int = 200, border_bits: int = 1) -> None:
    """
    Generate an ArUco marker and save it as PNG.
    
    Args:
        marker_id: ID of the marker (0-249 for DICT_6X6_250)
        size: Size in pixels (width = height)
        border_bits: White border around the marker
    """
    # Create output directory
    output_dir = "markers"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get the ArUco dictionary
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Generate the marker image
    marker_image = aruco.generateImageMarker(aruco_dict, marker_id, size)
    
    # Add white border for easier printing/cutting
    border_size = int(size * 0.1)  # 10% border
    bordered_image = cv2.copyMakeBorder(
        marker_image,
        border_size, border_size, border_size, border_size,
        cv2.BORDER_CONSTANT,
        value=255
    )
    
    # Save the marker
    output_path = os.path.join(output_dir, f"marker_{marker_id}.png")
    cv2.imwrite(output_path, bordered_image)
    
    print(f"✓ Generated ArUco marker ID {marker_id}")
    print(f"  Size: {bordered_image.shape[1]}x{bordered_image.shape[0]} pixels")
    print(f"  Saved to: {output_path}")
    print(f"\n  Print this marker and attach it to your robot!")
    print(f"  Recommended print size: 5-10 cm for good detection at 1-3 meters")


def generate_multiple_markers(count: int = 5, size: int = 200) -> None:
    """Generate multiple markers on a single sheet for printing."""
    output_dir = "markers"
    os.makedirs(output_dir, exist_ok=True)
    
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
    
    # Calculate grid layout
    cols = min(count, 3)
    rows = (count + cols - 1) // cols
    
    margin = 20
    cell_size = size + margin * 2
    sheet_width = cols * cell_size
    sheet_height = rows * cell_size
    
    # Create white sheet
    sheet = 255 * np.ones((sheet_height, sheet_width), dtype=np.uint8)
    
    for i in range(count):
        row = i // cols
        col = i % cols
        
        marker = aruco.generateImageMarker(aruco_dict, i, size)
        
        x = col * cell_size + margin
        y = row * cell_size + margin
        
        sheet[y:y+size, x:x+size] = marker
        
        # Add ID label
        cv2.putText(sheet, f"ID: {i}", (x, y + size + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, 0, 1)
    
    output_path = os.path.join(output_dir, f"marker_sheet_{count}.png")
    cv2.imwrite(output_path, sheet)
    
    print(f"✓ Generated marker sheet with {count} markers")
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate ArUco markers for robot tracking"
    )
    parser.add_argument(
        "--id", type=int, default=0,
        help="Marker ID to generate (0-249, default: 0)"
    )
    parser.add_argument(
        "--size", type=int, default=200,
        help="Marker size in pixels (default: 200)"
    )
    parser.add_argument(
        "--sheet", type=int, default=0,
        help="Generate a sheet with N markers (0 = single marker)"
    )
    
    args = parser.parse_args()
    
    if args.sheet > 0:
        generate_multiple_markers(args.sheet, args.size)
    else:
        if args.id < 0 or args.id > 249:
            print("Error: Marker ID must be between 0 and 249")
            return
        generate_marker(args.id, args.size)


if __name__ == "__main__":
    main()
