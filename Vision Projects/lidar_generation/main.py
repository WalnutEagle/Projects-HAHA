import cv2
import numpy as np
from depth_estimation import DepthEstimator
from point_cloud import depth_to_point_cloud, visualize_point_cloud

def main(video_path):
    # Camera intrinsic parameters (example values, adjust as needed)
    intrinsic_matrix = np.array([[525.0, 0.0, 319.5],
                                  [0.0, 525.0, 239.5],
                                  [0.0, 0.0, 1.0]])

    # Initialize depth estimator
    depth_estimator = DepthEstimator()

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))
        depth_map = depth_estimator.estimate_depth(frame)

        # Convert depth map to point cloud
        points = depth_to_point_cloud(depth_map, intrinsic_matrix)

        # Visualize point cloud
        visualize_point_cloud(points)

    cap.release()

if __name__ == "__main__":
    video_path = "input_video.mp4"  # Replace with your video file path
    main(video_path)
