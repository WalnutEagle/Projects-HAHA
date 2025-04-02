import cv2
import numpy as np

def detect_road_path(input_video_path):
    """
    Detects road paths (lines) in a video and displays the processed video in real time.
    """
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
        
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale and apply Canny edge detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # Use Hough Transform to detect lines
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=50, maxLineGap=20)
            if lines is not None:
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # red line
            
            # Show frame in real time
            cv2.imshow('Road Path Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    input_video = 'input.mp4'  # Replace with your input video path
    detect_road_path(input_video)
    print("Road path detection complete.")
