import cv2
import numpy as np
import torch
from train_mask_rcnn import MaskRCNN

# Load the trained in-house Mask R-CNN model.
num_classes = 81
model = MaskRCNN(num_classes)
model.load_state_dict(torch.load("inhouse_mask_rcnn.pth"))
model.eval()

def inhouse_segment_objects(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            input_frame = cv2.resize(frame, (224, 224)).astype(np.float32) / 255.0
            input_frame = torch.tensor(input_frame).permute(2, 0, 1).unsqueeze(0)

            with torch.no_grad():
                cls_output, mask_output = model(input_frame)
                mask = (mask_output[0, 0] > 0.5).numpy()
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
                color_mask = np.zeros_like(frame)
                color_mask[:, :, 1] = mask * 255

                overlay = cv2.addWeighted(frame, 0.7, color_mask, 0.3, 0)
                cv2.imshow("Inhouse Segmentation", overlay)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print("In-house Segmentation complete.")

if __name__ == "__main__":
    input_video = 'input.mp4'  # Replace with your input video path.
    inhouse_segment_objects(input_video)
