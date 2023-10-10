"""
    Evaluate the model on the validation set
"""

import os
import cv2
import random
from ultralytics import YOLO

def pred_image(img_path, conf_thres = 0.25, model = 'model/v8n_10.pt'):
    # Random an image
    cls_path = os.path.join(img_path, random.choice(os.listdir(img_path)))
    img_path = os.path.join(cls_path, random.choice(os.listdir(cls_path)))
    img_path = os.path.join(img_path, random.choice(os.listdir(img_path)))
    
    # Load image
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load model
    model = YOLO(model)
    
    # Get prediction
    print('Processing...')
    result = model.predict(
        source = image,
        conf = conf_thres,
        save = False
    )
    print('Successfully!')
    
    # Show result
    thickness = 2
    for r in result:
        boxes = r.boxes
        for box in boxes:
            col = (0, 0, 255)
            b = box.xyxy[0]
            pt1, pt2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
            image = cv2.rectangle(image, pt1, pt2, col, thickness)
    
    # Display the image
    cv2.imshow('Image', image)
    cv2.waitKey()

def pred_video(video_path, conf_thres = 0.25, model = 'model/detect/v8n_20.pt'):
    # Random a video
    video_path = os.path.join(video_path, random.choice(os.listdir(video_path)))
    video_path = os.path.join(video_path, random.choice(os.listdir(video_path)))

    # Load video
    cap = cv2.VideoCapture(video_path)
    
    # Load model
    model = YOLO(model)
    
    count = -1
    nb_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    thickness = 2
    while True:
        # Read frame
        count += 1
        _, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        result = model.predict(
            source = frame,
            conf = conf_thres,
            save = False
        )
        
        # Show result
        for r in result:
            boxes = r.boxes
            for box in boxes:
                col = (0, 0, 255)
                b = box.xyxy[0]
                pt1, pt2 = (int(b[0]), int(b[1])), (int(b[2]), int(b[3]))
                frame = cv2.rectangle(frame, pt1, pt2, col, thickness)
        
        # Show frame
        cv2.imshow("Video", frame)
        if cv2.waitKey(10) == 27:
            break
        if count >= nb_frames - 1:
            break
    
    # Release the webcam and destroy all active window
    cap.release()
    cv2.destroyAllWindows()

def main(mode = 'image'):
    img_val_path = 'D:\Khanh\Others\Hand_Gesture\datasets\VinAI_INTERNAL\exp_20230623_recorded_frames\images'
    vid_val_path = 'D:\Khanh\Others\Hand_Gesture\datasets\VinAI_INTERNAL\exp_20230623_recorded_videos'
    
    if mode == 'image':
        pred_image(img_val_path, conf_thres = 0.3, model = 'model/v8n_20.pt')
    elif mode == 'video':
        pred_video(vid_val_path, conf_thres = 0.25, model = 'model/v8n_20.pt')
    else:
        print('Invalid mode!')

if __name__ == '__main__':
    main('video')
    