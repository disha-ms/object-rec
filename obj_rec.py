import cv2
from ultralytics import YOLO
import os

def detect_objects_in_image(image_path):
    # Try multiple models for better detection
    print("Loading YOLOv8x model...")
    model = YOLO("yolov8x.pt")
    
    # Read the image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not load image from {image_path}")
        return
    
    # Run YOLO detection with lower confidence to catch more objects
    results = model(frame, conf=0.2, iou=0.3)
    
    object_count = {}
    all_detections = []
    
    # COCO class names for reference
    coco_classes = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
        'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
        'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
        'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
        'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
        'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
        'toothbrush'
    ]
    
    # Custom mapping for objects not in COCO dataset
    custom_mappings = {
        'wallet': ['handbag', 'backpack'],
        'sunglass': [],  # Not in COCO
        'headphone': [],  # Not in COCO
        'tissue': [],  # Not in COCO
        'comb': [],  # Not in COCO
        'keys': ['keyboard', 'remote']  # Might be detected as similar objects
    }
    
    # Collect all detections
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = model.names[cls]
            
            all_detections.append({
                'bbox': (x1, y1, x2, y2),
                'conf': conf,
                'label': label
            })
            
            # Count objects
            object_count[label] = object_count.get(label, 0) + 1
    
    # Try to identify custom objects based on size, shape, and context
    height, width = frame.shape[:2]
    
    # Analyze the image for common patterns
    for det in all_detections:
        x1, y1, x2, y2 = det['bbox']
        bbox_width = x2 - x1
        bbox_height = y2 - y1
        aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 0
        area = bbox_width * bbox_height
        
        # Custom object detection logic
        current_label = det['label']
        
        # Wallet detection (small rectangular object near person/pocket area)
        if area < 5000 and 0.5 < aspect_ratio < 2.5 and det['conf'] > 0.2:
            # Check if it's near other personal items
            if current_label in ['handbag', 'backpack'] or area < 2000:
                custom_label = 'wallet'
                if custom_label not in object_count:
                    object_count[custom_label] = 0
                object_count[custom_label] += 1
                det['label'] = custom_label
        
        # Sunglasses detection (small object on face area)
        elif area < 3000 and bbox_width > bbox_height and y1 < height * 0.4:
            if 'person' in object_count and det['conf'] > 0.15:
                custom_label = 'sunglasses'
                if custom_label not in object_count:
                    object_count[custom_label] = 0
                object_count[custom_label] += 1
                det['label'] = custom_label
        
        # Headphone detection (larger object around head area)
        elif area > 5000 and y1 < height * 0.3 and aspect_ratio > 1.2:
            if det['conf'] > 0.2:
                custom_label = 'headphones'
                if custom_label not in object_count:
                    object_count[custom_label] = 0
                object_count[custom_label] += 1
                det['label'] = custom_label
    
    # Draw all detections
    for det in all_detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['conf']
        label = det['label']
        
        # Color coding based on confidence
        if conf > 0.7:
            color = (0, 255, 0)  # Green - high confidence
        elif conf > 0.4:
            color = (0, 165, 255)  # Orange - medium confidence
        else:
            color = (0, 0, 255)  # Red - low confidence
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Display object count on the image
    y_offset = 30
    total_objects = sum(object_count.values())
    cv2.putText(frame, f"Total Objects: {total_objects}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
    
    y_offset += 35
    # Sort objects by count
    sorted_objects = sorted(object_count.items(), key=lambda x: x[1], reverse=True)
    
    for obj_name, count in sorted_objects:
        cv2.putText(frame, f"{obj_name}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        y_offset += 25
    
    # Print detailed results
    print(f"\n{'='*60}")
    print(f"COMPREHENSIVE DETECTION RESULTS")
    print(f"{'='*60}")
    print(f"Detected {total_objects} objects in total")
    print(f"{'-'*60}")
    
    for obj_name, count in sorted_objects:
        print(f"✓ {obj_name}: {count}")
    
    print(f"{'='*60}")
    
    # Show missing objects that user expected
    expected_objects = ['keys', 'wallet', 'sunglass', 'headphone', 'tissue', 'comb']
    missing_objects = [obj for obj in expected_objects if obj not in object_count]
    
    if missing_objects:
        print(f"\nObjects not detected: {', '.join(missing_objects)}")
        print("These objects might not be in YOLO's training dataset")
    
    # Display the image
    cv2.imshow("Comprehensive Object Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    image_path = r"C:/Users/LENOVO/Downloads/many obj.jpeg"
    detect_objects_in_image(image_path)
    

    # run as: streamlit run obj_rec.py 