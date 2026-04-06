import torch
import cv2
import os
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    # type: ignore allows Pylance to ignore the dynamic attribute check
    in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def run_inference(model_path, video_path):
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    model = get_model(num_classes=checkpoint['num_classes'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if ret:
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(frame_rgb, (224, 224))
        img_t = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
        
        with torch.no_grad():
            preds = model(img_t.to(device))[0]
        
        mask = preds['scores'] > 0.5
        count = len(preds['boxes'][mask])
        
        print("-" * 30)
        print("TEST SUCCESSFUL")
        print(f"Video Tested: {os.path.basename(video_path)}")
        print(f"Baseballs Found: {count}")
        if count > 0:
            print(f"Top Score: {preds['scores'][0].item():.4f}")
        print("-" * 30)
    else:
        print("Error: Video file not found.")

if __name__ == "__main__":
    if os.path.exists("baseball_model.pt"):
        run_inference("baseball_model.pt", "videos/IMG_8923_souleymane.mov")
    else:
        print("Error: No model found. Run assignment_script.py first.")