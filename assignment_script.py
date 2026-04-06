import os
import xml.etree.ElementTree as ET
import cv2
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import Dataset, DataLoader

# Setup Device for Mac GPU (M1/M2/M3)
device = torch.device("cpu")
print(f"System Check: Running on {device}")

def parse_xml_annotations(xml_path):
    """Parses CVAT XML and returns normalized coordinates per frame."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        h_elem = root.find('.//original_size/height')
        w_elem = root.find('.//original_size/width')
        
        if h_elem is None or w_elem is None or h_elem.text is None or w_elem.text is None:
            return {}

    except Exception:
        return {}

    h, w = float(h_elem.text), float(w_elem.text)
    frame_boxes = {}

    for track in root.findall('track'):
        for box in track.findall('box'):
            if box.attrib.get('outside', '0') == '1':
                continue
            f_num = int(box.attrib['frame'])
            coords = [
                float(box.attrib['xtl'])/w, 
                float(box.attrib['ytl'])/h,
                float(box.attrib['xbr'])/w, 
                float(box.attrib['ybr'])/h
            ]
            if f_num not in frame_boxes:
                frame_boxes[f_num] = []
            frame_boxes[f_num].append(coords)
    return frame_boxes

class BaseballDataset(Dataset):
    def __init__(self, pairs, img_size=224):
        self.img_size = img_size
        self.samples = []
        for v_path, x_path in pairs:
            print(f"Loading Video: {os.path.basename(v_path)}")
            boxes = parse_xml_annotations(x_path)
            cap = cv2.VideoCapture(v_path)
            f_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if f_idx in boxes:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(frame_rgb, (img_size, img_size))
                    self.samples.append((img, boxes[f_idx]))
                f_idx += 1
            cap.release()
        print(f"Dataset Ready: {len(self.samples)} frames loaded.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, boxes = self.samples[idx]
        image = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
        p_boxes = [[b[0]*self.img_size, b[1]*self.img_size, b[2]*self.img_size, b[3]*self.img_size] for b in boxes]
        
        target = {
            "boxes": torch.tensor(p_boxes, dtype=torch.float32),
            "labels": torch.ones(len(p_boxes), dtype=torch.int64),
            "image_id": torch.tensor(idx),
            "area": torch.tensor([(b[2]-b[0])*(b[3]-b[1]) for b in p_boxes], dtype=torch.float32),
            "iscrowd": torch.zeros(len(p_boxes), dtype=torch.int64)
        }
        return image, target

def get_model(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features # type: ignore
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == "__main__":
    V_DIR, X_DIR = "videos/", "annotations/"
    video_files = sorted([f for f in os.listdir(V_DIR) if f.endswith('.mov')])
    pairs = [(os.path.join(V_DIR, f), os.path.join(X_DIR, f.replace('.mov', '.xml'))) for f in video_files]
    valid_pairs = [p for p in pairs if os.path.exists(p[1])]
    
    dataset = BaseballDataset(valid_pairs)
    train_size = int(0.8 * len(dataset))
    train_set, _ = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])
    
    def collate_fn(batch):
        return tuple(zip(*batch))

    loader = DataLoader(train_set, batch_size=4, shuffle=True, collate_fn=collate_fn, num_workers=0)
    
    model = get_model().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    print("\n--- Starting Training Loop ---")
    model.train()
    for epoch in range(3):
        epoch_loss = 0.0
        for i, (imgs, targs) in enumerate(loader):
            imgs = [img.to(device) for img in imgs]
            targs = [{k: v.to(device) for k, v in t.items()} for t in targs]
            
            loss_dict = model(imgs, targs)
            
            # Using torch.stack forces Pylance to see this as a Tensor, not an int
            losses: torch.Tensor = torch.stack(list(loss_dict.values())).sum()
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            current_loss_val = losses.item()
            epoch_loss += current_loss_val
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1} | Batch {i}/{len(loader)} | Loss: {current_loss_val:.4f}")

        print(f"✅ Epoch {epoch+1}/3 Complete - Avg Loss: {epoch_loss/len(loader):.4f}\n")

    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': 3,
        'img_size': 224,
        'num_classes': 2
    }, "baseball_model.pt")
    print("SUCCESS: baseball_model.pt saved.")