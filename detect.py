import sys
import PIL as pil
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


scr_img_path = sys.argv[1]
dest_img_path = sys.argv[2]


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=True)
in_features=model.roi_heads.box_predictor.cls_score.in_features
num_classes=4
model.roi_heads.box_predictor =  FastRCNNPredictor(in_features,num_classes)
model

model.load_state_dict(torch.load("chkpt/model.pth", map_location=device))


class_colors = {
    1:['RBC', 'red', (80, 0, 0, 40)],
    2:['WBC', 'green', (0, 80, 0, 40)],
    3:['PLT', 'blue', (0, 0, 80, 40)],
}

def draw_box(img, boxes, labels):
    img = pil.Image.fromarray(img)
    draw = pil.ImageDraw.Draw(img, "RGBA")
    font = pil.ImageFont.truetype("NotoSans-VariableFont_wdth,wght.ttf", 12)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        label = label.cpu().numpy().item()
        name = class_colors[label][0]
        color = class_colors[label][1]
        fill_color = class_colors[label][2]
        draw.rectangle(list([x1, y1-20, x2, y1]), fill=color, outline=color)
        draw.text([(x1+x2)/2, (y1-20+y1)/2], name, 'black', anchor='mm', font=font)
        draw.rectangle(list([x1, y1, x2, y2]), fill=fill_color, outline=color)
    return img


def eval(threshold):
    model.to(device)
    model.eval()
    with torch.no_grad():
        img = torchvision.transforms.ToTensor()(pil.Image.open(scr_img_path).convert('RGB'))
        out = model([img.to(device)])
        keep = torchvision.ops.nms(out[0]['boxes'], out[0]['scores'], threshold)
        im = (img.permute(1,2,0).cpu().detach().numpy()*255).astype('uint8')
        img = draw_box(im, out[0]['boxes'][keep], out[0]['labels'][keep])
        img.save(dest_img_path)
        
        
eval(threshold=0.45)