import os
import torch
import torch.nn.functional as F
from build.lib.models.vision_transformer import vit_huge
from build.lib.models.attentive_pooler import AttentiveClassifier
from build.lib.datasets.data_manager import init_data
from evals.video_classification_frozen.utils import make_transforms, ClipAggregation
import csv
import torchvision.transforms as transforms
import cv2

# import sys
# sys.exit(0)


# Set the device to use for inference
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
if torch.cuda.is_available():
    device = "cuda:0"
if torch.backends.mps.is_available():
    device = "mps"
print("Device:", device)

# Set the paths for the pre-trained model and test dataset
pretrained_path = 'checkpoints/pretrained_models/vith16-384.pth.tar'
attentive_probe_path = 'checkpoints/pretrained_models/k400-probe.pth.tar'
test_data_path = 'test.csv'

# Set the model and dataset parameters
model_name = 'vit_huge'
patch_size = 16
crop_size = 384
num_classes = 400
frames_per_clip = 16
tubelet_size = 2

# Initialize the pre-trained encoder
encoder = vit_huge(
    img_size=crop_size,
    num_frames=frames_per_clip,
    tubelet_size=tubelet_size,
)
encoder.to(device)
checkpoint = torch.load(pretrained_path, map_location='cpu')
pretrained_dict = checkpoint['target_encoder']
pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
encoder.load_state_dict(pretrained_dict)
encoder = ClipAggregation(encoder, tubelet_size=tubelet_size).to(device)
encoder.eval()

# Initialize the attentive classifier
classifier = AttentiveClassifier(
    embed_dim=encoder.embed_dim,
    num_heads=encoder.num_heads,
    depth=1,
    num_classes=num_classes,
).to(device)
attentive_probe_dict = torch.load(attentive_probe_path, map_location='cpu')['classifier']
attentive_probe_dict = {k.replace('module.', ''): v for k, v in attentive_probe_dict.items()}
classifier.load_state_dict(attentive_probe_dict)
classifier.eval()

# Create the data loader for the test dataset
transform = make_transforms(training=False, crop_size=crop_size)
test_loader, _ = init_data(
    data='VideoDataset',
    root_path=[test_data_path],
    transform=transform,
    batch_size=1,
    world_size=1,
    rank=0,
    clip_len=frames_per_clip,
    num_workers=0,
    training=False,
)

# Load the class label mapping
label_map = {}
with open('checkpoints/pretrained_models/kinetics_400_labels.csv', 'r') as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        class_index, class_label = row
        label_map[int(class_index)] = class_label

def extract_tensor(nested_list):
    if isinstance(nested_list, torch.Tensor):
        return nested_list
    elif isinstance(nested_list, list):
        return extract_tensor(nested_list[0])
    else:
        raise ValueError("Unexpected type in the nested list.")

# Perform inference on the test dataset
for data in test_loader:
    clips = [[dij.to(device) for dij in di] for di in data[0]]
    clip_indices = [d.to(device) for d in data[2]]

    with torch.no_grad():
        outputs = encoder(clips, clip_indices)
        outputs = extract_tensor(outputs)
        
        outputs = classifier(outputs)
        outputs = F.softmax(outputs, dim=1)

    # Get the predicted class index and probability
    pred_class_index = outputs.argmax().item()
    pred_class_prob = outputs.max().item()

    # Map the predicted class index to its corresponding label
    pred_class_label = label_map[pred_class_index]

    # Print the inference result with the class label
    print(f'Predicted class index: {pred_class_index}, Label: {pred_class_label}, Probability: {pred_class_prob:.4f}')
