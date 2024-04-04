import os
import torch
import torch.nn.functional as F
from build.lib.models.vision_transformer import vit_huge
from build.lib.models.attentive_pooler import AttentiveClassifier
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
pretrained_path = "checkpoints/pretrained_models/vith16-384.pth.tar"
attentive_probe_path = "checkpoints/pretrained_models/k400-probe.pth.tar"

# Set the model and dataset parameters
model_name = "vit_huge"
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
checkpoint = torch.load(pretrained_path, map_location=device)
pretrained_dict = checkpoint["target_encoder"]
pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
encoder.load_state_dict(pretrained_dict)
encoder = ClipAggregation(
    encoder,
    tubelet_size=tubelet_size,
    # use_pos_embed=True,
    # attend_across_segments=True,
).to(device)
encoder.eval()

# Initialize the attentive classifier
classifier = AttentiveClassifier(
    embed_dim=encoder.embed_dim,
    num_heads=encoder.num_heads,
    depth=1,
    num_classes=num_classes,
).to(device)
attentive_probe_dict = torch.load(attentive_probe_path, map_location=device)[
    "classifier"
]
attentive_probe_dict = {
    k.replace("module.", ""): v for k, v in attentive_probe_dict.items()
}
classifier.load_state_dict(attentive_probe_dict)
classifier.eval()

# Load the class label mapping
label_map = {}
with open("checkpoints/pretrained_models/kinetics_400_labels.csv", "r") as f:
    csv_reader = csv.reader(f)
    next(csv_reader)  # Skip the header row
    for row in csv_reader:
        class_index, class_label = row
        label_map[int(class_index)] = class_label

# Define the video path
video_path = "calibration/playing_guitar.mp4"

# Define the transform pipeline
normalize = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize(size=int(crop_size * 256 / 224)),
        transforms.CenterCrop(size=crop_size),
        transforms.Normalize(mean=normalize[0], std=normalize[1]),
    ]
)

# Load the video and preprocess the frames
video = cv2.VideoCapture(video_path)
clips = []
frames = []

while True:
    ret, frame = video.read()
    if not ret:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = transform(frame)
    frames.append(frame)

    if len(frames) == frames_per_clip:
        clip = torch.stack(frames)
        clips.append(clip)
        frames = []

# Handle any remaining frames
if len(frames) > 0:
    remaining_frames = frames_per_clip - len(frames)
    for _ in range(remaining_frames):
        frames.append(frames[-1])  # Repeat the last frame
    clip = torch.stack(frames)
    clips.append(clip)

# Stack the clips into a single tensor
# clips = torch.stack(clips)

print("clips[0].shape", clips[0].shape)

re_clips = []
for clip in clips:
    B = 1  # Batch size of 1
    C, H, W = clip.shape[1], clip.shape[2], clip.shape[3]
    T = clip.shape[0]  # Number of frames
    re_clip = clip.view(B, C, T, H, W)  # Reshape to (B, C, T, H, W)
    re_clips.append(re_clip.to(device))

clips = re_clips

print("clips[0].shape", clips[0].shape)

clips = clips[:4]

results = []

# Perform inference on the video
with torch.no_grad():
    outputs = encoder([clips])
    # outputs = outputs[0]  # Extract the tensor from the list
    for outputs in outputs:
        for output in outputs:
            print("output.shape", output.shape)
            output = classifier(output)
            output = F.softmax(output, dim=1)
            results.append(output)

# Get the predicted class label

for result in results:
    print("result.shape", result.shape)
    _, predicted_class = torch.max(result, 1)
    predicted_label = label_map[predicted_class.item()]

    print("Predicted label:", predicted_label)
