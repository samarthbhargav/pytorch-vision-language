import utils.arg_parser
import torch
import torch.nn.functional as F
from models.model_loader import ModelLoader
from train.trainer_loader import TrainerLoader
from utils.data.data_prep import DataPreparation
import utils.arg_parser
from utils.misc import get_split_str
from PIL import Image
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.interpolate import interp2d
from collections import namedtuple
import torchvision.transforms as transforms

# Read bounding boxes data
BBox = namedtuple('BBox', ['x', 'y', 'width', 'height'])
num2id = {}
with open('data/cub/CUB_200_2011/images.txt') as file:
    for line in file:
        values = line.split()
        num2id[values[0]] = values[1]
id2box = {}
with open('data/cub/CUB_200_2011/bounding_boxes.txt') as file:
    for line in file:
        values = line.split()
        id2box[num2id[values[0]]] = BBox(*[int(float(x)) for x in values[1:]])

# Get default arguments
args = utils.arg_parser.get_args()

# Overwrite required args
args.model = 'gve'
args.dataset = 'cub'
args.pretrained_model = 'vgg16'
args.num_epochs = 1
args.batch_size = 1
# set to train because we need gradients for Grad-CAM
args.train = True
args.eval_ckpt = 'data/vgg-ic-gve-best-ckpt.pth'
args.ic_ckpt = 'data/cub/image_classifier_ckpt.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation
print("Preparing Data ...")
split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
split = 'test'
data_prep = DataPreparation(args.dataset, args.data_path)
dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
        batch_size=args.batch_size, num_workers=args.num_workers)

# Load VGE model
print("Loading Model ...")
ml = ModelLoader(args, dataset, device)
model = getattr(ml, args.model)()
print(model, '\n')
print("Loading Model Weights ...")
evaluation_state_dict = torch.load(args.eval_ckpt, map_location='cpu')
model_dict = model.state_dict(full_dict=True)
model_dict.update(evaluation_state_dict)
model.load_state_dict(model_dict)
# Disable dropout and batch normalization
model.eval()
# The model actually has a vision model but we need to
# probe the feature extraction process
model.has_vision_model = False
vgg_feat_layers = model.image_classifier.vision_model.pretrained_model.features
vgg_class_layers = None

visual = np.zeros((224, 224))

# Grad-CAM
def process_fmap_grad(grad):
    print('Called hook! Gradient has shape', grad.shape)
    # Extract single feature map gradient from batch
    fmap_grad = grad[0]
    # and compute global average
    a_k = fmap_grad.mean(dim=-1).mean(dim=-1)
    grad_cam = F.relu(torch.sum(a_k[:, None, None] * fmap_grad, dim=0)).data.numpy()

    nx, ny = grad_cam.shape
    x = np.linspace(0, 224, nx, endpoint=False)
    y = np.linspace(0, 224, ny, endpoint=False)
    f = interp2d(x, y, grad_cam)
    xx = np.linspace(0, 224, 224, endpoint=False)
    yy = np.linspace(0, 224, 224, endpoint=False)
    visual[:] = f(xx, yy)

    print('Done')

def get_features_labels(image_input):
    # Forward pass until layer 28
    for i in range(29):
        image_input = vgg_feat_layers[i](image_input)
    features = image_input
    features.register_hook(process_fmap_grad)

    # Finish forward pass
    for i in range(29, len(vgg_feat_layers)):
        features = vgg_feat_layers[i](features)
    # Compact bilinear pooling
    features = model.image_classifier.cbp(features)
    # Element-wise signed square root layer and L2 normalization
    features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
    features = torch.nn.functional.normalize(features, dim=-1)

    logits = model.image_classifier.linear(features)
    _, labels = torch.max(logits.data, 1)

    return features, labels

# The trainer already provides a method to extract an explanation
trainer_creator = getattr(TrainerLoader, args.model)
trainer = trainer_creator(args, model, dataset, data_loader, logger=None, device=device)

transform = transforms.Compose([transforms.Resize(224),
                                transforms.CenterCrop(224),
                                transforms.ToTensor()])
images_path = 'data/cub/images/'
grad_fractions = []
for i, (image_input, word_inputs, word_targets, lengths, ids, *excess) in enumerate(data_loader):
    raw_image = Image.open(os.path.join(images_path, ids[0]))
    bbox_np = np.zeros((raw_image.size[1], raw_image.size[0], 3), dtype=np.uint8)
    bbox = id2box[ids[0]]
    bbox_np[bbox.y: bbox.y + bbox.height, bbox.x: bbox.x + bbox.width, :] = 255
    bbox_np = transform(Image.fromarray(bbox_np)).data.numpy().sum(axis=0)
    bbox_np[bbox_np > 0] = 1

    # Enable for Grad-CAM
    image_input.requires_grad = True

    # Get feature maps from the conv layer, and final features
    features, label = get_features_labels(image_input)
    features.retain_grad()
    # Generate explanation
    outputs, log_probs = model.generate_sentence(features, trainer.start_word, trainer.end_word, label)
    explanation = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1])

    # Plot results
    np_image = image_input.squeeze().permute(1, 2, 0).data.numpy()
    np_image = np_image - np.min(np_image)
    np_image = np_image * 255 / np.max(np_image)
    np_image = np_image.astype(np.uint8)
    #image = Image.fromarray(np_image)
    #plt.figure(figsize=(15, 15))
    #plt.imshow(image)
    #plt.contour(bbox_np)
    #plt.title(explanation)
    #plt.axis('off')
    #plt.show()

    masks = np.zeros((224, 224, len(log_probs)))
    visual = np.zeros((224, 224))
    model.zero_grad()
    log_probs.sum().backward()

    mask = visual
    mask = np.clip(mask, 0, np.max(mask))
    mask = mask/np.max(mask)
    # Mask the image
    masked = (mask[..., np.newaxis] * np_image).astype(np.uint8)
    #plt.figure(figsize=(15, 15))
    #plt.imshow(masked)
    #plt.contour(bbox_np)
    #plt.axis('off')

    fraction = 1 - np.sum(mask - (bbox_np * mask)) / mask.sum()
    grad_fractions.append(fraction)
    #plt.title('{:.1f}% of gradient within box'.format(percentage * 100))
    print('[{:d}/{:d}]'.format(i+1, len(dataset.coco.imgs)))
    print('mean = {:.6f}\nstd = {:.6f}'.format(np.mean(grad_fractions),
                                               np.std(grad_fractions)))

    #plt.show()
print('Gradient-to-box ratio')
print('mean = {:.6f}\nstd = {:.6f}'.format(np.mean(grad_fractions),
                                           np.std(grad_fractions)))

