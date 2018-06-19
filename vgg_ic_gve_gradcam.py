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

# Given an image id, retrieve image and label
# (assuming the image exists in the corresponding dataset!)
images_path = 'data/cub/images/'
img_ids = ('070.Green_Violetear/Green_Violetear_0072_60858.jpg',)
# img_ids = ('121.Grasshopper_Sparrow/Grasshopper_Sparrow_0078_116052.jpg'
#     '030.Fish_Crow/Fish_Crow_0073_25977.jpg',
#     '014.Indigo_Bunting/Indigo_Bunting_0027_11579.jpg',
#     '047.American_Goldfinch/American_Goldfinch_0040_32323.jpg',
#     '070.Green_Violetear/Green_Violetear_0072_60858.jpg',
#     '109.American_Redstart/American_Redstart_0126_103091.jpg',
#     '098.Scott_Oriole/Scott_Oriole_0017_795832.jpg',
#     '027.Shiny_Cowbird/Shiny_Cowbird_0043_796857.jpg',
#     '114.Black_throated_Sparrow/Black_Throated_Sparrow_0043_107236.jpg',
#     '003.Sooty_Albatross/Sooty_Albatross_0048_1130.jpg',
#     '127.Savannah_Sparrow/Savannah_Sparrow_0032_120109.jpg',
#     '177.Prothonotary_Warbler/Prothonotary_Warbler_0022_174138.jpg',
#     '131.Vesper_Sparrow/Vesper_Sparrow_0083_125718.jpg',
#     '026.Bronzed_Cowbird/Bronzed_Cowbird_0073_796226.jpg',
#     '177.Prothonotary_Warbler/Prothonotary_Warbler_0033_174123.jpg')

for img_id in img_ids:
    raw_image = Image.open(os.path.join(images_path, img_id))
    image_input = dataset.get_image(img_id).unsqueeze(dim=0)
    image_input.requires_grad = True
    #label = dataset.get_class_label(img_id)

    # Get feature maps from the conv layer, and final features
    features, label = get_features_labels(image_input)
    features.retain_grad()
    # Generate explanation
    outputs, log_probs = model.generate_sentence(features, trainer.start_word, trainer.end_word, label)
    explanation = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1])

    #log_probs[0].backward()
    #print('features.grad:\n', features.grad)
    #print('features.grad.sum():\n', features.grad.sum())
    #print('image_input.grad:\n', image_input.grad)
    #print('image_input.grad.sum():\n', image_input.grad.sum())

    # Plot results
    np_image = image_input.squeeze().permute(1, 2, 0).data.numpy()
    np_image = np_image - np.min(np_image)
    np_image = np_image * 255 / np.max(np_image)
    np_image = np_image.astype(np.uint8)
    image = Image.fromarray(np_image)
    plt.figure(figsize=(15, 15))
    plt.imshow(image)
    plt.title(explanation)
    plt.axis('off')
    plt.show()

    masks = np.zeros((224, 224, len(log_probs)))

    for i, log_p in enumerate(log_probs):
        model.zero_grad()
        log_probs[i].backward(retain_graph=True)

        # Plot results

        #plt.subplot(1, 2, 1)
        #plt.imshow(image)
        #plt.title(explanation)
        #plt.axis('off')
        #plt.subplot(1, 2, 2)
        # Scale grad-cam to the interval [0, 1]
        #visual_sc = visual / np.max(visual)
        masks[..., i] = visual#_sc**2

    mask_avg = np.mean(masks, axis=2)

    for i, log_p in enumerate(log_probs):
        mask = masks[..., i] - mask_avg
        mask = np.clip(mask, 0, np.max(mask))
        mask = mask/np.max(mask)
        # Mask the image
        masked = (mask[..., np.newaxis] * np_image).astype(np.uint8)
        plt.figure(figsize=(15, 15))
        plt.imshow(masked)
        word = dataset.vocab.get_word_from_idx(outputs[i].item())
        plt.title(word)
        plt.axis('off')
        plt.savefig('{:d}{:s}.png'.format(i+1, word))
