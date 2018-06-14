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
args.eval_ckpt = 'data/vgg-vge-best-ckpt.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data preparation
print("Preparing Data ...")
split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
data_prep = DataPreparation(args.dataset, args.data_path)
dataset, data_loader = data_prep.get_dataset_and_loader(split, args.pretrained_model,
        batch_size=args.batch_size, num_workers=args.num_workers)

# Load VGE model
print("Loading Model ...")
ml = ModelLoader(args, dataset)
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
vgg_feat_layers = model.vision_model.pretrained_model.features
vgg_class_layers = model.vision_model.pretrained_model.classifier

grad_cam = np.zeros((14, 14))

# Grad-CAM
def process_fmap_grad(grad):
    print('Gradient has shape', grad.shape)
    # Extract single feature map gradient from batch
    fmap_grad = grad[0]
    # and compute global average
    a_k = fmap_grad.mean(dim=-1).mean(dim=-1)
    grad_cam[:] = F.relu(torch.sum(a_k[:, None, None] * fmap_grad, dim=0)).data.numpy()

    print('Done')

def get_vgg_features(image_input):
    # Forward pass until layer 28
    for i in range(29):
        image_input = vgg_feat_layers[i](image_input)
    features = image_input
    features.register_hook(process_fmap_grad)

    # Finish forward pass
    for i in range(29, len(vgg_feat_layers)):
        features = vgg_feat_layers[i](features)
    features = features.view(features.size(0), -1)
    for layer in vgg_class_layers:
        features = layer(features)

    return features

# The trainer already provides a method to extract an explanation
trainer_creator = getattr(TrainerLoader, args.model)
trainer = trainer_creator(args, model, dataset, data_loader, logger=None, device=device)

# Given an image id, retrieve image and label
# (assuming the image exists in the corresponding dataset!)
images_path = 'data/cub/images/'
img_ids = ('165.Chestnut_sided_Warbler/Chestnut_Sided_Warbler_0016_164060.jpg',)
    # '041.Scissor_tailed_Flycatcher/Scissor_Tailed_Flycatcher_0023_42117.jpg',
    # '151.Black_capped_Vireo/Black_Capped_Vireo_0043_797458.jpg',
    # '155.Warbling_Vireo/Warbling_Vireo_0030_158488.jpg',
    # '008.Rhinoceros_Auklet/Rhinoceros_Auklet_0030_797509.jpg',
    # '079.Belted_Kingfisher/Belted_Kingfisher_0105_70550.jpg',
    # '089.Hooded_Merganser/Hooded_Merganser_0049_79136.jpg',
    # '064.Ring_billed_Gull/Ring_Billed_Gull_0074_52258.jpg',
    # '098.Scott_Oriole/Scott_Oriole_0016_92398.jpg',
    # '013.Bobolink/Bobolink_0053_10166.jpg',
    # '003.Sooty_Albatross/Sooty_Albatross_0040_796375.jpg',
    # '026.Bronzed_Cowbird/Bronzed_Cowbird_0086_796259.jpg',
    # '092.Nighthawk/Nighthawk_0018_83639.jpg',
    # '035.Purple_Finch/Purple_Finch_0025_28174.jpg',
    # '037.Acadian_Flycatcher/Acadian_Flycatcher_0045_795587.jpg',
    # '066.Western_Gull/Western_Gull_0028_55680.jpg')

for img_id in img_ids:
    raw_image = Image.open(os.path.join(images_path, img_id))
    image_input = dataset.get_image(img_id).unsqueeze(dim=0)
    image_input.requires_grad = True
    label = dataset.get_class_label(img_id)

    # Get feature maps from the conv layer, and final features
    features = get_vgg_features(image_input)
    #image_input.retain_grad()
    # Generate explanation
    outputs, log_probs = model.generate_sentence(features, trainer.start_word, trainer.end_word, label)
    explanation = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1])

    # Plot results
    #plt.figure(figsize=(15, 15))
    #plt.imshow(raw_image)
    #plt.title(explanation)
    #plt.axis('off')
    #plt.show()
    # for i, log_p in enumerate(log_probs):
    #     model.zero_grad()
    #     log_probs[i].backward(retain_graph=True)
    #
    #     # Plot results
    #     plt.figure(figsize=(15,15))
    #     plt.subplot(1, 2, 1)
    #     plt.imshow(raw_image)
    #     plt.title(explanation)
    #     plt.axis('off')
    #     plt.subplot(1, 2, 2)
    #     plt.imshow(grad_cam)
    #     plt.title(dataset.vocab.get_word_from_idx(outputs[i].item()))
    #     plt.axis('off')
    #     plt.show()


