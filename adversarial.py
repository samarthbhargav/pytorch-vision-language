import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import utils.arg_parser
from models.model_loader import ModelLoader
from PIL import Image
from scipy.interpolate import interp2d
from torch.autograd import Variable
from train.trainer_loader import TrainerLoader
from utils.data.data_prep import DataPreparation
from utils.misc import get_split_str
from torchvision import transforms
from utils.transform import UnNormalize

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
split = 'test'
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


def norm_img(img):
    return transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))(img)
def transform_img(img):
    transform_center_crop = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    return transform_center_crop(img)

# The trainer already provides a method to extract an explanation
trainer_creator = getattr(TrainerLoader, args.model)
trainer = trainer_creator(args, model, dataset, data_loader, logger=None, device=device)

# Given an image id, retrieve image and label
# (assuming the image exists in the corresponding dataset!)
images_path = 'data/cub/images/'
img_ids = ('100.Brown_Pelican/Brown_Pelican_0122_94022.jpg',)
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

def forward(model, trainer, image_input, captions, labels, device = "cpu"):
    lengths = [len(cap) - 1 for cap in captions]
    word_inputs = torch.zeros(len(captions), max(lengths)).long()
    word_targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        word_inputs[i, :end] = cap[:-1]
        word_targets[i, :end] = cap[1:]

    labels_onehot = model.convert_onehot(labels)
    labels_onehot = labels_onehot.to(device)
    
    def get_feature_map(module, input, output):
        print("please fucking work")

    # layer = model.vision_model.pretrained_model.features._modules['0']
    # layer.register_forward_hook(get_feature_map) 
    features = get_vgg_features(image_input)
    features.retain_grad()
    
    model.zero_grad()

    outputs = model(
        features, word_inputs, lengths, labels, labels_onehot=labels_onehot
    )    

    # return trainer.criterion(
    #     outputs, word_targets.squeeze(0)
    # ), None
    
    # Generate explanation
    sample_ids, log_ps, lengths = model.generate_sentence(
        features, trainer.start_word, trainer.end_word, label, sample=True)
    
    explanation = " ".join(
        [dataset.vocab.get_word_from_idx(idx.item()) for idx in sample_ids]
    )
    
    #print(sample_ids.size(), log_ps.size(), lengths.size(), labels.size())
    sample_ids = sample_ids.unsqueeze(dim=0)
    log_ps = log_ps.unsqueeze(dim=0)
    # lengths = lengths.unsqueeze(dim=0)
    #labels = labels.unsqueeze(dim=0)
    #print(sample_ids.size(), log_ps.size(), lengths.size(), labels.size())

    lengths = lengths.cpu().numpy()
    sort_idx = np.argsort(-lengths)
    lengths = lengths[sort_idx]
    sort_idx = torch.tensor(sort_idx, device=device, dtype=torch.long)
    labels = labels[sort_idx]
    labels = labels.to(device)
    log_ps = log_ps[sort_idx, :]
    sample_ids = sample_ids[sort_idx, :]

    class_pred = model.sentence_classifier(sample_ids, lengths)
    class_pred = F.softmax(class_pred, dim=1)
    rewards = class_pred.gather(1, labels.view(-1, 1)).squeeze()
    r_loss = -(log_ps.sum(dim=1) * rewards).sum()

    loss = trainer.rl_lambda * r_loss / labels.size(0) + trainer.criterion(
        outputs, word_targets.squeeze(0)
    )

    return loss, explanation

def tensor_to_img(tensor):
    assert tensor.dim() == 3
    img = np.transpose(tensor.cpu().numpy(), (1,2,0))
    return img

for img_id in img_ids:
    raw_image = Image.open(os.path.join(images_path, img_id)).convert('RGB')
    image_input_no_norm = transform_img(raw_image)
    image_input = norm_img(image_input_no_norm)
    print(image_input_no_norm.max())
    label = dataset.get_class_label(img_id)

    image_input = image_input.unsqueeze(0)
    image_input.requires_grad = True
    
    #image_input.register_hook(process_grad)

    # print(image_input, image_input.requires_grad)
    labels = dataset.get_class_label(img_id)

    ann_id = random.choice(dataset.coco.imgToAnns[img_id])["id"]

    # base_id = dataset.ids[ann_id]
    tokens = dataset.tokens[ann_id]
    caption = []
    caption.append(dataset.vocab(dataset.vocab.start_token))
    caption.extend([dataset.vocab(token) for token in tokens])
    caption.append(dataset.vocab(dataset.vocab.end_token))
    captions = torch.Tensor([caption])
    
    loss, explanation = forward(model, trainer, image_input, captions, labels)
    loss.backward(retain_graph=True)
    print("Explanation:", explanation)

    # print(image_input.size(), image_input.requires_grad)
    # print(image_input, image_input.requires_grad)
    # print(image_input.grad.data)
    # print(image_input.grad.data.size())

    epsilon = 0.00001 
    x_grad  = torch.sign(image_input.grad.data)
    x_adversarial = torch.clamp(image_input_no_norm.data + epsilon * x_grad, 0, 1)
    x_adversarial_norm = norm_img(x_adversarial.squeeze(0)).unsqueeze(0)

    _, explanation_adv = forward(model, trainer, x_adversarial_norm, captions, labels)


    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


    print(x_adversarial.size(), image_input_no_norm.size())    
    #x_adv = transforms.ToPILImage()(x_adversarial.squeeze(0))
    x_adv = tensor_to_img(unnorm(x_adversarial_norm.squeeze(0)))
    x_org = tensor_to_img(unnorm(image_input.squeeze(0).detach()))

    # x_adv = 
    # x_org = 

    #x_org = transforms.ToPILImage()(image_input_no_norm)

    #print(x_adv.size(), x_org.size())
    print(x_adv.max())
    print(x_org.max())
    plt.subplot(1, 2, 1)
    plt.imshow(x_org)
    plt.title(explanation)
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_adv)
    plt.title(explanation_adv)
    plt.axis('off')
    plt.show()
    