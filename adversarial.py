import random
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
img_ids = ('121.Grasshopper_Sparrow/Grasshopper_Sparrow_0078_116052.jpg',
    '030.Fish_Crow/Fish_Crow_0073_25977.jpg',
    '014.Indigo_Bunting/Indigo_Bunting_0027_11579.jpg',
    '047.American_Goldfinch/American_Goldfinch_0040_32323.jpg',
    '070.Green_Violetear/Green_Violetear_0072_60858.jpg',
    '109.American_Redstart/American_Redstart_0126_103091.jpg',
    '098.Scott_Oriole/Scott_Oriole_0017_795832.jpg',
    '027.Shiny_Cowbird/Shiny_Cowbird_0043_796857.jpg',
    '114.Black_throated_Sparrow/Black_Throated_Sparrow_0043_107236.jpg',
    '003.Sooty_Albatross/Sooty_Albatross_0048_1130.jpg',
    '127.Savannah_Sparrow/Savannah_Sparrow_0032_120109.jpg',
    '177.Prothonotary_Warbler/Prothonotary_Warbler_0022_174138.jpg',
    '131.Vesper_Sparrow/Vesper_Sparrow_0083_125718.jpg',
    '026.Bronzed_Cowbird/Bronzed_Cowbird_0073_796226.jpg',
    '177.Prothonotary_Warbler/Prothonotary_Warbler_0033_174123.jpg')

def forward(model, trainer, image_input, captions, labels, device = "cpu"):
    np.random.seed(42)
    torch.manual_seed(42)
    lengths = [len(cap) - 1 for cap in captions]
    word_inputs = torch.zeros(len(captions), max(lengths)).long()
    word_targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        word_inputs[i, :end] = cap[:-1]
        word_targets[i, :end] = cap[1:]

    labels_onehot = model.convert_onehot(labels)
    labels_onehot = labels_onehot.to(device)
   
    features, label = get_features_labels(image_input)
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


def norm_img(img):
    return transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                std=(0.229, 0.224, 0.225))(img)
def transform_img(img):
    transform_center_crop = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
    return transform_center_crop(img)

def tensor_to_img(tensor):
    assert tensor.dim() == 3
    img = np.transpose(tensor.cpu().numpy(), (1,2,0))
    return img


def do_the_thing():
    features, label = get_features_labels(image_input)
    features.retain_grad()
    
    outputs, log_probs = model.generate_sentence(features, trainer.start_word, trainer.end_word, label)
    explanation = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1])

    log_probs[4].backward(retain_graph=True)

    epsilon = 0.00000000001
    x_grad  = torch.sign(image_input.grad.data)
    x_adversarial = torch.clamp(image_input.data + epsilon * x_grad, 0, 1)
    #x_adversarial_norm = norm_img(x_adversarial.squeeze(0)).unsqueeze(0)
    x_adversarial.requires_grad = True

    model.zero_grad()

    features_adv, label_adv = get_features_labels(x_adversarial)
    features_adv.retain_grad()

    
    outputs_adv, log_probs_adv = model.generate_sentence(features_adv, trainer.start_word, trainer.end_word, label_adv)
    explanation_adv = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs_adv][:-1])

    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    x_adv = tensor_to_img(unnorm(x_adversarial.squeeze(0).detach()))
    x_org = tensor_to_img(unnorm(image_input.squeeze(0).detach()))

    plt.subplot(1, 2, 1)
    plt.imshow(x_org)
    plt.title("Original: {}".format(explanation))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_adv)
    plt.title("New: {}".format(explanation_adv))
    plt.axis('off')
    plt.show()
        

method = "per_word"
for img_id in img_ids:
    raw_image = Image.open(os.path.join(images_path, img_id)).convert('RGB')
    image_input = norm_img(transform_img(raw_image))
    image_input = image_input.unsqueeze(0)
    image_input.requires_grad = True
    
    label = dataset.get_class_label(img_id)
    labels = dataset.get_class_label(img_id)
    ann_id = random.choice(dataset.coco.imgToAnns[img_id])["id"]

    # base_id = dataset.ids[ann_id]
    tokens = dataset.tokens[ann_id]
    caption = []
    caption.append(dataset.vocab(dataset.vocab.start_token))
    caption.extend([dataset.vocab(token) for token in tokens])
    caption.append(dataset.vocab(dataset.vocab.end_token))
    captions = torch.Tensor([caption])


    model.zero_grad()

    if method == "per_word":
        features, label = get_features_labels(image_input)
        features.retain_grad()
        
        outputs, log_probs = model.generate_sentence(features, trainer.start_word, trainer.end_word, label)
        explanation = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1])

        log_probs[5].backward()
    else:    
        loss, explanation = forward(model, trainer, image_input, captions, labels, device)
        loss.backward(retain_graph=True)

    epsilon = 0.1
    x_grad  = torch.sign(image_input.grad.data)
    print(x_grad)
    print(x_grad.view(1, 1, -1).sum())
    x_adversarial = image_input.data + epsilon * x_grad
    #x_adversarial = norm_img(x_adversarial.squeeze(0)).unsqueeze(0)
    x_adversarial.requires_grad = True
    
    if method == "per_word":
        outputs_adv, _ = model.generate_sentence(features, trainer.start_word, trainer.end_word, label)
        explanation_adv = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs_adv][:-1])
    else:
        _, explanation_adv = forward(model, trainer, x_adversarial, captions, labels, device)
        
    print(explanation)
    print(explanation_adv)

    unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    x_adv = tensor_to_img(unnorm(x_adversarial.squeeze(0).detach()))
    x_org = tensor_to_img(unnorm(image_input.squeeze(0).detach()))

    plt.subplot(1, 2, 1)
    plt.imshow(x_org)
    plt.title("Original: {}".format(explanation))
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(x_adv)
    plt.title("New: {}".format(explanation_adv))
    plt.axis('off')
    plt.show()
   


    # print(x_adversarial.size(), image_input_no_norm.size())    
    # #x_adv = transforms.ToPILImage()(x_adversarial.squeeze(0))
    # 
    # 

    # # x_adv = 
    # # x_org = 

    # #x_org = transforms.ToPILImage()(image_input_no_norm)

    # #print(x_adv.size(), x_org.size())
    # print(x_adv.max())
    # print(x_org.max())
    # plt.subplot(1, 2, 1)
    # plt.imshow(x_org)
    # plt.title(explanation)
    # plt.axis('off')
    # plt.subplot(1, 2, 2)
    # plt.imshow(x_adv)
    # plt.title(explanation_adv)
    # plt.axis('off')
    # plt.show()
    