import utils.arg_parser
import torch
from models.model_loader import ModelLoader
from train.trainer_loader import TrainerLoader
from utils.data.data_prep import DataPreparation
import utils.arg_parser
from utils.misc import get_split_str
from PIL import Image
import matplotlib.pyplot as plt
import os

# Get default arguments
args = utils.arg_parser.get_args()

# Overwrite required args
args.model = 'gve'
args.dataset = 'cub'
#args.pretrained_model = 'vgg16'
args.num_epochs = 1
args.batch_size = 1
# set to train because we need gradients for Grad-CAM
args.train = True
#args.eval_ckpt = 'data/vgg-vge-ckpt-e20.pth'
args.eval_ckpt = 'data/vge-best-ckpt.pth'

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

# The trainer already provides a method to extract an explanation
trainer_creator = getattr(TrainerLoader, args.model)
trainer = trainer_creator(args, model, dataset, data_loader, logger=None, device=device)

# Given an image id, retrieve image and label
# (assuming the image exists in the corresponding dataset!)
images_path = 'data/cub/images/'
img_ids = ('200.Common_Yellowthroat/Common_Yellowthroat_0040_190427.jpg',
    '046.Gadwall/Gadwall_0024_30942.jpg',
    '123.Henslow_Sparrow/Henslow_Sparrow_0010_796600.jpg',
    '191.Red_headed_Woodpecker/Red_Headed_Woodpecker_0039_183446.jpg',
    '044.Frigatebird/Frigatebird_0023_43110.jpg',
    '012.Yellow_headed_Blackbird/Yellow_Headed_Blackbird_0062_8310.jpg',
    '002.Laysan_Albatross/Laysan_Albatross_0005_565.jpg',
    '104.American_Pipit/American_Pipit_0121_100040.jpg',
    '122.Harris_Sparrow/Harris_Sparrow_0074_116539.jpg',
    '087.Mallard/Mallard_0044_76317.jpg',
    '135.Bank_Swallow/Bank_Swallow_0031_129507.jpg')

for img_id in img_ids:
    raw_image = Image.open(os.path.join(images_path, img_id))
    image_input = dataset.get_image(img_id).unsqueeze(dim=0)
    label = dataset.get_class_label(img_id)

    # Generate explanation (skip EOS)
    outputs, log_probs = model.generate_sentence(image_input, trainer.start_word, trainer.end_word, label)
    explanation = ' '.join([dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs[:-1]])

    # Plot results
    plt.figure(figsize=(15,15))
    plt.imshow(raw_image)
    plt.title(explanation)
    plt.axis('off')
    plt.show()


