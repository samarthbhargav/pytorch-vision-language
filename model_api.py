import sys

import torch
from models.model_loader import ModelLoader
from PIL import Image
from train.trainer_loader import TrainerLoader
from utils import arg_parser
from utils.data.data_prep import DataPreparation
from utils.misc import get_split_str
import torch.nn.functional as F
from attribute_chunker import AttributeChunker
from scipy.interpolate import interp2d

import numpy as np


def get_model():
    old_args = sys.argv[:]
    sys.argv = old_args[:1]
    # Get default arguments
    args = arg_parser.get_args()
    sys.argv = old_args

    args.model = "gve"
    args.dataset = "cub"
    args.pretrained_model = "vgg16"
    args.num_epochs = 1
    args.batch_size = 1
    # set to train because we need gradients for Grad-CAM
    args.train = True
    args.eval_ckpt = "data/vgg-ic-gve-best-ckpt.pth"
    args.ic_ckpt = "data/cub/image_classifier_ckpt.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preparation
    print("Preparing Data ...")
    split = get_split_str(args.train, bool(args.eval_ckpt), args.dataset)
    split = "test"
    data_prep = DataPreparation(args.dataset, args.data_path)
    dataset, data_loader = data_prep.get_dataset_and_loader(
        split,
        args.pretrained_model,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Load VGE model
    print("Loading Model ...")
    ml = ModelLoader(args, dataset, device)
    model = getattr(ml, args.model)()
    print(model, "\n")
    print("Loading Model Weights ...")
    evaluation_state_dict = torch.load(args.eval_ckpt, map_location="cpu")
    model_dict = model.state_dict(full_dict=True)
    model_dict.update(evaluation_state_dict)
    model.load_state_dict(model_dict)
    # Disable dropout and batch normalization
    model.eval()

    model.has_vision_model = False

    vgg_feat_layers = (
        model.image_classifier.vision_model.pretrained_model.features
    )
    vgg_class_layers = None

    visual = np.zeros((224, 224))

    trainer_creator = getattr(TrainerLoader, args.model)
    trainer = trainer_creator(
        args, model, dataset, data_loader, logger=None, device=device
    )


    return model, trainer, dataset, vgg_feat_layers

class ExplanationModel:
    def __init__(self):
        self.model, self.trainer, self.dataset, self.vgg_feat_layers = get_model()
        self.chunker = AttributeChunker()


    def get_features_labels(self, image_input, process_fmap_grad):
        # Forward pass until layer 28
        for i in range(29):
            image_input = self.vgg_feat_layers[i](image_input)
        features = image_input
        features.register_hook(process_fmap_grad)

        # Finish forward pass
        for i in range(29, len(self.vgg_feat_layers)):
            features = self.vgg_feat_layers[i](features)
        # Compact bilinear pooling
        features = self.model.image_classifier.cbp(features)
        # Element-wise signed square root layer and L2 normalization
        features = torch.sign(features) * torch.sqrt(torch.abs(features) + 1e-12)
        features = torch.nn.functional.normalize(features, dim=-1)

        logits = self.model.image_classifier.linear(features)
        _, labels = torch.max(logits.data, 1)

        return features, labels

    def generate(self, image, word_highlights=False, adversarial=False):

        # Grad-CAM
        def process_fmap_grad(grad):
            print("Called hook! Gradient has shape", grad.shape)
            # Extract single feature map gradient from batch
            fmap_grad = grad[0]
            # and compute global average
            a_k = fmap_grad.mean(dim=-1).mean(dim=-1)
            grad_cam = F.relu(
                torch.sum(a_k[:, None, None] * fmap_grad, dim=0)
            ).data.numpy()

            nx, ny = grad_cam.shape
            x = np.linspace(0, 224, nx, endpoint=False)
            y = np.linspace(0, 224, ny, endpoint=False)
            f = interp2d(x, y, grad_cam)
            xx = np.linspace(0, 224, 224, endpoint=False)
            yy = np.linspace(0, 224, 224, endpoint=False)
            visual[:] = f(xx, yy)

            print("Done")

        
        img_id = image["id"]
        raw_image = Image.open(image["path"])
        image_input = self.dataset.get_image(img_id).unsqueeze(dim=0)

        image_input.requires_grad = True

        
        # Get feature maps from the conv layer, and final features
        features, label = self.get_features_labels(image_input, process_fmap_grad)
        features.retain_grad()

        # Generate explanation
        outputs, log_probs = self.model.generate_sentence(
            features, self.trainer.start_word, self.trainer.end_word, label
        )
        explanation = " ".join(
            [self.dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1]
        )

        np_image = image_input.squeeze().permute(1, 2, 0).data.numpy()
        np_image = np_image - np.min(np_image)
        np_image = np_image * 255 / np.max(np_image)
        np_image = np_image.astype(np.uint8)

        word_masks = None
        
        if word_highlights:
            masks = np.zeros((224, 224, len(log_probs)))
            visual = np.zeros((224, 224))
            for i, log_p in enumerate(log_probs):
                self.model.zero_grad()
                log_probs[i].backward(retain_graph=True)
                masks[..., i] = visual
            mask_avg = np.mean(masks, axis=2)
            
            word_masks = {}
            final_masks = np.zeros((224, 224, len(log_probs)))
            for i, log_p in enumerate(log_probs):
                mask = masks[..., i] - mask_avg
                mask = np.clip(mask, 0, np.max(mask))
                mask = mask/np.max(mask)
                # Mask the image
                masked = (mask[..., np.newaxis] * np_image).astype(np.uint8)
                word = self.dataset.vocab.get_word_from_idx(outputs[i].item())
                word_masks[(i, word)] = masked
        
        return explanation, np_image, word_masks


class CounterFactualExplanationModel:
    def generate_counterfactual_explanation(self, image):
        return image["caption"]
