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
import random
import numpy as np
from utils.transform import UnNormalize


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

def tensor_to_img(tensor):
    assert tensor.dim() == 3
    np_image = np.transpose(tensor.cpu().numpy(), (1,2,0))
    np_image = np_image - np.min(np_image)
    np_image = np_image * 255 / np.max(np_image)
    np_image = np_image.astype(np.uint8)
    return np_image

class ExplanationModel:
    def __init__(self):
        self.model, self.trainer, self.dataset, self.vgg_feat_layers = get_model()
        self.chunker = AttributeChunker()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    def compute_full_loss(self, model, trainer, image_input, captions, labels, device = "cpu"):
        def process_fmap_grad(grad):
            pass
        np.random.seed(42)
        torch.manual_seed(42)  
        lengths = [len(cap) - 1 for cap in captions]
        word_inputs = torch.zeros(len(captions), max(lengths)).long()
        word_targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            word_inputs[i, :end] = cap[:-1]
            word_targets[i, :end] = cap[1:]

        labels_onehot = self.model.convert_onehot(labels)
        labels_onehot = labels_onehot.to(device)
    
        features, label = self.get_features_labels(image_input, process_fmap_grad)
        features.retain_grad()
        
        self.model.zero_grad()

        outputs = self.model(
            features, word_inputs, lengths, labels, labels_onehot=labels_onehot
        )    

        # Generate explanation
        sample_ids, log_ps, lengths = self.model.generate_sentence(
            features, self.trainer.start_word, self.trainer.end_word, label, sample=True)
        
        explanation = " ".join(
            [self.dataset.vocab.get_word_from_idx(idx.item()) for idx in sample_ids]
        )
        
        sample_ids = sample_ids.unsqueeze(dim=0)
        log_ps = log_ps.unsqueeze(dim=0)

        lengths = lengths.cpu().numpy()
        sort_idx = np.argsort(-lengths)
        lengths = lengths[sort_idx]
        sort_idx = torch.tensor(sort_idx, device=device, dtype=torch.long)
        labels = labels[sort_idx]
        labels = labels.to(device)
        log_ps = log_ps[sort_idx, :]
        sample_ids = sample_ids[sort_idx, :]

        class_pred = self.model.sentence_classifier(sample_ids, lengths)
        class_pred = F.softmax(class_pred, dim=1)
        rewards = class_pred.gather(1, labels.view(-1, 1)).squeeze()
        r_loss = -(log_ps.sum(dim=1) * rewards).sum()

        loss = self.trainer.rl_lambda * r_loss / labels.size(0) + self.trainer.criterion(
            outputs, word_targets.squeeze(0)
        )

        return loss, explanation

    def get_img(self, img_id):
        return tensor_to_img(self.dataset.get_image(img_id))

    def generate_adversarial(self, img_id, epsilon = 0.1, word_index=None):
        
        image_input = self.dataset.get_image(img_id).unsqueeze(dim=0)
        image_input.requires_grad = True

        label = self.dataset.get_class_label(img_id)
        labels = self.dataset.get_class_label(img_id)
        ann_id = random.choice(self.dataset.coco.imgToAnns[img_id])["id"]
        

        if word_index:
            self.model.zero_grad()
            features, label = self.get_features_labels(image_input)
            features.retain_grad()
        
            outputs, log_probs = self.model.generate_sentence(features, self.trainer.start_word, self.trainer.end_word, label)
            explanation = ' '.join([self.dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs][:-1])

            log_probs[word_index].backward()
        else:
            tokens = self.dataset.tokens[ann_id]
            caption = []
            caption.append(self.dataset.vocab(self.dataset.vocab.start_token))
            caption.extend([self.dataset.vocab(token) for token in tokens])
            caption.append(self.dataset.vocab(self.dataset.vocab.end_token))
            captions = torch.Tensor([caption])
    
            loss, explanation = self.compute_full_loss(self.model, self.trainer, image_input, captions, labels, self.device)
            loss.backward(retain_graph=True)
        
        x_grad  = torch.sign(image_input.grad.data)
        x_adversarial = image_input.data + epsilon * x_grad
        x_adversarial.requires_grad = True
        
        if word_index:
            outputs_adv, _ = self.model.generate_sentence(features, self.trainer.start_word, self.trainer.end_word, label)
            explanation_adv = ' '.join([self.dataset.vocab.get_word_from_idx(idx.item()) for idx in outputs_adv][:-1])
        else:
            _, explanation_adv = self.compute_full_loss(self.model, self.trainer, x_adversarial, captions, labels, self.device)
            
        unnorm = UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        #x_adv = tensor_to_img(unnorm(x_adversarial.squeeze(0).detach()))
        #x_org = tensor_to_img(unnorm(image_input.squeeze(0).detach()))
        x_adv = tensor_to_img(x_adversarial.squeeze(0).detach())
        x_org = tensor_to_img(image_input.squeeze(0).detach())

        return explanation, x_org, explanation_adv, x_adv
    
    def generate(self, image, word_highlights=False):
        np.random.seed(42)
        torch.manual_seed(42)
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
            chunks = self.chunker.chunk(explanation)
            
            masks = np.zeros((224, 224, len(chunks)))
            visual = np.zeros((224, 224))
            
            for i, chunk in enumerate(chunks):
                self.model.zero_grad()
                log_probs[chunk.position].backward(retain_graph=True)
                masks[..., i] = visual
            
            mask_avg = np.mean(masks, axis=2)
            
            word_masks = {}
            final_masks = np.zeros((224, 224, len(log_probs)))
            for i, chunk in enumerate(chunks):
                mask = masks[..., i] - mask_avg
                mask = np.clip(mask, 0, np.max(mask))
                mask = mask/np.max(mask)
                # Mask the image
                masked = (mask[..., np.newaxis] * np_image).astype(np.uint8)
                word_masks[(chunk.position, chunk.attribute)] = masked
            
        return explanation, np_image, word_masks

    