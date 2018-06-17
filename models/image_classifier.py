import torch
import torch.nn as nn
from .pretrained_models import PretrainedModel
from .bilinear import CompactBilinearPooling


class BilinearImageClassifier(nn.Module):
    def __init__(self, model_name, bilinear_dim, num_classes):
        super(BilinearImageClassifier, self).__init__()

        self.vision_model = PretrainedModel(model_name, layers_to_truncate=3)

        # First hack for VGG16
        img_feat_size = 512

        self.cbp = CompactBilinearPooling(img_feat_size, bilinear_dim)
        self.linear = nn.Linear(bilinear_dim, num_classes)
        self.init_weights()
        self.output_size = num_classes

    def init_weights(self):
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)

    def state_dict(self, *args, full_dict=False, **kwargs):
        return super().state_dict(*args, **kwargs)

    def get_bilinear_features(self, x):
        # Second hack for VGG16: reshaping feature maps
        x = self.vision_model(x).view(-1, 512, 7, 7)
        x = self.cbp(x)
        # Element-wise signed square root layer and L2 normalization
        x = torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-12)
        x = nn.functional.normalize(x, dim=-1)
        return x

    def forward(self, image):
        bilinear_features = self.get_bilinear_features(image)
        logits = self.linear(bilinear_features)
        return logits

    def get_features_labels(self, image):
        bilinear_features = self.get_bilinear_features(image)
        logits = self.linear(bilinear_features)
        _, labels = torch.max(logits.data, 1)
        return bilinear_features, labels
