import torch.nn as nn
import torch.nn.functional as F
from .pretrained_models import PretrainedModel
from .bilinear import CompactBilinearPooling


class BilinearImageClassifier(nn.Module):
    def __init__(self, model_name, bilinear_dim, num_classes):
        super(BilinearImageClassifier, self).__init__()

        self.vision_model = PretrainedModel(model_name, layers_to_truncate=1)
        img_feat_size = self.vision_model.output_size

        self.cbp = CompactBilinearPooling(img_feat_size, img_feat_size, bilinear_dim)
        self.linear = nn.Linear(bilinear_dim, num_classes)
        self.init_weights()
        self.output_size = num_classes

    def init_weights(self):
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)

    def state_dict(self, *args, full_dict=False, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, image):
        image_features = self.vision_model(image)
        bilinear_features = self.cbp(image_features, image_features)
        logits = self.linear(bilinear_features)
        return logits
