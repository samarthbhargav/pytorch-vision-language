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

    def get_bilinear_features(self, image):
        # Second hack for VGG16: reshaping feature maps
        image_features = self.vision_model(image).view(-1, 512, 7, 7)
        bilinear_features = self.cbp(image_features)
        # TODO: element-wise signed square root layer
        return bilinear_features

    def forward(self, image):
        bilinear_features = self.get_bilinear_features(image)
        logits = self.linear(bilinear_features)
        return logits
