import torch
import torch.nn as nn
import torch.nn.functional as F

from .pretrained_models import PretrainedModel

class ImageClassifier(nn.Module):
    def __init__(self, input, hidden_size, num_classes):
        super(ImageClassifier, self).__init__()

        self.vision_model = PretrainedModel(input,
                    layers_to_truncate=1)
        img_feat_size = self.vision_model.output_size

        self.linear1 = nn.Linear(img_feat_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_classes)
        self.init_weights()
        self.output_size = num_classes

    def init_weights(self):
        self.linear1.weight.data.uniform_(-0.1, 0.1)
        self.linear2.weight.data.uniform_(-0.1, 0.1)
        self.linear1.bias.data.fill_(0)
        self.linear2.bias.data.fill_(0)

    def state_dict(self, *args, full_dict=False, **kwargs):
        return super().state_dict(*args, **kwargs)

    def forward(self, image):
        image_features = self.vision_model(image)
        outputs = F.relu(self.linear1(image_features))
        outputs = F.relu(self.linear2(outputs))
        return outputs