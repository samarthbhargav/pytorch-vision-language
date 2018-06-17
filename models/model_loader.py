import torch

from .lrcn import LRCN
from .gve import GVE
from .sentence_classifier import SentenceClassifier
from .image_classifier import BilinearImageClassifier

class ModelLoader:
    def __init__(self, args, dataset, device):
        self.args = args
        self.dataset = dataset
        if device.type == 'cpu':
            self.location = 'cpu'
        else:
            self.location = '{}:{}'.format(device.type, device.index)

    def lrcn(self):
        # LRCN arguments
        pretrained_model = self.args.pretrained_model
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)

        layers_to_truncate = self.args.layers_to_truncate

        lrcn = LRCN(pretrained_model, embedding_size, hidden_size, vocab_size,
                layers_to_truncate)

        return lrcn

    def gve(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # GVE arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)

        ic = None
        # There are three ways to get image features
        # 1) Use precomputed features
        if self.dataset.use_image_features:
            input_type = self.dataset.input_size
        else:
            # 2) Use a pretrained compact bilinear classifier
            if self.args.ic_ckpt is not None:
                ic = self.ic()
                ic.load_state_dict(torch.load(self.args.ic_ckpt, map_location=self.location))
                input_type = ic.bilinear_dim
                for param in ic.parameters():
                    param.requires_grad = False
                ic.eval()
            # 3) Use features from a pretrained model like VGG16
            else:
                input_type = self.args.pretrained_model

        num_classes = self.dataset.num_classes

        sc = self.sc()
        sc.load_state_dict(torch.load(self.args.sc_ckpt, map_location=self.location))
        for param in sc.parameters():
            param.requires_grad = False
        sc.eval()

        gve = GVE(input_type, embedding_size, hidden_size, vocab_size, ic, sc,
                num_classes)

        if self.args.weights_ckpt:
            gve.load_state_dict(torch.load(self.args.weights_ckpt))

        return gve



    def sc(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # Sentence classifier arguments
        embedding_size = self.args.embedding_size
        hidden_size = self.args.hidden_size
        vocab_size = len(self.dataset.vocab)
        num_classes = self.dataset.num_classes

        sc = SentenceClassifier(embedding_size, hidden_size, vocab_size,
                num_classes)

        return sc

    def ic(self):
        # Make sure dataset returns labels
        self.dataset.set_label_usage(True)
        # Image classifier arguments
        pretrained_model = self.args.pretrained_model
        bilinear_dim = self.args.bilinear_dim
        num_classes = self.dataset.num_classes

        ic = BilinearImageClassifier(pretrained_model, bilinear_dim, num_classes)

        return ic
