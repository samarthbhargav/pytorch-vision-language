from .lrcn_trainer import LRCNTrainer
from .gve_trainer import GVETrainer
from .sentence_classifier_trainer import SCTrainer
from .image_classifier_trainer import ImageTrainer

class TrainerLoader:
    lrcn = LRCNTrainer
    gve = GVETrainer
    sc = SCTrainer
    ic = ImageTrainer
