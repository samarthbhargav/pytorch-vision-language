import os

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
import numpy as np

class ImageTrainer:

    def __init__(self, args, model, dataset, data_loader, logger, device, checkpoint=None):
        self.model = model
        self.dataset = dataset
        self.data_loader = data_loader
        self.train = args.train
        self.logger = logger
        self.device = device

        model.to(self.device)

        if checkpoint is None:
            self.criterion = nn.CrossEntropyLoss()
            self.params = filter(lambda p: p.requires_grad, model.parameters())
            self.optimizer = torch.optim.Adam(self.params, lr=args.learning_rate)
            self.total_steps = len(data_loader)
            self.num_epochs = args.num_epochs
            self.log_step = args.log_step
            self.curr_epoch = 0

    def train_epoch(self):
        result = []

        for i, (images, word_inputs, word_targets, lengths, ids, labels) in enumerate(self.data_loader):
            # Prepare mini-batch dataset
            if self.train:
                labels = labels.to(self.device)

                loss = self.train_step(images, labels)
                result.append(loss.data.item())

                step = self.curr_epoch * self.total_steps + i + 1
                self.logger.scalar_summary('batch_loss', loss.data.item(), step)

            else:
                labels = labels.to(self.device)
                score = self.eval_step(labels, lengths)
                result.append(score)

            # TODO: Add proper logging
            # Print log info
            if i % self.log_step == 0:
                print("Epoch [{}/{}], Step [{}/{}]".format(self.curr_epoch,
                    self.num_epochs, i, self.total_steps), end='')
                if self.train:
                    print(", Loss: {:.4f}, Perplexity: {:5.4f}".format(loss.data.item(),
                                np.exp(loss.data.item())), end='')
                print()


        self.curr_epoch += 1

        if self.train:
            self.logger.scalar_summary('epoch_loss', np.mean(result), self.curr_epoch)
        else:
            result = np.sum(result, axis=0)
            result = result[1] / result[0]
            print("Evaluation Accuracy: {}".format(result))


        return result


    def train_step(self, image_features, class_labels):
        # Forward, Backward and Optimize
        self.model.zero_grad()
        outputs = self.model(image_features)
        loss = self.criterion(outputs, class_labels)
        loss.backward()
        self.optimizer.step()

        return loss


    def eval_step(self, image_features, class_labels):
        outputs = self.model(image_features)
        _, predicted = torch.max(outputs.data, 1)

        return [class_labels.size(0), (predicted == class_labels).sum()]
