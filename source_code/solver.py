from random import shuffle
import numpy as np

import torch
from torch.autograd import Variable

class DiceLoss(torch.nn.Module):

    def __init__(self, smooth=1.):
        super(DiceLoss,self).__init__()
        self.smooth = smooth
        self.m = torch.nn.Softmax(dim=1)
        # t = torch.ones([1, C, 20, 20])
        # if ignore_background:
        #     t[:, 0, :, :] = 0.0
        # self.ignore = t.contiguous().view(-1)
        # if torch.cuda.is_available():
        #     self.ignore.cuda()

    def forward(self, input, target):
        # input = torch.sigmoid(input)
        smooth = self.smooth

        iflat = self.m(input).contiguous().view(-1)# * self.ignore
        tflat = target.contiguous().view(-1)# * self.ignore

        intersection_mask = iflat * tflat

        intersection = intersection_mask.sum()

        return 1 - ((2. * intersection + smooth) /
                  (iflat.sum() + tflat.sum() + smooth))


class Solver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), C=24):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.C = C
        self.dice_loss = DiceLoss()

        self._reset_histories()

    def one_hot(self, targets, C=24):
        targets_extend = targets.clone()
        targets_extend.unsqueeze_(1)
        one_hot = torch.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot.scatter_(1, targets_extend, 1)
        # if self.ignore_background:
        #     one_hot = one_hot[:, :-1]

        return one_hot

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.train_dice_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.val_dice_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        # optim = self.optim(model.parameters(), **self.optim_args)
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):

                OHtargets = self.one_hot(targets=targets)
                inputs, OHtargets, targets = Variable(inputs), Variable(OHtargets), Variable(targets)
                if model.is_cuda:
                    inputs, targets, OHtargets = inputs.cuda(), targets.cuda(), OHtargets.cuda()

                optim.zero_grad()
                outputs = model(inputs)

                loss = 0
                for s in range(inputs.size()[0]):
                    loss += self.loss_func(outputs[s].view(self.C, -1).transpose(1, 0), targets[s].view(-1))
                loss /= inputs.size()[0]

                loss.backward()
                optim.step()

                d_score = 1 - self.dice_loss(outputs, OHtargets).data.cpu().numpy()

                self.train_loss_history.append(loss.data.cpu().numpy())
                self.train_dice_history.append(d_score)
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)

                    last_log_nth_dice = self.train_dice_history[-log_nth:]
                    train_dice = np.mean(last_log_nth_dice)

            _, preds = torch.max(outputs, 1)

            targets_mask = targets > 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/IoU/loss: %.3f / %.3f / %.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_dice,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            val_dices = []
            model.eval()
            for inputs, targets in val_loader:
                OHtargets = self.one_hot(targets=targets)
                inputs, OHtargets, targets = Variable(inputs), Variable(OHtargets), Variable(targets)
                if model.is_cuda:
                    inputs, targets, OHtargets = inputs.cuda(), targets.cuda(), OHtargets.cuda()

                outputs = model.forward(inputs)
                loss = 0
                for s in range(inputs.size()[0]):
                    loss += self.loss_func(outputs[s].view(self.C, -1).transpose(1, 0), targets[s].view(-1))
                loss /= inputs.size()[0]

                d_score = 1 - self.dice_loss(outputs, OHtargets).data.cpu().numpy()

                val_losses.append(loss.data.cpu().numpy())
                val_dices.append(d_score)

                _, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets > 0
                scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                val_scores.append(scores)

            model.train()
            val_acc, val_loss, val_dice = np.mean(val_scores), np.mean(val_losses), np.mean(val_dices)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            self.val_dice_history.append(val_dice)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/IoU/loss: %.3f / %.3f / %.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_dice,
                                                                   val_loss))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.\n')
        return val_acc



class dSolver(object):
    default_adam_args = {"lr": 1e-4,
                         "betas": (0.9, 0.999),
                         "eps": 1e-8,
                         "weight_decay": 0.0}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=DiceLoss(), C=24):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.C = C

        self._reset_histories()

    def one_hot(self, targets, C=24):
        targets_extend = targets.clone()
        targets_extend.unsqueeze_(1)
        one_hot = torch.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot.scatter_(1, targets_extend, 1)

        return one_hot

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.train_dice_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.val_dice_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        # optim = self.optim(model.parameters(), **self.optim_args)
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):


                # SET YOUR OWN NUMBER OF CLASSES HERE
                OHtargets = self.one_hot(targets=targets)
                inputs, OHtargets = Variable(inputs), Variable(OHtargets)
                if model.is_cuda:
                    inputs, targets, OHtargets = inputs.cuda(), targets.cuda(), OHtargets.cuda()


                optim.zero_grad()
                outputs = model(inputs)

                loss = self.loss_func(outputs, OHtargets)
                loss.backward()
                optim.step()

                d_score = 1 - loss.data.cpu().numpy()

                self.train_loss_history.append(loss.data.cpu().numpy())
                self.train_dice_history.append(d_score)
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)

                    last_log_nth_dice = self.train_dice_history[-log_nth:]
                    train_dice = np.mean(last_log_nth_dice)

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0 e.g. for segmentation
            targets_mask = targets > 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/IoU/loss: %.3f / %.3f / %.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_dice,
                                                                   train_loss))

            # VALIDATION
            val_losses = []
            val_scores = []
            val_dices = []
            model.eval()
            for inputs, targets in val_loader:

                ############## ONE HOT GETTING BUSY HERE
                OHtargets = self.one_hot(targets=targets)
                inputs, OHtargets = Variable(inputs), Variable(OHtargets)
                if model.is_cuda:
                    inputs, targets, OHtargets = inputs.cuda(), targets.cuda(), OHtargets.cuda()

                outputs = model.forward(inputs)
                loss = self.loss_func(outputs, OHtargets)
                val_losses.append(loss.data.cpu().numpy())

                d_score = 1 - loss.data.cpu().numpy()
                val_dices.append(d_score)

                _, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets > 0
                scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                val_scores.append(scores)

            model.train()
            val_acc, val_loss, val_dice = np.mean(val_scores), np.mean(val_losses), np.mean(val_dices)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            self.val_dice_history.append(val_dice)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/IoU/loss: %.3f / %.3f / %.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_dice,
                                                                   val_loss))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.\n')
        return val_acc

class cSolver(object):
    default_adam_args = {"lr": 1e-4,
                        "momentum":0.99}

    def __init__(self, optim=torch.optim.Adam, optim_args={},
                 loss_func=torch.nn.CrossEntropyLoss(), C=24):
        optim_args_merged = self.default_adam_args.copy()
        optim_args_merged.update(optim_args)
        self.optim_args = optim_args_merged
        self.optim = optim
        self.loss_func = loss_func
        self.dice_loss = DiceLoss()
        self.C = C

        self._reset_histories()

    def one_hot(self, targets, C=24):
        targets_extend = targets.clone()
        targets_extend.unsqueeze_(1)
        one_hot = torch.FloatTensor(targets_extend.size(0), C, targets_extend.size(2), targets_extend.size(3)).zero_()
        one_hot.scatter_(1, targets_extend, 1)

        return one_hot

    def _reset_histories(self):
        """
        Resets train and val histories for the accuracy and the loss.
        """
        self.train_loss_history = []
        self.train_acc_history = []
        self.train_dice_history = []
        self.val_acc_history = []
        self.val_loss_history = []
        self.val_dice_history = []

    def train(self, model, train_loader, val_loader, num_epochs=10, log_nth=0):
        """
        Train a given model with the provided data.
        Inputs:
        - model: model object initialized from a torch.nn.Module
        - train_loader: train data in torch.utils.data.DataLoader
        - val_loader: val data in torch.utils.data.DataLoader
        - num_epochs: total number of training epochs
        - log_nth: log training accuracy and loss every nth iteration
        """
        # optim = self.optim(model.parameters(), **self.optim_args)
        optim = self.optim(filter(lambda p: p.requires_grad, model.parameters()), **self.optim_args)
        self._reset_histories()
        iter_per_epoch = len(train_loader)

        if torch.cuda.is_available():
            model.cuda()

        print('START TRAIN.')
        ########################################################################
        # TODO:                                                                #
        # Write your own personal training method for our solver. In each      #
        # epoch iter_per_epoch shuffled training batches are processed. The    #
        # loss for each batch is stored in self.train_loss_history. Every      #
        # log_nth iteration the loss is logged. After one epoch the training   #
        # accuracy of the last mini batch is logged and stored in              #
        # self.train_acc_history. We validate at the end of each epoch, log    #
        # the result and store the accuracy of the entire validation set in    #
        # self.val_acc_history.                                                #
        #                                                                      #
        # Your logging could like something like:                              #
        #   ...                                                                #
        #   [Iteration 700/4800] TRAIN loss: 1.452                             #
        #   [Iteration 800/4800] TRAIN loss: 1.409                             #
        #   [Iteration 900/4800] TRAIN loss: 1.374                             #
        #   [Epoch 1/5] TRAIN acc/loss: 0.560/1.374                            #
        #   [Epoch 1/5] VAL   acc/loss: 0.539/1.310                            #
        #   ...                                                                #
        ########################################################################

        for epoch in range(num_epochs):
            # TRAINING

            for i, (inputs, targets) in enumerate(train_loader, 1):


                # SET YOUR OWN NUMBER OF CLASSES HERE
                OHtargets = self.one_hot(targets=targets)
                inputs, OHtargets, targets = Variable(inputs), Variable(OHtargets), Variable(targets)
                if model.is_cuda:
                    inputs, targets, OHtargets = inputs.cuda(), targets.cuda(), OHtargets.cuda()


                optim.zero_grad()
                outputs = model(inputs)

                loss1 = 0
                for s in range(inputs.size()[0]):
                    loss1 += self.loss_func(outputs[s].view(self.C, -1).transpose(1, 0), targets[s].view(-1))
                loss1 /= inputs.size()[0]
                loss2 = self.dice_loss(outputs, OHtargets)

                loss = (loss1 + loss2)/2

                loss.backward()
                optim.step()

                d_score = 1 - loss2.data.cpu().numpy()

                self.train_dice_history.append(d_score)
                self.train_loss_history.append(loss.data.cpu().numpy())
                if log_nth and i % log_nth == 0:
                    last_log_nth_losses = self.train_loss_history[-log_nth:]
                    train_loss = np.mean(last_log_nth_losses)

                    last_log_nth_dice = self.train_dice_history[-log_nth:]
                    train_dice = np.mean(last_log_nth_dice)

            _, preds = torch.max(outputs, 1)

            # Only allow images/pixels with label >= 0 e.g. for segmentation
            targets_mask = targets > 0
            train_acc = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
            self.train_acc_history.append(train_acc)
            if log_nth:
                print('[Epoch %d/%d] TRAIN acc/IoU/loss: %.3f / %.3f / %.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   train_acc,
                                                                   train_dice,
                                                                   train_loss))
            # VALIDATION
            val_losses = []
            val_scores = []
            val_dices = []
            model.eval()
            for inputs, targets in val_loader:

                ############## ONE HOT GETTING BUSY HERE
                OHtargets = self.one_hot(targets=targets)
                inputs, OHtargets, targets = Variable(inputs), Variable(OHtargets), Variable(targets)
                if model.is_cuda:
                    inputs, targets, OHtargets = inputs.cuda(), targets.cuda(), OHtargets.cuda()

                outputs = model.forward(inputs)

                loss1 = 0
                for s in range(inputs.size()[0]):
                    loss1 += self.loss_func(outputs[s].view(self.C, -1).transpose(1, 0), targets[s].view(-1))
                loss1 /= inputs.size()[0]
                loss2 = self.dice_loss(outputs, OHtargets)

                loss = (loss1 + loss2)/2

                val_losses.append(loss.data.cpu().numpy())

                d_score = 1 - loss2.data.cpu().numpy()
                val_dices.append(d_score)

                _, preds = torch.max(outputs, 1)

                # Only allow images/pixels with target >= 0 e.g. for segmentation
                targets_mask = targets >= 0
                scores = np.mean((preds == targets)[targets_mask].data.cpu().numpy())
                val_scores.append(scores)

            model.train()
            val_acc, val_loss, val_dice = np.mean(val_scores), np.mean(val_losses), np.mean(val_dices)
            self.val_acc_history.append(val_acc)
            self.val_loss_history.append(val_loss)
            self.val_dice_history.append(val_dice)
            if log_nth:
                print('[Epoch %d/%d] VAL   acc/IoU/loss: %.3f / %.3f / %.3f' % (epoch + 1,
                                                                   num_epochs,
                                                                   val_acc,
                                                                   val_dice,
                                                                   val_loss))

        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        print('FINISH.\n')
        return val_acc
