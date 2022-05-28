#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import itertools

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
import numpy as np

from bayes.bayesian_privacy_accountant import BayesianPrivacyAccountant
from utils.dp_mechanism import cal_sensitivity, Laplace, Gaussian_Simple


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class LocalUpdate:
    def __init__(self, args, dataset=None, idxs=None, dp_mechanism='no_dp', dp_epsilon=20,
                 dp_delta=1e-5, dp_clip=20):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True,
                                    num_workers=2, drop_last=True)  # TODO
        self.dp_mechanism = dp_mechanism
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.dp_clip = dp_clip
        self.idxs = idxs

        # newly added
        self.gpu_id = 0

    def train(self, student):
        student.train()

        # Bayes accountant
        accountant = BayesianPrivacyAccountant(powers=[2, 4, 8, 16, 32],
                                               total_steps=len(self.ldr_train) * self.args.local_ep)

        # train and update
        criterion = nn.CrossEntropyLoss(reduction='none').to(self.args.device)
        optimizer = torch.optim.SGD(student.parameters(), lr=self.args.lr)

        sampling_prob = 0.1
        max_grad_norm = self.args.C

        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)

        epoch_loss = []
        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (inputs, labels) in enumerate(self.ldr_train):
                inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)
                input_v = Variable(inputs)
                label_v = Variable(labels.long().view(-1) % 10)

                batch_size = float(len(inputs))

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = student(input_v)
                loss = criterion(outputs, label_v)

                # max_grad_norm = opt.C * 0.9**epoch
                if accountant:
                    grads_est = []
                    num_subbatch = 8
                    for i in range(num_subbatch):
                        grad_sample = torch.autograd.grad(
                            loss[np.delete(range(int(batch_size)), i)].mean(),
                            [p for p in student.parameters() if p.requires_grad],
                            retain_graph=True
                        )
                        with torch.no_grad():
                            grad_sample = torch.cat([g.view(-1) for g in grad_sample])
                            grad_sample /= max(1.0, grad_sample.norm().item() / max_grad_norm)
                            grads_est += [grad_sample]
                    with torch.no_grad():
                        grads_est = torch.stack(grads_est)
                        LocalUpdate.sparsify_update(grads_est, p=sampling_prob, use_grad_field=False)

                (loss.mean()).backward()
                batch_loss.append(loss.mean().item())

                if accountant:
                    with torch.no_grad():
                        torch.nn.utils.clip_grad_norm_(student.parameters(), max_grad_norm)
                        for group in optimizer.param_groups:
                            for p in group['params']:
                                if p.grad is not None:
                                    p.grad += torch.randn_like(p.grad) * (self.args.sigma * max_grad_norm)
                        LocalUpdate.sparsify_update(student.parameters(), p=sampling_prob)

                optimizer.step()
                scheduler.step()

                if accountant:
                    with torch.no_grad():
                        batch_size = float(len(inputs))
                        q = batch_size / len(self.ldr_train.dataset)
                        # NOTE:
                        # Using combinations within a set of gradients (like below)
                        # does not actually produce samples from the correct distribution
                        # (for that, we need to sample pairs of gradients independently).
                        # However, the difference is not significant, and it speeds up computations.
                        pairs = list(zip(*itertools.combinations(grads_est, 2)))
                        accountant.accumulate(
                            ldistr=(torch.stack(pairs[0]), self.args.sigma * max_grad_norm),
                            rdistr=(torch.stack(pairs[1]), self.args.sigma * max_grad_norm),
                            q=q,
                            steps=1,
                        )

            batch_loss_avg = sum(batch_loss) / len(batch_loss)
            epoch_loss.append(batch_loss_avg)

            # print training stats every epoch
            running_eps = accountant.get_privacy(target_delta=1e-5) if accountant else None
            print(f"Epoch: %d/%d. Loss: %.3f. Privacy (𝜀,𝛿): %s" %
                  (epoch + 1, self.args.local_ep, batch_loss_avg, running_eps))

        # add noises to parameters
        if self.dp_mechanism != 'no_dp':
            self.add_noise(student)
        return student.state_dict(), sum(epoch_loss) / len(epoch_loss), scheduler.get_last_lr()[0]

    def clip_gradients(self, net):
        if self.dp_mechanism == 'Laplace':
            # Laplace use 1 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(1) / self.dp_clip)
        elif self.dp_mechanism == 'Gaussian':
            # Gaussian use 2 norm
            for k, v in net.named_parameters():
                v.grad /= max(1, v.grad.norm(2) / self.dp_clip)

    def add_noise(self, net):
        sensitivity = cal_sensitivity(self.args.lr, self.dp_clip, len(self.idxs))
        if self.dp_mechanism == 'Laplace':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Laplace(epsilon=self.dp_epsilon, sensitivity=sensitivity, size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise
        elif self.dp_mechanism == 'Gaussian':
            with torch.no_grad():
                for k, v in net.named_parameters():
                    noise = Gaussian_Simple(epsilon=self.dp_epsilon, delta=self.dp_delta, sensitivity=sensitivity,
                                            size=v.shape)
                    noise = torch.from_numpy(noise).to(self.args.device)
                    v += noise

    def test(self, test_loader, net):
        """
            Compute test accuracy.
        """
        correct = 0.0
        total = 0.0

        for inputs, labels in test_loader:
            inputs, labels = inputs.to(self.args.device), labels.to(self.args.device)

            outputs = net(Variable(inputs))
            _, predicted = torch.max(outputs.data, 1)
            # print(predicted.cpu().numpy())
            # print(labels.cpu().numpy())
            total += labels.size(0)
            correct += (predicted == (labels.long().view(-1) % 10)).sum()
            # print torch.cat([predicted.view(-1, 1), (labels.long() % 10)], dim=1)

        print('Accuracy of the network on test images: %f %%' % (100 * float(correct) / total))
        return 100 * float(correct) / total

    @staticmethod
    def sparsify_update(params, p, use_grad_field=True):
        init = True
        for param in params:
            if param is not None:
                if init:
                    idx = torch.zeros_like(param, dtype=torch.bool)
                    idx.bernoulli_(1 - p)
                if use_grad_field:
                    if param.grad is not None:
                        idx = torch.zeros_like(param, dtype=torch.bool)
                        idx.bernoulli_(1 - p)
                        param.grad.data[idx] = 0
                else:
                    init = False
                    param.data[idx] = 0
        return idx
