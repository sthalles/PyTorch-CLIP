# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch
import torch.distributed as dist
from transformers import DistilBertModel, DistilBertConfig

class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, text_encoder_checkpoints="distilbert-base-uncased"):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.image_encoder = base_encoder(num_classes=mlp_dim)
        configuration = DistilBertConfig()
        configuration.max_position_embeddings = 80

        # Initializing a model from the configuration
        self.text_encoder = DistilBertModel(configuration)

        image_output_dim = self.image_encoder.fc.weight.shape[1]
        self.image_encoder.fc = nn.Identity()

        # linear projectors
        self.image_projection = nn.Linear(image_output_dim, dim, bias=False)
        self.text_projection = nn.Linear(768, dim, bias=False)
        self.cross_entropy = nn.CrossEntropyLoss()

    def contrastive_loss(self, v, t):
        # normalize
        v = nn.functional.normalize(v, dim=1)
        t = nn.functional.normalize(t, dim=1)

        v = AllGather.apply(v)
        t = AllGather.apply(t)

        # Einstein sum is more intuitive
        logits = torch.matmul(v, t.T) / self.T
        N = logits.shape[0]  # batch size per GPU

        rank = 0
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            rank = torch.distributed.get_rank()

        labels = torch.arange(N, dtype=torch.long).cuda()
        l1 = self.cross_entropy(logits / self.T, labels)
        l2 = self.cross_entropy(logits.T / self.T, labels)

        return (l1 + l2) / 2

    def forward(self, images, tokens):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        image_embed = self.image_encoder(images)
        text_embed = self.text_encoder(**tokens).last_hidden_state[:,0]

        image_embed = self.image_projection(image_embed)
        text_embed = self.text_projection(text_embed)

        return self.contrastive_loss(image_embed, text_embed)


class AllGather(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            outputs = [torch.zeros_like(x) for _ in range(dist.get_world_size())]
            dist.all_gather(outputs, x)
            return torch.cat(outputs, 0)
        return x

    @staticmethod
    def backward(ctx, grads):
        if (
            dist.is_available()
            and dist.is_initialized()
            and (dist.get_world_size() > 1)
        ):
            s = (grads.shape[0] // dist.get_world_size()) * dist.get_rank()
            e = (grads.shape[0] // dist.get_world_size()) * (dist.get_rank() + 1)
            grads = grads.contiguous()
            dist.all_reduce(grads)
            return grads[s:e]
        return grads