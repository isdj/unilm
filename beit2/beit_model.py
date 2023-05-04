import importlib
from functools import partial
from typing import Dict
import torch
import torch.nn as nn
from timm.models import create_model

from typing import Optional

import beit2.modeling_vqkd # needed for registering the models
from beit2.modeling_pretrain import VisionTransformerForMaskedImageModeling
from beit2.mae import MaskedAutoencoderViT

CodebookSize = 8192
CodebookDim = 32
TokenizerProcessType = "imagenet_norm"

in_chans = 1
patch_size = 16
# vis_ratio = 0.25
vis_ratio = 1-75/196 # beit's parameter for pretraining
image_key = "img"
img_shape = (224, 224)
NH = img_shape[0]//patch_size
NW = img_shape[1]//patch_size
DefaultTokenizerWeightsPath = 'https://conversationhub.blob.core.windows.net/beit-share-public/beitv2/vqkd_encoder_base_decoder_3x768x12_clip-d5036aa7.pth'


class BeitTrainingModule(nn.Module):

    def __init__(
            self,
            mae_model_config: Dict,
            tokenizer_weights_path: str = DefaultTokenizerWeightsPath,
            flip_for_tokenizer: bool = False,
        ):
        """
        this module has a tokenizer model, initialized from trained checkpoint and a masked-autoencoder model that we wish to train.
        in training, the tokenizer is frozen. it tokenizes each patch and returns the index of the token for each patch (out of CodebookSize options).
        the model should learn to predict the semantic classes that are the tokens, represented by their indices.
        BEiT v2 paper: https://arxiv.org/abs/2208.06366
        """

        super(BeitTrainingModule, self).__init__()

        self.tokenizer = create_model(
            "vqkd_encoder_base_decoder_3x768x12_clip",
            pretrained=True,
            pretrained_weight=tokenizer_weights_path,
            as_tokenzer=True,
            n_code=CodebookSize,
            code_dim=CodebookDim,
            process_type=TokenizerProcessType,
            teacher_model_type=None, # not training tokenizer, just using its output for beit_v2 training
        ).eval()
        for param in self.tokenizer.parameters():
            param.requires_grad = False
        self._flip_for_tokenizer = flip_for_tokenizer

        self.mae_model = MaskedAutoencoderViT(**mae_model_config)

    def train(self, mode: bool = True):
        if mode is False:
            # means we called eval()
            # we will call eval() on self.mae_model so that if someone override eval() it will be called
            # e.g. foundation_model.models.masked_autoencoder.mae_model.MAEModel
            self.mae_model.eval()
        else:
            # keeping tokenizer in eval mode. this is important for batch_norm
            self.mae_model.train()
        return self

    def load_state_dict(self, state_dict, strict: bool = True):
        # not loading tokenizer weights (we use the already trained one)
        prefix = "mae_model."
        remove_prefix = lambda x: x[len(prefix):]
        mae_model_state_dict = {remove_prefix(key): val for key, val in state_dict.items() if key.startswith(prefix)}
        res = self.mae_model.load_state_dict(mae_model_state_dict, strict=strict)
        # returning torch.nn.modules.module._IncompatibleKeys with "mae_model." in key names
        # unfortunately, torch.nn.modules.module._IncompatibleKeys dont allow to set values to its fields, so we cant
        # do res.missing_keys = ... instead we create a new object with the correct values
        return type(res)(
            missing_keys=type(res.missing_keys)(prefix + k for k in res.missing_keys),
            unexpected_keys = type(res.unexpected_keys)(prefix + k for k in res.unexpected_keys),
        )

    def forward(self, batch):
        with torch.no_grad():
            image_for_tokenizer = batch["image_for_tokenizer"]
            if self._flip_for_tokenizer:
                image_for_tokenizer = image_for_tokenizer.flip(2) # B,C,H,W
            labels = self.tokenizer.get_codebook_indices(image_for_tokenizer)  # same as duplicating the gray channel to 3 channels
            if self._flip_for_tokenizer:
                patches_shape = (-1,) + self.tokenizer.token_shape
                labels = labels.reshape(patches_shape).flip(1).reshape(labels.shape) # B,N -> B,NH,NW -> B,NH,NW (flipped) -> B,N

        outputs = self.mae_model(batch)

        if isinstance(outputs, torch.Tensor):
            return {
                "tokenizer_labels": labels,
                "beit_model_output": outputs
            }
        else:
            assert isinstance(outputs, dict)
            assert "beit_model_output" in outputs and "tokenizer_labels" not in outputs, \
                f"output keys dont match Beit training module - {list(outputs.keys())}"
            outputs["tokenizer_labels"] = labels
            return outputs
