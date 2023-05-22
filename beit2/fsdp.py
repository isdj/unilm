import argparse
import os
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp

from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP, checkpoint_module
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy, size_based_auto_wrap_policy

from torch_xla.debug.metrics import metrics_report

from beit2.beit_model import BeitTrainingModule
from beit2.beit_local_dataset import BeitLocalDataset
from beit2.beit_v2_loss import BeitV2Loss

from itertools import chain
from typing import Dict, Union

patch_size = 16
image_key = "img"
img_shape = (224, 224)
NH = img_shape[0]//patch_size
NW = img_shape[1]//patch_size


def synchronize(message: str = "sync-workers"):
    xm.rendezvous(message)

def broadcast_params(model, root_rank: int = 0):
    parameters_and_buffers = []
    for p in chain(model.parameters(), model.buffers()):
    # Set all params in non-master devices to zero so that all_reduce is
    # equivalent to broadcasting parameters from master to other devices.
        if xm.get_ordinal() != root_rank:
            zero = torch.tensor(0, dtype=p.data.dtype, device=p.data.device)
            p.data.mul_(zero)
            parameters_and_buffers.append(p.data)
    xm.wait_device_ops()
    xm.all_reduce(xm.REDUCE_SUM, parameters_and_buffers)
    xm.mark_step()
    synchronize("broadcast_xla_master_model_param")


def mp_fn(local_rank):
    device = xm.xla_device()
    print(device)
    mae_model_config = {'patch_size': patch_size, 'in_chans': 1,
                        'depth': 12, 'num_heads': 12, 'embed_dim': 768,
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}

    large_model_config = {'patch_size': patch_size, 'in_chans': 1,
                        'depth': 24, 'num_heads': 16, 'embed_dim': 1024,
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}

    huge_model_config = {'patch_size': patch_size, 'in_chans': 1,
                        'depth': 32, 'num_heads': 16, 'embed_dim': 1280,
                        'decoder_depth': 2, 'decoder_num_heads': 16, 'decoder_embed_dim': 512}

    model = BeitTrainingModule(mae_model_config=mae_model_config)
    loss = BeitV2Loss()
    #broadcast_params(model)
    optimizer = torch.optim.AdamW(params=model.parameters(), amsgrad=False)


    num_params = sum(p.numel() for p in model.parameters())
    print(f'built model with {num_params / 1e6}M params')
    batch_size=128
    dl = pl.MpDeviceLoader(DataLoader(BeitLocalDataset(batch_size * 10000), batch_size=batch_size, num_workers=10), device)
    import time
    model.train()
    log_interval=10
    t0 = time.perf_counter()
    from functools import partial
    import timm

    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={timm.models.vision_transformer.Block})
    auto_wrapper_callable =  lambda m, *args, **kwargs: FSDP(
        checkpoint_module(m), *args, **kwargs)
    model = FSDP(model,
                 auto_wrap_policy=auto_wrap_policy,
                 auto_wrapper_callable = auto_wrapper_callable
                 )

    #model.to(device)
    # optimizer.to(device) # doesnt work
    loss.to(device)
#    for batch_idx in range(1, 1001):
    for batch_idx, batch_data in enumerate(dl, start=1):
        optimizer.zero_grad(set_to_none=True)

        if isinstance(batch_data, dict):
            for k, v in batch_data.items():
                batch_data[k] = v.to(device)
        else:
            batch_data = batch_data.to(device)
        outputs = model(batch_data)
        l = loss(outputs, batch_data)['loss']

        l.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and local_rank == 0:
            time_passed = time.perf_counter() - t0
            samples_processed = xm.xrt_world_size() * batch_size * log_interval
            print(f'{samples_processed / time_passed} samples/second')
            with open('/tmp/beit_vanilla_met_report.txt', 'w') as f:
                    f.write(metrics_report())

            t0 = time.perf_counter()

    
if __name__ == '__main__':
    # main()
    nprocs = 1
    xmp.spawn(mp_fn,
              args=(),
              nprocs=nprocs)
