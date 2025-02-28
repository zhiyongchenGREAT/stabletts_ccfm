import os
#set your device ids here
#os.environ['CUDA_VISIBLE_DEVICES'] = '4,5,6,7'

import torch
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataclasses import asdict

from datas.dataset import StableDataset, collate_fn
from datas.sampler import DistributedBucketSampler
from text import symbols
from config_bd import MelConfig, ModelConfig, TrainConfig
from models.model import StableTTS

from utils.scheduler import get_cosine_schedule_with_warmup
from utils.load import continue_training

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12346'
    dist.init_process_group("gloo" if os.name == "nt" else "nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()
    
def _init_config(model_config: ModelConfig, mel_config: MelConfig, train_config: TrainConfig):
    
    if not os.path.exists(train_config.model_save_path):
        print(f'Creating {train_config.model_save_path}')
        os.makedirs(train_config.model_save_path, exist_ok=True)
        
def freeze_layers(model: StableTTS):
    # return model
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.ref_encoder.parameters():
        param.requires_grad = False
    for param in model.dp.parameters():
        param.requires_grad = False
    model.fake_content.requires_grad = False
    model.fake_speaker.requires_grad = False
    print('freezed')
    return model

def train(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model_config = ModelConfig()
    mel_config = MelConfig()
    train_config = TrainConfig()
    
    _init_config(model_config, mel_config, train_config)
    
    model = StableTTS(len(symbols), mel_config.num_mels, **asdict(model_config)).to(rank)
    model = freeze_layers(model)
    
    model = DDP(model, device_ids=[rank])

    train_dataset = StableDataset(train_config.train_dataset_path, mel_config.hop_size)
    train_sampler = DistributedBucketSampler(train_dataset, train_config.batch_size, [32,300,400,500,600,700,800,900,1000,1100], num_replicas=world_size, rank=rank)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4, pin_memory=True, collate_fn=collate_fn, persistent_workers=True)

    # optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_config.learning_rate)
    optimizer = optim.AdamW(model.parameters(), lr=train_config.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(train_config.warmup_steps), num_training_steps=train_config.num_epochs * len(train_dataloader))
    
    # load latest checkpoints if possible
    current_epoch = continue_training(train_config.model_save_path, model, optimizer)

    model.train()
    for epoch in range(current_epoch, train_config.num_epochs):  # loop over the train_dataset multiple times
        train_dataloader.batch_sampler.set_epoch(epoch)
        if rank == 0:
            dataloader = tqdm(train_dataloader)
        else:
            dataloader = train_dataloader
            
        for batch_idx, datas in enumerate(dataloader):
            datas = [data.to(rank, non_blocking=True) for data in datas]
            x, x_lengths, y, y_lengths, z, z_lengths = datas
            optimizer.zero_grad()
            dur_loss, diff_loss, prior_loss, _ = model(x, x_lengths, y, y_lengths, z, z_lengths)
            loss = dur_loss + diff_loss + prior_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
        if rank == 0 and epoch % train_config.save_interval == 0:
            torch.save(model.module.state_dict(), os.path.join(train_config.model_save_path, f'checkpoint_{epoch}.pt'))
            torch.save(optimizer.state_dict(), os.path.join(train_config.model_save_path, f'optimizer_{epoch}.pt'))
        print(f"Rank {rank}, Epoch {epoch}, Loss {loss.item()}")

    cleanup()
    
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)