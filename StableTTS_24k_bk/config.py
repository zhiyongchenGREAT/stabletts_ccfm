from dataclasses import dataclass

@dataclass
class MelConfig:
    num_mels: int = 100
    n_fft: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    sampling_rate: int = 24000
    fmin: int = 0
    fmax: int = None
            
@dataclass
class ModelConfig:
    hidden_channels: int = 256
    filter_channels: int = 1024
    n_heads: int = 4
    n_enc_layers: int = 3 
    n_dec_layers: int = 6 
    kernel_size: int = 3
    p_dropout: int = 0.1
    gin_channels: int = 256
    boundary: float = 0
            
@dataclass
class TrainConfig:
    train_dataset_path: str = 'filelists/filelist.json'
    test_dataset_path: str = 'filelists/filelist.json' # not used
    batch_size: int = 16
    learning_rate: float = 1e-4
    num_epochs: int = 10000
    model_save_path: str = './checkpoints_6seg_s0'
    log_dir: str = './runs'
    log_interval: int = 16
    save_interval: int = 1
    warmup_steps: int = 200
