import torch
import torch.nn as nn

from dataclasses import asdict

from utils.audio import LogMelSpectrogram
from config import ModelConfig, MelConfig
from models.model import StableTTS

from text import symbols
from text import cleaned_text_to_sequence
from text.mandarin import chinese_to_cnm3
from text.english import english_to_ipa2
from text.japanese import japanese_to_ipa2

from datas.dataset import intersperse
from utils.audio import load_and_resample_audio
from utils.meldataset import get_mel_spectrogram

def get_vocoder(model_path, config_path) -> nn.Module:
    from BigVGAN.api import BigVGANWrapper
    vocoder = BigVGANWrapper(config_path, model_path)
    return vocoder

class StableTTSAPI(nn.Module):
    def __init__(self, tts_model_path, vocoder_model_path, vocoder_config_path):
        super().__init__()

        self.mel_config = MelConfig()
        self.tts_model_config = ModelConfig()
        
        # text to mel spectrogram
        self.tts_model = StableTTS(len(symbols), self.mel_config.num_mels, **asdict(self.tts_model_config))
        self.tts_model.load_state_dict(torch.load(tts_model_path, map_location='cpu', weights_only=True))
        self.tts_model.eval()
        
        # mel spectrogram to waveform
        self.vocoder_model = get_vocoder(vocoder_model_path, vocoder_config_path)
        
        self.g2p_mapping = {
            'chinese': chinese_to_cnm3,
            'japanese': japanese_to_ipa2,
            'english': english_to_ipa2,
        }
        self.supported_languages = self.g2p_mapping.keys()
        
    @ torch.inference_mode()
    def inference(self, text, ref_audio, language, step, temperature=1.0, length_scale=1.0, solver=None, cfg=3.0):
        device = next(self.parameters()).device
        phonemizer = self.g2p_mapping.get(language)
        
        text = phonemizer(text)
        text = torch.tensor(intersperse(cleaned_text_to_sequence(text), item=0), dtype=torch.long, device=device).unsqueeze(0)
        text_length = torch.tensor([text.size(-1)], dtype=torch.long, device=device)
        
        ref_audio = load_and_resample_audio(ref_audio, self.mel_config.sampling_rate).to(device)
        ref_audio = get_mel_spectrogram(ref_audio, self.mel_config)
        
        mel_output = self.tts_model.synthesise(text, text_length, step, temperature, ref_audio, length_scale, solver, cfg)['decoder_outputs']
        audio_output = self.vocoder_model(mel_output).squeeze(0)
        return audio_output.cpu(), mel_output.cpu()
    
    def get_params(self):
        tts_param = sum(p.numel() for p in self.tts_model.parameters()) / 1e6
        vocoder_param = sum(p.numel() for p in self.vocoder_model.parameters()) / 1e6
        return tts_param, vocoder_param
    
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tts_model_path = './checkpoints/checkpoint_0.pt'
    vocoder_model_path = './vocoders/pretrained/vocos.pt'
    
    model = StableTTSAPI(tts_model_path, vocoder_model_path, 'vocos')
    model.to(device)
    
    text = '樱落满殇祈念集……殇歌花落集思祈……樱花满地集于我心……揲舞纷飞祈愿相随……'
    audio = './audio_1.wav'
    
    audio_output, mel_output = model.inference(text, audio, 'chinese', 10, solver='dopri5', cfg=3)
    print(audio_output.shape)
    print(mel_output.shape)
    
    import torchaudio
    torchaudio.save('output.wav', audio_output, MelConfig().sample_rate)
    
    
