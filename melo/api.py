import re
import torch
import av
import numpy as np
import torch.nn as nn
from tqdm import tqdm

from . import utils
from .models import SynthesizerTrn
from .split_utils import split_sentence
from .download_utils import load_or_download_config, load_or_download_model

class TTS(nn.Module):
    def __init__(self, 
                language,
                device='auto',
                config_path=None,
                ckpt_path=None):
        super().__init__()
        if device == 'auto':
            device = 'cpu'
            if torch.cuda.is_available(): 
                device = 'cuda'
            if torch.backends.mps.is_available(): 
                device = 'mps'
        if 'cuda' in device:
            assert torch.cuda.is_available()

        hps = load_or_download_config(language, config_path=config_path)

        num_languages = hps.num_languages
        num_tones = hps.num_tones
        symbols = hps.symbols

        model = SynthesizerTrn(
            len(symbols),
            hps.data.filter_length // 2 + 1,
            hps.train.segment_size // hps.data.hop_length,
            n_speakers=hps.data.n_speakers,
            num_tones=num_tones,
            num_languages=num_languages,
            **hps.model,
        ).to(device)

        model.eval()
        self.model = model
        self.symbol_to_id = {s: i for i, s in enumerate(symbols)}
        self.hps = hps
        self.device = device
    
        checkpoint_dict = load_or_download_model(language, device, ckpt_path=ckpt_path)
        self.model.load_state_dict(checkpoint_dict['model'], strict=True)
        
        language = language.split('_')[0]
        self.language = 'ZH_MIX_EN' if language == 'ZH' else language

    @staticmethod
    def audio_numpy_concat(segment_data_list, sr, speed=1.):
        audio_segments = []
        for segment_data in segment_data_list:
            audio_segments += segment_data.reshape(-1).tolist()
            audio_segments += [0] * int((sr * 0.05) / speed)
        audio_segments = np.array(audio_segments).astype(np.float32)
        return audio_segments

    @staticmethod
    def split_sentences_into_pieces(text, language, quiet=False):
        texts = split_sentence(text, language_str=language)
        if not quiet:
            print(" > Text split to sentences.")
            print('\n'.join(texts))
            print(" > ===========================")
        return texts

    def tts_to_file(self, text, speaker_id, output_path=None, sdp_ratio=0.2, noise_scale=0.6, noise_scale_w=0.8, speed=1.0, quiet=False,):
        language = self.language
        texts = self.split_sentences_into_pieces(text, language, quiet)
        audio_list = []
        tx = tqdm(texts)
        for t in tx:
            if language in ['EN', 'ZH_MIX_EN']:
                t = re.sub(r'([a-z])([A-Z])', r'\1 \2', t)
            device = self.device
            bert, ja_bert, phones, tones, lang_ids = utils.get_text_for_tts_infer(t, language, self.hps, device, self.symbol_to_id)
            with torch.no_grad():
                x_tst = phones.to(device).unsqueeze(0)
                tones = tones.to(device).unsqueeze(0)
                lang_ids = lang_ids.to(device).unsqueeze(0)
                bert = bert.to(device).unsqueeze(0)
                ja_bert = ja_bert.to(device).unsqueeze(0)
                x_tst_lengths = torch.LongTensor([phones.size(0)]).to(device)
                del phones
                speakers = torch.LongTensor([speaker_id]).to(device)
                audio = self.model.infer(
                        x_tst,
                        x_tst_lengths,
                        speakers,
                        tones,
                        lang_ids,
                        bert,
                        ja_bert,
                        sdp_ratio=sdp_ratio,
                        noise_scale=noise_scale,
                        noise_scale_w=noise_scale_w,
                        length_scale=1. / speed,
                    )[0][0, 0].data.cpu().float().numpy()
                del x_tst, tones, lang_ids, bert, ja_bert, x_tst_lengths, speakers
                # 
            audio_list.append(audio)
        torch.cuda.empty_cache()
        audio = self.audio_numpy_concat(audio_list, sr=self.hps.data.sampling_rate, speed=speed)

        if output_path is None:
            return audio
        else:
            if audio.ndim == 1:
                audio = audio.reshape(-1, 1)
                channels = 1
                layout = 'mono'
            elif audio.ndim == 2:
                channels = audio.shape[1]
                layout = 'stereo' if channels == 2 else 'mono'
            else:
                raise ValueError(f"Unsupported audio data shape: {audio.shape}")
            
            if audio.dtype != np.float32:
                audio = audio.astype(np.float32)
            
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / np.max(np.abs(audio))
            
            codec_name = 'pcm_s16le'
            
            output_container = av.open(output_path, 'w')
            audio_stream = output_container.add_stream(codec_name, rate=self.hps.data.sampling_rate)
            if hasattr(audio_stream, 'layout'):
                audio_stream.layout = layout
            
            chunk_size = 1024
            total_samples = audio.shape[0]
            
            for start in range(0, total_samples, chunk_size):
                end = min(start + chunk_size, total_samples)
                chunk = audio[start:end]
                
                chunk = (chunk * 32767).astype(np.int16)
                
                chunk = chunk.T
                
                frame = av.AudioFrame.from_ndarray(
                    chunk, 
                    format='s16',
                    layout=layout
                )
                frame.sample_rate = self.hps.data.sampling_rate
                
                for packet in audio_stream.encode(frame):
                    output_container.mux(packet)
            
            for packet in audio_stream.encode(None):
                output_container.mux(packet)
            
            output_container.close()