import torch
from . import utils
from huggingface_hub import hf_hub_download

LANG_TO_HF_REPO_ID = {
    'EN': 'myshell-ai/MeloTTS-English',
    'EN_V2': 'myshell-ai/MeloTTS-English-v2',
    'EN_NEWEST': 'myshell-ai/MeloTTS-English-v3',
    'FR': 'myshell-ai/MeloTTS-French',
    'JP': 'myshell-ai/MeloTTS-Japanese',
    'ES': 'myshell-ai/MeloTTS-Spanish',
    'ZH': 'myshell-ai/MeloTTS-Chinese',
    'KR': 'myshell-ai/MeloTTS-Korean',
}

def load_or_download_config(locale, config_path=None):
    if config_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        config_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="config.json")
    return utils.get_hparams_from_file(config_path)

def load_or_download_model(locale, device, ckpt_path=None):
    if ckpt_path is None:
        language = locale.split('-')[0].upper()
        assert language in LANG_TO_HF_REPO_ID
        ckpt_path = hf_hub_download(repo_id=LANG_TO_HF_REPO_ID[language], filename="checkpoint.pth")
    return torch.load(ckpt_path, map_location=device)