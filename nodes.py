import os
import glob
import random
import torch
import torchaudio
from einops import rearrange
from stable_audio_tools import get_pretrained_model
from stable_audio_tools.inference.generation import generate_diffusion_cond
import numpy as np

from safetensors.torch import load_file
from .util_config import get_model_config
from stable_audio_tools.models.factory import create_model_from_config
from stable_audio_tools.models.utils import load_ckpt_state_dict
from .preview_audio import PreViewAudio

def get_device():
    # logging.info("[INFO] SDXLNodesLib get_device")
    device = "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    print(f"[INFO]device: {device}")
    return device

device = get_device()# "cuda" if torch.cuda.is_available() else "cpu"

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False
base_path = os.path.dirname(os.path.realpath(__file__))
os.makedirs("models/audio_checkpoints", exist_ok=True)

# Our any instance wants to be a wildcard string
any = AnyType("audio")
model_files = [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.safetensors")] + [os.path.basename(file) for file in glob.glob("models/audio_checkpoints/*.ckpt")]
if len(model_files) == 0:
    model_files.append("Put models in models/audio_checkpoints")


def generate_audio(prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save_name, save_path, model_filename):
    model_path = f"models/audio_checkpoints/{model_filename}"
    if model_filename.endswith(".safetensors") or model_filename.endswith(".ckpt"):
        model_config = get_model_config()
        model = create_model_from_config(model_config)
        model.load_state_dict(load_ckpt_state_dict(model_path))
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    else:
        model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
        sample_rate = model_config["sample_rate"]
        sample_size = model_config["sample_size"]
    
    model = model.to(device)

    conditioning = [{
        "prompt": prompt,
        "seconds_start": 0,
        "seconds_total": 30
    }]
    
    seed = np.random.randint(0, np.iinfo(np.int32).max)

    output = generate_diffusion_cond(
        model,
        steps=steps,
        cfg_scale=cfg_scale,
        conditioning=conditioning,
        sample_size=sample_size,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        sampler_type=sampler_type,
        device=device,
        seed=seed,
    )

    output = rearrange(output, "b d n -> d (b n)")

    output = output.to(torch.float32).div(torch.max(torch.abs(output))).clamp(-1, 1).mul(32767).to(torch.int16).cpu()
    back_path =""
    save = True
    if save:
        if save_name == "":
            seed_range = (0, 4294967295)
            seed = random.randint(*seed_range)
            save_name = f"{seed}.wav"
        if save_path == "":
            save_path = "audio"
        path = os.path.dirname(os.path.realpath(__file__))+"/../../"
        back_path = os.path.join(save_path, save_name)
        _path = path+ "output/" +save_path
        if not os.path.exists(_path):
            os.makedirs(_path, exist_ok=True)
        _save_file = f'{_path}/{save_name}'
        print(f"wav Saving to {_save_file}, back_path: {back_path}")
        torchaudio.save(f"{_save_file}", output, sample_rate)
    
    # Convert to bytes
    audio_bytes = output.numpy().tobytes()
    return audio_bytes, sample_rate,back_path

class StableAudioSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"default": "128 BPM tech house drum loop"}),
                "model_filename": (model_files, ),
                "steps": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sample_size": ("INT", {"default": 65536, "min": 1, "max": 1000000}),
                "sigma_min": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sigma_max": ("FLOAT", {"default": 500.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
                "sampler_type": ("STRING", {"default": "dpmpp-3m-sde"}),
                "save_path": ("STRING", {"default": "audio"}),
                "save_name": ("STRING", {"default": "output.wav"}),
            }
        }

    RETURN_TYPES = ("AUDIO", "INT",any)
    RETURN_NAMES = ("audio","rate","bytes",)
    FUNCTION = "sample"
    OUTPUT_NODE = True

    CATEGORY = "Liam/Audio"
    DESCRIPTION = "stable-audio-open-1.0"

    def sample(self, prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, save_name, save_path, model_filename):
        audio_bytes, sample_rate,save_file = generate_audio(prompt, steps, cfg_scale, sample_size, sigma_min, sigma_max, sampler_type, device, save_name, save_path, model_filename)
        return (save_file, sample_rate,audio_bytes)



NODE_CLASS_MAPPINGS = {
    "StableAudioSamplerLiam": StableAudioSampler,
    "PreViewAudioLiam": PreViewAudio
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StableAudioSamplerLiam": "Stable Diffusion Audio Sampler @Liam",
     "PreViewAudioLiam": "PreView Audio @Liam"
}
