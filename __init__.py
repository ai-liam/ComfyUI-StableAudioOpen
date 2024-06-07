"""
@author: liam-ai
@title: StableAudioSampler
@nickname: stableaudio
@description: A Simple integration of Stable Audio Diffusion with knobs and stuff!
"""

WEB_DIRECTORY = "./web"

from .nodes import StableAudioSampler, NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

