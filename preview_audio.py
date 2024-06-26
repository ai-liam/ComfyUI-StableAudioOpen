
import os


class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO",),}
                }

    CATEGORY =  "Liam/Audio"
    DESCRIPTION = "preview audio"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        subfolder = audio.replace(f"/{audio_name}","")
        print(f"audio: {audio} audio_name: {audio_name}, subfolder: {subfolder}")
        return {"ui": {"audio":[audio_name,subfolder]}}
    
