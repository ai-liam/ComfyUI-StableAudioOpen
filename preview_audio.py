
import os


class PreViewAudio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required":
                    {"audio": ("AUDIO",),}
                }

    CATEGORY = "audio_liam"
    DESCRIPTION = "preview audio"

    RETURN_TYPES = ()

    OUTPUT_NODE = True

    FUNCTION = "load_audio"

    def load_audio(self, audio):
        audio_name = os.path.basename(audio)
        # tmp_path = os.path.dirname(audio)
        # audio_root = os.path.basename(tmp_path)
        subfolder = audio.replace(f"/{audio_name}","")
        print(f"audio: {audio} audio_name: {audio_name}, subfolder: {subfolder}")
        return {"ui": {"audio":[audio_name,subfolder]}}
    
