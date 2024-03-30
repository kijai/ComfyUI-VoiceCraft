import os
import torch
import torchaudio

from .data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
)
from .models import voicecraft

import model_management as mm
import comfy.utils
import folder_paths
script_directory = os.path.dirname(os.path.abspath(__file__))
espeak_lib_path = os.path.join(script_directory, 'espeak-ng', 'libespeak-ng.dll')

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"  
    
class voicecraft_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "espeak_library_path": ("STRING", {"default": espeak_lib_path, "multiline":True}),
            },
            
        }

    RETURN_TYPES = ("VCMODEL",)
    RETURN_NAMES = ("voicecraft_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "VoiceCraft"

    def loadmodel(self, espeak_library_path):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
  

        if not hasattr(self, 'model') or self.model == None:
           
            if espeak_library_path != "":
                TextTokenizer.set_library(espeak_library_path)

            model_path = os.path.join(folder_paths.models_dir,'voicecraft')

            text_tokenizer = TextTokenizer(backend="espeak")
           
            voicecraft_name="giga830M.pth"
            
            if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id="pyp1/VoiceCraft", ignore_patterns=["giga330M.pth"], 
                                    local_dir=model_path, local_dir_use_symlinks=False)
            state_dict = torch.load(os.path.join(model_path, voicecraft_name), map_location="cpu")

            encodec_fn = os.path.join(model_path,"encodec_4cb2048_giga.th")

            audio_tokenizer = AudioTokenizer(signature=encodec_fn)

            phn2num = state_dict['phn2num']
           
            config = state_dict["config"]
            self.model = voicecraft.VoiceCraft(config)
            self.model.load_state_dict(state_dict["model"])
            self.model.eval()
            self.model.to(device)
            voicecraft_model = {
                "text_tokenizer": text_tokenizer,
                "audio_tokenizer": audio_tokenizer,
                "model": self.model,
                "phn2num": phn2num,
                "config": config,
            }

        return (voicecraft_model,)
    
class voicecraft_process:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "voicecraft_model": ("VCMODEL",),
            "original_audio_path": ("STRING", {"default": os.path.join(script_directory, 'demo', 'YOU1000000023_S0000047.wav'), "multiline":True}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            "cut_off_sec": ("FLOAT", {"default": 3.01, "min": 0, "max": 4096, "step": 0.01}),
            "top_k": ("INT", {"default": 0, "min": 0, "max": 1024}),
            "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0, "max": 100}),
            "stop_repetition": ("INT", {"default": 3, "min": 0, "max": 1024}),
            "sample_batch_size": ("INT", {"default": 4, "min": 1, "max": 1024}),
            #"orig_transcript": ("STRING", {"default": "Hello world!", "multiline":True}),
            "target_transcript": ("STRING", {"default": "To unlock the multitude of control types and artistic flows possible with AI, we want to build tooling and infrastructure to empower a community of tool-builders, who in-turn empower a world of budding artists.", "multiline":True}),
            },
    
        }

    RETURN_TYPES = ("VHS_AUDIO", "INT", "STRING", )
    RETURN_NAMES = ("vhs_audio", "audio_dur", "gen_audio_path", )
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, voicecraft_model,cut_off_sec, original_audio_path, seed, target_transcript, top_k, top_p, temperature, stop_repetition, sample_batch_size):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        # hyperparameters for inference
        left_margin = 0.08 # not used for TTS, only for speech editing
        right_margin = 0.08 # not used for TTS, only for speech editing
        codec_audio_sr = 16000
        codec_sr = 50
        kvcache = 1
        silence_tokens=[1388,1898,131]
        
        # adjust the below three arguments if the generation is not as good
        #stop_repetition = 3 # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
        #sample_batch_size = 4 # if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4
        torch.manual_seed(seed)

        decode_config = {
            'top_k': top_k, 
            'top_p': top_p, 
            'temperature': temperature, 
            'stop_repetition': stop_repetition, 
            'kvcache': kvcache, 
            "codec_audio_sr": codec_audio_sr, 
            "codec_sr": codec_sr, 
            "silence_tokens": silence_tokens, 
            "sample_batch_size": sample_batch_size
            }
        #demo_dir = os.path.join(script_directory,'demo')
        #orig_audio = os.path.join(demo_dir, "84_121550_000074_000000.wav")
        #orig_transcript = "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,"
        #temp_folder = os.path.join(demo_dir, "temp")
       # os.makedirs(temp_folder, exist_ok=True)
        #os.system(f"copy {orig_audio} {temp_folder}")
        #filename = os.path.splitext(orig_audio.split("/")[-1])[0]
        #with open(os.path.join(temp_folder, filename,f"{filename}.txt"), "w") as f:
        #    f.write(orig_transcript)
        # run MFA to get the alignment
        #align_temp = os.path.join(temp_folder,"mfa_alignments")
        #os.makedirs(align_temp, exist_ok=True)
        #os.system(f"mfa align -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}")
        #audio_fn = os.path.join(temp_folder,f"{filename}.wav")
        audio_fn = original_audio_path
        #transcript_fn = f"{temp_folder}/{filename}.txt"
        #align_fn = f"{align_temp}/{filename}.csv"
        #cut_off_sec = 3.01 # NOTE: according to forced-alignment file, the word "common" stop as 3.01 sec, this should be different for different audio
        #target_transcript = "We believe that AI has the potential to allow billions to experience creative fulfilment this century. However, in order to reach this potential, artistic control is key - it's the difference between something feeling like it was made by you rather than for you. To unlock the multitude of control types and artistic flows possible with AI, we want to build tooling and infrastructure to empower a community of tool-builders, who in-turn empower a world of budding artists."
        info = torchaudio.info(audio_fn)
        audio_dur = info.num_frames / info.sample_rate

        assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
        prompt_end_frame = int(cut_off_sec * info.sample_rate)
        #prompt_end_frame = 0

        from .inference_tts_scale import inference_one_sample
        concated_audio, gen_audio = inference_one_sample(
            voicecraft_model["model"], 
            voicecraft_model["config"], 
            voicecraft_model["phn2num"], 
            voicecraft_model["text_tokenizer"], 
            voicecraft_model["audio_tokenizer"], 
            audio_fn, 
            target_transcript, 
            device, 
            decode_config, 
            prompt_end_frame
            )
        
        # save segments for comparison
        concated_audio, gen_audio = concated_audio[0].cpu(), gen_audio[0].cpu()
        print(type(concated_audio), type(gen_audio))   

        # Define the sample rate
        sample_rate = codec_audio_sr

        # Save generated audio to a WAV file
        gen_audio_path = os.path.join(script_directory, "temp", "gen_audio.wav")
        torchaudio.save(gen_audio_path, gen_audio, sample_rate)
        gen_audio_info = torchaudio.info(gen_audio_path)
        gen_audio_dur = gen_audio_info.num_frames / gen_audio_info.sample_rate
        #print(f"Generated audio saved to {gen_audio_path}")
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            imageio_ffmpeg_path = get_ffmpeg_exe()
        except:
            print("Failed to import imageio_ffmpeg")
        args = [imageio_ffmpeg_path, "-v", "error", "-i", gen_audio_path]
        try:
            import subprocess
            res =  subprocess.run(args + ["-f", "wav", "-"],
                                stdout=subprocess.PIPE, check=True).stdout
        except:
            print("Failed to run ffmpeg")
  
        audio_lambda = lambda: res
        return (audio_lambda, gen_audio_dur, gen_audio_path)

NODE_CLASS_MAPPINGS = {
    "voicecraft_model_loader": voicecraft_model_loader,
    "voicecraft_process": voicecraft_process,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "voicecraft_model_loader": "VoiceCraft Model Loader",
    "voicecraft_process": "VoiceCraft Process",
}
