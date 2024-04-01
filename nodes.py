import os
import torch
import torchaudio
import time

from .data.tokenizer import (
    AudioTokenizer,
    TextTokenizer,
    tokenize_audio,
    tokenize_text
)
from .models import voicecraft

import model_management as mm
import comfy.utils
import folder_paths
import platform
script_directory = os.path.dirname(os.path.abspath(__file__))
folder_paths.add_model_folder_path("voicecraft_samples", os.path.join(script_directory, "demo"))
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0"

    
class voicecraft_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            
            },
        }
        

    RETURN_TYPES = ("VCMODEL",)
    RETURN_NAMES = ("voicecraft_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "VoiceCraft"

    def loadmodel(self, espeak_library_path=""):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
  
        if not hasattr(self, 'model') or self.model == None:
            if espeak_library_path == "" and platform.system() == "Windows":
                print("espeak_library_path not set, using default")
                espeak_library_path_windows = os.path.join(script_directory, 'espeak-ng', 'libespeak-ng.dll')
                espeak_library_path = espeak_library_path_windows
                print("Setting espeak_library_path: ", espeak_library_path_windows)
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

class audiocraft_model_loader:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audiocraft_model": (
            [   
                'musicgen-small',
                'musicgen-medium',
                'musicgen-large',
                'musicgen-melody',
                'musicgen-melody-large',
                'musicgen-small-stereo',
                'musicgen-medium-stereo',
                'musicgen-large-stereo',
                'musicgen-melody-stereo',
                'musicgen-melody-large-stereo',
                'magnet-small-10secs',
                'magnet-small-30secs',
                'magnet-medium-10secs',
                'magnet-medium-30secs',
                'magnet-medium-10secs',
                'audio-magnet-small',
                'audio-magnet-medium'

            ],
            {
            "default": 'musicgen-melody'
             }),
            },
        }

    RETURN_TYPES = ("ACMODEL",)
    RETURN_NAMES = ("audiocraft_model",)
    FUNCTION = "loadmodel"
    CATEGORY = "VoiceCraft"
    DESCRIPTION = "Loads (and downloads) Facebook's audiocraft model from Hugginface to ComfyUI/models/audiocraft"

    def loadmodel(self, audiocraft_model):
        mm.soft_empty_cache()
        device = mm.get_torch_device()
        
        if not hasattr(self, 'model') or self.model == None or audiocraft_model != audiocraft_model["model_name"]:
            model_path = os.path.join(folder_paths.models_dir,'audiocraft',audiocraft_model)
            
            if not os.path.exists(model_path):
                from huggingface_hub import snapshot_download
                snapshot_download(repo_id=f"facebook/{audiocraft_model}", ignore_patterns=["*.safetensors"], 
                                    local_dir=model_path, local_dir_use_symlinks=False)
            if "magnet" in audiocraft_model:
                from audiocraft.models import MAGNeT
                self.model = MAGNeT.get_pretrained(model_path)
            if "musicgen" in audiocraft_model:
                from audiocraft.models import MusicGen
                self.model = MusicGen.get_pretrained(model_path)
            audiocraft_model = {
                "model": self.model,
                "model_name": audiocraft_model
            }

        return (audiocraft_model,)
    
class voicecraft_process:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "voicecraft_model": ("VCMODEL",),
            "audio_tensor": ("VCAUDIOTENSOR", ),
            "sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            #"cut_off_sec": ("FLOAT", {"default": 3.01, "min": 0, "max": 4096, "step": 0.01}),
            "top_k": ("FLOAT", {"default": 0, "min": 0, "max": 1024, "step": 0.01}),
            "top_p": ("FLOAT", {"default": 0.8, "min": 0, "max": 1, "step": 0.01}),
            "temperature": ("FLOAT", {"default": 1.0, "min": 0, "max": 100, "step": 0.01}),
            "stop_repetition": ("INT", {"default": 3, "min": 0, "max": 1024, "step": 1}),
            "sample_batch_size": ("INT", {"default": 4, "min": 1, "max": 1024, "step": 1}),
            "target_transcript": ("STRING", {"default": "But when I had approached so near to them The common To unlock the multitude of control types and artistic flows possible with AI, we want to build tooling and infrastructure to empower a community of tool-builders, who in-turn empower a world of budding artists.", "multiline":True}),
            },
        }

    RETURN_TYPES = ("VHS_AUDIO", "INT", "STRING", "VCAUDIOTENSOR",)
    RETURN_NAMES = ("vhs_audio", "audio_dur", "gen_audio_path", "audio_tensor",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audio_tensor, sample_rate, voicecraft_model, seed, target_transcript, top_k, top_p, temperature, stop_repetition, sample_batch_size):
        device = mm.get_torch_device()
        mm.soft_empty_cache()
        offload_device = mm.unet_offload_device()
        torch.manual_seed(seed)
        # hyperparameters for inference
        #left_margin = 0.08 # not used for TTS, only for speech editing
        #right_margin = 0.08 # not used for TTS, only for speech editing
        codec_audio_sr = 16000
        codec_sr = 50
        kvcache = 1
        silence_tokens=[1388,1898,131]
        
        # adjust the below three arguments if the generation is not as good
        #stop_repetition = 3 # if there are long silence in the generated audio, reduce the stop_repetition to 3, 2 or even 1
        #sample_batch_size = 4 # if there are long silence or unnaturally strecthed words, increase sample_batch_size to 2, 3 or even 4
        
        #demo_dir = os.path.join(script_directory,'demo')
        #orig_audio = os.path.join(demo_dir, "84_121550_000074_000000.wav")
        #orig_transcript = "But when I had approached so near to them The common object, which the sense deceives, Lost not by distance any of its marks,"
        #temp_folder = os.path.join(demo_dir, "temp")
        #os.makedirs(temp_folder, exist_ok=True)
        #os.system(f"copy {orig_audio} {temp_folder}")
        #filename = os.path.splitext(orig_audio.split("/")[-1])[0]
        #with open(os.path.join(temp_folder, filename,f"{filename}.txt"), "w") as f:
        #    f.write(orig_transcript)
        # run MFA to get the alignment
        #align_temp = os.path.join(temp_folder,"mfa_alignments")
        #os.makedirs(align_temp, exist_ok=True)
        #os.system(f"mfa align -j 1 --output_format csv {temp_folder} english_us_arpa english_us_arpa {align_temp}")
        #audio_fn = os.path.join(temp_folder,f"{filename}.wav")
        #audio_fn = folder_paths.get_full_path("voicecraft_samples", original_sample)
        #transcript_fn = f"{temp_folder}/{filename}.txt"
        #align_fn = f"{align_temp}/{filename}.csv"
        #cut_off_sec = 3.01 # NOTE: according to forced-alignment file, the word "common" stop as 3.01 sec, this should be different for different audio
        #target_transcript = "We believe that AI has the potential to allow billions to experience creative fulfilment this century. However, in order to reach this potential, artistic control is key - it's the difference between something feeling like it was made by you rather than for you. To unlock the multitude of control types and artistic flows possible with AI, we want to build tooling and infrastructure to empower a community of tool-builders, who in-turn empower a world of budding artists."
        #info = torchaudio.info(audio_fn)
        audio_dur = audio_tensor.shape[-1] / sample_rate

        #assert cut_off_sec < audio_dur, f"cut_off_sec {cut_off_sec} is larger than the audio duration {audio_dur}"
        #prompt_end_frame = int(cut_off_sec * sample_rate)
    
        # phonemize
        text_tokens = [voicecraft_model["phn2num"][phn] for phn in
                tokenize_text(
                    voicecraft_model["text_tokenizer"], text=target_transcript.strip()
                ) if phn in voicecraft_model["phn2num"]
            ]
        text_tokens = torch.LongTensor(text_tokens).unsqueeze(0)
        text_tokens_lens = torch.LongTensor([text_tokens.shape[-1]])

        # encode audio
        #if audio_tensor == None:
            #encoded_frames = tokenize_audio(voicecraft_model["audio_tokenizer"], audio_fn, offset=0, num_frames=prompt_end_frame)
        #else:
        encoded_frames = voicecraft_model["audio_tokenizer"].encode(audio_tensor.unsqueeze(0))

        original_audio = encoded_frames[0][0].transpose(2,1) # [1,T,K]
        assert original_audio.ndim==3 and original_audio.shape[0] == 1 and original_audio.shape[2] == voicecraft_model["config"].n_codebooks, original_audio.shape
        print(f"original audio length: {original_audio.shape[1]} codec frames, which is {original_audio.shape[1]/codec_sr:.2f} sec.")

        # forward
        voicecraft_model["model"].to(device)
        stime = time.time()
        if sample_batch_size <= 1:
            print(f"running inference with batch size 1")
            concat_frames, gen_frames = voicecraft_model["model"].inference_tts(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                original_audio[...,:voicecraft_model["config"].n_codebooks].to(device), # [1,T,8]
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stop_repetition=stop_repetition,
                kvcache=kvcache,
                silence_tokens=eval(silence_tokens) if type(silence_tokens)==str else silence_tokens
            ) # output is [1,K,T]
        else:
            print(f"running inference with batch size {sample_batch_size}, i.e. return the shortest among {sample_batch_size} generations.")
            concat_frames, gen_frames = voicecraft_model["model"].inference_tts_batch(
                text_tokens.to(device),
                text_tokens_lens.to(device),
                original_audio[...,:voicecraft_model["config"].n_codebooks].to(device), # [1,T,8]
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                stop_repetition=stop_repetition,
                kvcache=kvcache,
                batch_size = sample_batch_size,
                silence_tokens=eval(silence_tokens) if type(silence_tokens)==str else silence_tokens
            ) # output is [1,K,T]
        voicecraft_model["model"].to(offload_device)
        print(f"inference on one sample take: {time.time() - stime:.4f} sec.")

        print(f"generated encoded_frames.shape: {gen_frames.shape}, which is {gen_frames.shape[-1]/codec_sr} sec.")
        
        # concat_sample = voicecraft_model["audio_tokenizer"].decode(
        #     [(concat_frames, None)] # [1,T,8] -> [1,8,T]
        # )
             
        gen_sample = voicecraft_model["audio_tokenizer"].decode(
            [(gen_frames, None)]
        )
        gen_audio_tensor = gen_sample[0].cpu()

        # Save generated audio to a WAV file
        gen_audio_path = os.path.join(script_directory, "temp", "gen_audio.wav")
        torchaudio.save(gen_audio_path, gen_audio_tensor, sample_rate)
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

        return (audio_lambda, gen_audio_dur, gen_audio_path, gen_audio_tensor,)
class audio_tensor_to_vhs_audio:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_tensor": ("VCAUDIOTENSOR",),
            "sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
           
             },
    
        }

    RETURN_TYPES = ("VHS_AUDIO", "INT",)
    RETURN_NAMES = ("vhs_audio", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audio_tensor, sample_rate):

        # Save generated audio to a WAV file
        audio_path = os.path.join(script_directory, "temp", "gen_audio.wav")
        
        torchaudio.save(audio_path, audio_tensor, sample_rate)
        audio_info = torchaudio.info(audio_path)
        
        audio_dur = audio_info.num_frames / audio_info.sample_rate
 
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            imageio_ffmpeg_path = get_ffmpeg_exe()
        except:
            print("Failed to import imageio_ffmpeg")
        args = [imageio_ffmpeg_path, "-v", "error", "-i", audio_path]
        try:
            import subprocess
            res =  subprocess.run(args + ["-f", "wav", "-"],
                                stdout=subprocess.PIPE, check=True).stdout
        except:
            print("Failed to run ffmpeg")
  
        audio_lambda = lambda: res

        # Return the new audio_lambda
        return (audio_lambda, audio_dur,)
    
class vhs_audio_to_audio_tensor:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "vhs_audio": ("VHS_AUDIO",),
            "target_sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
            "target_channels": ("INT", {"default": 1, "min": 1, "max": 2}),
             },
    
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT",)
    RETURN_NAMES = ("audio_tensor", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, vhs_audio, target_sample_rate, target_channels):
        import io
        # Convert the byte stream to a tensor
        audio_bytes = vhs_audio()
        audio_buffer = io.BytesIO(audio_bytes)
        audio_tensor, sample_rate = torchaudio.load(audio_buffer)
        assert audio_tensor.shape[0] in [1, 2], "Audio must be mono or stereo."
        if target_channels == 1:
            audio_tensor = audio_tensor.mean(0, keepdim=True)
        elif target_channels == 2:
            *shape, _, length = audio_tensor.shape
            audio_tensor = audio_tensor.expand(*shape, target_channels, length)
        elif audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.expand(target_channels, -1)
        resampled_audio_tensor = torchaudio.functional.resample(audio_tensor, sample_rate, target_sample_rate)
        audio_dur = audio_tensor.shape[1] / target_sample_rate
        
        return (resampled_audio_tensor, audio_dur,)

class audio_tensor_concat:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_tensor1": ("VCAUDIOTENSOR",),
            "audio_tensor2": ("VCAUDIOTENSOR",),
             },
    
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT",)
    RETURN_NAMES = ("audio_tensor", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audio_tensor1, audio_tensor2):
        assert audio_tensor1.size(0) == audio_tensor2.size(0), "Tensors must have the same number of channels"

        concatenated_audio = torch.cat((audio_tensor1, audio_tensor2), dim=1)

        sample_rate = 16000
        audio_dur = concatenated_audio.shape[0] / sample_rate

        return (concatenated_audio, audio_dur,)
    
class audio_tensor_split:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_tensor": ("VCAUDIOTENSOR",),
            "sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
            "split_at_seconds": ("FLOAT", {"default": 8, "min": 0, "max": 4096, "step": 0.01}),
             },
    
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "VCAUDIOTENSOR",)
    RETURN_NAMES = ("audio_tensor1", "audio_tensor2",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audio_tensor, split_at_seconds, sample_rate):
        split_at_frames = int(split_at_seconds * sample_rate)
        audio_tensor1 = audio_tensor[:, :split_at_frames]
        audio_tensor2 = audio_tensor[:, split_at_frames:]

        return (audio_tensor1, audio_tensor2,)

class musicgen:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                "audiocraft_model": ("ACMODEL",),
                "sample_rate": ("INT", {"default": 16000, "min": 0, "max": 160000}),
                "duration": ("FLOAT", {"default": 8, "min": 0, "max": 4096, "step": 0.01}),
                "description": ("STRING", {"default": "happy rock", "multiline":True}),
             },
             "optional": {
                "melody": ("VCAUDIOTENSOR",),
             }
    
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT",)
    RETURN_NAMES = ("audio_tensor", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audiocraft_model, sample_rate, duration, description, melody=None):
        model = audiocraft_model['model']
        if "musicgen" in audiocraft_model['model_name']:
            model.set_generation_params(duration=duration) 
        if melody != None and description != "":
            audio_tensor = model.generate_with_chroma([description], melody, sample_rate)
        elif description == "" and melody is None:
            audio_tensor = model.generate_unconditional(1)
        elif melody != None and description == "":
            audio_tensor = model.generate(melody)
        else:
            audio_tensor = model.generate(description) 
        
        print(audio_tensor.shape)        
        audio_dur = audio_tensor.shape[2] / sample_rate
        return (audio_tensor[0].cpu(), audio_dur,)

class audio_tensor_enhance:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "audio_tensor": ("VCAUDIOTENSOR",),
            "original_sample_rate": ("INT", {"default": 16000, "min": 0, "max": 48000}),
             },
        }

    RETURN_TYPES = ("VCAUDIOTENSOR", "INT",)
    RETURN_NAMES = ("audio_tensor", "audio_dur",)
    FUNCTION = "process"
    CATEGORY = "VoiceCraft"

    def process(self, audio_tensor, original_sample_rate):
        from df.enhance import enhance, init_df, resample
        model, df_state, _ = init_df()
        sr = df_state.sr()
        print("df_state.sr: ",sr)

        audio = resample(audio_tensor, original_sample_rate, sr)
        enhanced = enhance(model, df_state, audio)
        return (enhanced,)
    
class whisper_transcription:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": { 
                "audio_tensor" : ("VCAUDIOTENSOR",),
                "whisper_model": (
                    [
                    "base",
                    "tiny",
                    "small",
                    "medium",
                    "large"
                    ]
                    ,),
                "search_word": ("STRING", {"default": "", "multiline":False}),
            }
        }

    RETURN_TYPES = ("STRING", "FLOAT", )
    RETURN_NAMES = ("transcript", "word_end_time",)
    FUNCTION = "whispertranscribe"
    CATEGORY = "VoiceCraft"

    def whispertranscribe(self, audio_tensor, whisper_model, search_word):
        import whisper
        import string
        # Define a translation table to remove punctuation
        remove_punct_table = str.maketrans('', '', string.punctuation + ' ')
        model = whisper.load_model(whisper_model)
        result = model.transcribe(audio_tensor.squeeze(0), word_timestamps=True)
   
        # Initialize variable to store the end time of the searched word
        word_end_time = None

        # Iterate over segments to access word timestamps
        for segment in result['segments']:
            # Iterate over words in the segment to access word timestamps
            for word in segment['words']:
                print(f"Transcribed word: {word['word']} - End time: {word['end']}")

                word_text = word['word'].translate(remove_punct_table).lower()
                word_end = word['end']

                # Check if the current word matches the search word
                if word_text == search_word.translate(remove_punct_table).lower():
                    word_end_time = word_end
                    break # Exit the loop once the word is found

            # If the word is found, break the outer loop as well
            if word_end_time is not None:
                break

        if word_end_time is not None:
            return (result["text"].strip(), word_end_time)
        else:
            return (result["text"].strip(), None)
    
NODE_CLASS_MAPPINGS = {
    "voicecraft_model_loader": voicecraft_model_loader,
    "voicecraft_process": voicecraft_process,
    "audio_tensor_concat": audio_tensor_concat,
    "audio_tensor_to_vhs_audio": audio_tensor_to_vhs_audio,
    "vhs_audio_to_audio_tensor": vhs_audio_to_audio_tensor,
    "musicgen": musicgen,
    "audiocraft_model_loader": audiocraft_model_loader,
    "audio_tensor_enhance": audio_tensor_enhance,
    "audio_tensor_split": audio_tensor_split,
    "whisper_transcription": whisper_transcription
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "voicecraft_model_loader": "VoiceCraft Model Loader",
    "voicecraft_process": "VoiceCraft Process",
    "audio_tensor_concat": "Audio Tensor Concat",
    "audio_tensor_to_vhs_audio": "Audio Tensor To VHS Audio",
    "vhs_audio_to_audio_tensor": "VHS Audio To Audio Tensor",
    "musicgen": "MusicGen",
    "audiocraft_model_loader": "AudioCraft Model Loader",
    "audio_tensor_enhance": "Audio Tensor Enhance",
    "audio_tensor_split": "Audio Tensor Split",
    "whisper_transcription": "Whisper Transcription"
}
