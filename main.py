import os
import sounddevice as sd
from pywhispercpp.model import Model as WhisperModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs
from TTS.config.shared_configs import BaseDatasetConfig
import torch
import queue
import threading
import numpy as np
import soundfile as sf
import webrtcvad
import collections
import time
import json

# --- Global State ---
stt, nllb, tts = None, None, None
nllb_tokenizer, nllb_model = None, None

# --- VAD-based Streaming Assistant ---
class Assistant:
    def __init__(self, commands_callback, input_device_index, sample_rate=16000):
        self.commands_callback = commands_callback
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        self.vad = webrtcvad.Vad(3)

    def _process_audio(self):
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        ring_buffer = collections.deque(maxlen=20)
        triggered = False
        voiced_frames = []

        print(f"VAD processing loop started for device {self.input_device_index}.", flush=True)
        with sd.InputStream(samplerate=self.sample_rate, device=self.input_device_index, channels=1, dtype='int16') as stream:
            while self.running:
                try:
                    frame = stream.read(frame_size)[0]
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    print('S' if is_speech else '.', end="", flush=True)

                    if not triggered:
                        ring_buffer.append((frame, is_speech))
                        num_voiced = len([f for f, s in ring_buffer if s])
                        if num_voiced > 0.5 * ring_buffer.maxlen:
                            print(f"\n[Speech Detected on Device {self.input_device_index}!]", flush=True)
                            triggered = True
                            voiced_frames.extend([f for f, s in ring_buffer])
                            ring_buffer.clear()
                    else:
                        voiced_frames.append(frame)
                        ring_buffer.append((frame, is_speech))
                        num_unvoiced = len([f for f, s in ring_buffer if not s])
                        if num_unvoiced > 0.8 * ring_buffer.maxlen:
                            print(f"\n[Speech Ended on Device {self.input_device_index}!]", flush=True)
                            triggered = False
                            audio_data = np.concatenate(voiced_frames)
                            ring_buffer.clear()
                            voiced_frames = []
                            
                            if self.commands_callback:
                                callback_thread = threading.Thread(target=self.commands_callback, args=(audio_data,)) # Corrected: escaped backslash for newline
                                callback_thread.start()
                except Exception as e:
                    print(f"Error in VAD processing thread for device {self.input_device_index}: {e}", flush=True)

    def start(self):
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._process_audio)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        if self.running:
            self.running = False
            if self.thread:
                self.thread.join()

# --- Core Functions ---
def list_audio_devices():
    print("\n" + "-"*20, flush=True)
    print("Available audio devices:", flush=True)
    print(sd.query_devices(), flush=True)
    print("-"*20 + "\n", flush=True)

def find_device_index(name_substring, is_input=False):
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_substring in device['name']:
            if is_input and device['max_input_channels'] > 0:
                return i
            elif not is_input and device['max_output_channels'] > 0:
                return i
    return None

def load_models(config):
    global stt, nllb, tts, nllb_tokenizer, nllb_model
    print("Loading models...", flush=True)
    os.makedirs("models", exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Using device: {device.upper()} ---", flush=True)
    
    stt = WhisperModel(config['models']['whisper'])
    print("Whisper model loaded.", flush=True)
    
    nllb_path = os.path.join("models", 'nllb')
    nllb_tokenizer = AutoTokenizer.from_pretrained(config['models']['nllb'], cache_dir=nllb_path)
    nllb_model = AutoModelForSeq2SeqLM.from_pretrained(config['models']['nllb'], cache_dir=nllb_path).to(device)
    print("NLLB model loaded.", flush=True)
    
    torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
    os.environ["COQUI_TOS_AGREED"] = "1"
    tts = TTS(config['models']['tts'], progress_bar=True).to(device)
    print("TTS model loaded.", flush=True)

# --- Pipeline Workers ---
def user_pipeline_worker(config, stt_lock, nllb_lock, tts_lock, mic_index, cable_input_index, audio_out_queue):
    user_voice_file = "user_voice.wav"
    voice_registered = False
    pipeline_config = config['user_pipeline']

    def translation_callback(audio_data):
        nonlocal voice_registered
        print(f"\n--- User Pipeline: Captured {len(audio_data) / 16000:.2f}s of audio. Processing... ---", flush=True)
        try:
            audio_float = audio_data.flatten().astype(np.float32) / 32768.0

            if pipeline_config['clone_voice'] and not voice_registered:
                print("Registering user voice...", flush=True)
                sf.write(user_voice_file, audio_float, 16000)
                voice_registered = True
                print(f"User voice registered and saved to {user_voice_file}", flush=True)
                return

            with stt_lock:
                text = "".join(segment.text for segment in stt.transcribe(audio_float))
            
            if not text.strip():
                return
            print(f"  > You said: {text}", flush=True)

            with nllb_lock:
                nllb_tokenizer.src_lang = "eng_Latn"
                inputs = nllb_tokenizer(text, return_tensors="pt").to(nllb_model.device)
                translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.lang_code_to_id[pipeline_config['target_language_nllb']])
                translated_text = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"  > Translated for others: {translated_text}", flush=True)

            with tts_lock:
                if pipeline_config['clone_voice'] and voice_registered:
                    wav_output = tts.tts(text=translated_text, speaker_wav=user_voice_file, language=pipeline_config['target_language_tts'])
                else:
                    wav_output = tts.tts(text=translated_text, language=pipeline_config['target_language_tts'])
            
            audio_out_queue.put((wav_output, cable_input_index))

        except Exception as e:
            print(f"Error in user pipeline callback: {e}", flush=True)

    assistant = Assistant(translation_callback, mic_index)
    assistant.start()
    print("User pipeline started.", flush=True)

def other_pipeline_worker(config, stt_lock, nllb_lock, tts_lock, cable_output_index, headphones_index, audio_out_queue):
    other_voice_file = "other_voice.wav"
    voice_registered = False
    pipeline_config = config['other_pipeline']

    def translation_callback(audio_data):
        nonlocal voice_registered
        print(f"\n--- Other Pipeline: Captured {len(audio_data) / 16000:.2f}s of audio. Processing... ---", flush=True)
        try:
            audio_float = audio_data.flatten().astype(np.float32) / 32768.0

            if pipeline_config['clone_voice'] and not voice_registered:
                print("Registering other's voice...", flush=True)
                sf.write(other_voice_file, audio_float, 16000)
                voice_registered = True
                print(f"Other's voice registered and saved to {other_voice_file}", flush=True)
                return

            with stt_lock:
                text = "".join(segment.text for segment in stt.transcribe(audio_float, language=pipeline_config['source_language_tts']))
            
            if not text.strip():
                return
            print(f"  > They said: {text}", flush=True)

            with nllb_lock:
                nllb_tokenizer.src_lang = pipeline_config['source_language_nllb']
                inputs = nllb_tokenizer(text, return_tensors="pt").to(nllb_model.device)
                translated_tokens = nllb_model.generate(**inputs, forced_bos_token_id=nllb_tokenizer.lang_code_to_id["eng_Latn"])
                translated_text = nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            print(f"  > Translated for you: {translated_text}", flush=True)

            with tts_lock:
                if pipeline_config['clone_voice'] and voice_registered:
                    wav_output = tts.tts(text=translated_text, speaker_wav=other_voice_file, language="en")
                else:
                    wav_output = tts.tts(text=translated_text, language="en")
            
            audio_out_queue.put((wav_output, headphones_index))

        except Exception as e:
            print(f"Error in other pipeline callback: {e}", flush=True)

    assistant = Assistant(translation_callback, cable_output_index)
    assistant.start()
    print("Other pipeline started.", flush=True)

if __name__ == "__main__":
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except FileNotFoundError:
        print("FATAL: config.json not found. Please create it.", flush=True)
        exit()
    except json.JSONDecodeError:
        print("FATAL: config.json is not a valid JSON file.", flush=True)
        exit()

    list_audio_devices()
    load_models(config)

    device_config = config['devices']
    mic_index = find_device_index(device_config['user_mic_name'], is_input=True)
    headphones_index = find_device_index(device_config['headphones_name'], is_input=False)
    cable_input_index = find_device_index(device_config['cable_input_name'], is_input=False)
    cable_output_index = find_device_index(device_config['cable_output_name'], is_input=True)

    if any(i is None for i in [mic_index, headphones_index, cable_input_index, cable_output_index]):
        print("\n---! ERROR !---", flush=True)
        print("Could not find all required audio devices. Please check your device names in config.json.", flush=True)
        exit()

    print("\n--- All audio devices found successfully ---", flush=True)
    print(f"Mic: {mic_index}, Headphones: {headphones_index}, Cable In: {cable_input_index}, Cable Out: {cable_output_index}", flush=True)

    print("\nMentalese is ready for a two-way conversation.", flush=True)
    
    stt_lock = threading.Lock()
    nllb_lock = threading.Lock()
    tts_lock = threading.Lock()
    audio_out_queue = queue.Queue()

    user_thread = threading.Thread(target=user_pipeline_worker, args=(config, stt_lock, nllb_lock, tts_lock, mic_index, cable_input_index, audio_out_queue))
    other_thread = threading.Thread(target=other_pipeline_worker, args=(config, stt_lock, nllb_lock, tts_lock, cable_output_index, headphones_index, audio_out_queue))
    user_thread.daemon = True
    other_thread.daemon = True
    user_thread.start()
    other_thread.start()

    try:
        while True:
            try:
                wav_output, device_index = audio_out_queue.get(timeout=0.1)
                sd.play(np.array(wav_output, dtype='float32'), samplerate=tts.synthesizer.output_sample_rate, device=device_index)
                sd.wait()
            except queue.Empty:
                continue
    except KeyboardInterrupt:
        print("\nApplication stopped by user.", flush=True)
