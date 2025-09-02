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
import librosa

class Assistant:
    def __init__(self, commands_callback, input_device_index, sample_rate=16000, vad_aggressiveness=3, start_threshold=0.5, stop_threshold=0.8):
        self.commands_callback = commands_callback
        self.input_device_index = input_device_index
        self.sample_rate = sample_rate
        self.running = False
        self.thread = None
        self.vad = webrtcvad.Vad(vad_aggressiveness)
        self.start_threshold = start_threshold
        self.stop_threshold = stop_threshold

    def _process_audio(self):
        frame_duration_ms = 30
        frame_size = int(self.sample_rate * frame_duration_ms / 1000)
        ring_buffer = collections.deque(maxlen=20)
        triggered = False
        voiced_frames = []

        with sd.InputStream(samplerate=self.sample_rate, device=self.input_device_index, channels=1, dtype='int16') as stream:
            while self.running:
                try:
                    frame = stream.read(frame_size)[0]
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)

                    if not triggered:
                        ring_buffer.append((frame, is_speech))
                        num_voiced = len([f for f, s in ring_buffer if s])
                        if num_voiced > self.start_threshold * ring_buffer.maxlen:
                            triggered = True
                            voiced_frames.extend([f for f, s in ring_buffer])
                            ring_buffer.clear()
                    else:
                        voiced_frames.append(frame)
                        ring_buffer.append((frame, is_speech))
                        num_unvoiced = len([f for f, s in ring_buffer if not s])
                        if num_unvoiced > self.stop_threshold * ring_buffer.maxlen:
                            triggered = False
                            audio_data = np.concatenate(voiced_frames)
                            ring_buffer.clear()
                            voiced_frames = []
                            
                            if self.commands_callback:
                                callback_thread = threading.Thread(target=self.commands_callback, args=(audio_data,))
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

def find_device_index(name_substring, is_input=False):
    if not name_substring:
        return None
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if name_substring in device['name']:
            if is_input and device['max_input_channels'] > 0:
                return i
            elif not is_input and device['max_output_channels'] > 0:
                return i
    return None

class MentaleseEngine:
    def __init__(self, config, ui_update_queue):
        self.config = config
        self.ui_update_queue = ui_update_queue
        self.stt, self.nllb_tokenizer, self.nllb_model, self.tts = None, None, None, None
        self.stt_lock = threading.Lock()
        self.nllb_lock = threading.Lock()
        self.tts_lock = threading.Lock()
        self.audio_out_queue = queue.Queue()
        self.running = False
        self.user_assistant = None
        self.other_assistant = None
        self.nllb_langs = []
        self.tts_langs = []

    def load_language_lists(self):
        try:
            self.ui_update_queue.put(("log", "Loading language lists..."))
            nllb_path = os.path.join("models", 'nllb')
            tokenizer = AutoTokenizer.from_pretrained(self.config['models']['nllb'], cache_dir=nllb_path)
            self.nllb_langs = sorted(list(tokenizer.lang_code_to_id.keys()))
            self.tts_langs = sorted(["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko"])
            self.ui_update_queue.put(("log", "Language lists loaded."))
            return True
        except Exception as e:
            self.ui_update_queue.put(("log", f"Error loading language lists: {e}"))
            return False

    def _load_heavy_models(self):
        self.ui_update_queue.put(("log", "Loading AI models (this may take a while)..."))
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ui_update_queue.put(("log", f"--- Using device: {device.upper()} ---"))
        
        self.ui_update_queue.put(("log", f"Loading Whisper model: {self.config['models']['whisper']}"))
        self.stt = WhisperModel(self.config['models']['whisper'])
        self.ui_update_queue.put(("log", "Whisper model loaded."))
        
        self.ui_update_queue.put(("log", f"Loading NLLB model: {self.config['models']['nllb']}..."))
        nllb_path = os.path.join("models", 'nllb')
        self.nllb_tokenizer = AutoTokenizer.from_pretrained(self.config['models']['nllb'], cache_dir=nllb_path)
        self.nllb_model = AutoModelForSeq2SeqLM.from_pretrained(self.config['models']['nllb'], cache_dir=nllb_path).to(device)
        self.ui_update_queue.put(("log", "NLLB model loaded."))
        
        self.ui_update_queue.put(("log", f"Loading TTS model: {self.config['models']['tts']}"))
        torch.serialization.add_safe_globals([XttsConfig, XttsAudioConfig, BaseDatasetConfig, XttsArgs])
        os.environ["COQUI_TOS_AGREED"] = "1"
        self.tts = TTS(self.config['models']['tts'], progress_bar=False).to(device)
        self.ui_update_queue.put(("log", "TTS model loaded."))
        self.ui_update_queue.put(("models_loaded", None))

    def start(self):
        if self.running:
            return
        self._load_heavy_models()
        self.running = True
        self._start_pipelines()
        self._start_audio_player()
        self.ui_update_queue.put(("log", "Mentalese Engine Started"))

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.user_assistant:
            self.user_assistant.stop()
        if self.other_assistant:
            self.other_assistant.stop()
        self.ui_update_queue.put(("log", "Mentalese Engine Stopped"))

    def _start_audio_player(self):
        def player_loop():
            while self.running:
                try:
                    wav_output, device_index = self.audio_out_queue.get(timeout=0.1)
                    
                    self.ui_update_queue.put(("log", f"--- Player: Got audio for device {device_index} ---"))
                    self.ui_update_queue.put(("log", f"--- Player: user_output_device={getattr(self, 'user_output_device', 'N/A')}, other_output_device={getattr(self, 'other_output_device', 'N/A')} ---"))

                    assistant_to_mute = None
                    if hasattr(self, 'user_output_device') and device_index == self.user_output_device:
                        self.ui_update_queue.put(("log", "--- Player: Muting OTHER assistant ---"))
                        assistant_to_mute = self.other_assistant
                    elif hasattr(self, 'other_output_device') and device_index == self.other_output_device:
                        self.ui_update_queue.put(("log", "--- Player: Muting USER assistant ---"))
                        assistant_to_mute = self.user_assistant
                    else:
                        self.ui_update_queue.put(("log", "--- Player: No assistant to mute ---"))

                    if assistant_to_mute:
                        assistant_to_mute.stop()

                    self.ui_update_queue.put(("log", f"Playing audio on device {device_index}"))
                    sd.play(np.array(wav_output, dtype='float32'), samplerate=self.tts.synthesizer.output_sample_rate, device=device_index)
                    sd.wait()

                    if assistant_to_mute:
                        self.ui_update_queue.put(("log", f"--- Player: Unmuting assistant ---"))
                        assistant_to_mute.start()

                except queue.Empty:
                    continue
        threading.Thread(target=player_loop, daemon=True).start()

    def _start_pipelines(self):
        device_config = self.config['devices']
        mic_index = find_device_index(device_config['user_mic_name'], is_input=True)
        headphones_index = find_device_index(device_config['headphones_name'], is_input=False)
        cable_input_index = find_device_index(device_config['cable_input_name'], is_input=False)
        cable_output_index = find_device_index(device_config['cable_output_name'], is_input=True)

        if any(i is None for i in [mic_index, headphones_index, cable_input_index, cable_output_index]):
            self.ui_update_queue.put(("log", "ERROR: Could not find all required audio devices."))
            return

        self.user_output_device = cable_input_index
        self.other_output_device = headphones_index

        self.user_assistant = self._create_pipeline_assistant("user", mic_index, cable_input_index)
        self.other_assistant = self._create_pipeline_assistant("other", cable_output_index, headphones_index)
        self.user_assistant.start()
        self.other_assistant.start()

    def _create_pipeline_assistant(self, pipeline_type, input_device, output_device):
        user_voice_file = "user_voice.wav"
        other_voice_file = "other_voice.wav"
        voice_registered = False
        pipeline_config = self.config[f'{pipeline_type}_pipeline']
        vad_config = self.config.get('vad_settings', {})

        def translation_callback(audio_data):
            nonlocal voice_registered
            self.ui_update_queue.put(("log", f"--- {pipeline_type.capitalize()} Pipeline: Captured {len(audio_data) / 16000:.2f}s of audio. ---"))
            try:
                audio_float = audio_data.flatten().astype(np.float32) / 32768.0
                voice_file = user_voice_file if pipeline_type == 'user' else other_voice_file

                if pipeline_config['clone_voice'] and not voice_registered:
                    self.ui_update_queue.put(("log", f"Registering {pipeline_type} voice..."))
                    sf.write(voice_file, audio_float, 16000)
                    voice_registered = True
                    self.ui_update_queue.put(("log", f"{pipeline_type.capitalize()} voice registered."))
                    return

                with self.stt_lock:
                    lang = pipeline_config.get('source_language_tts')
                    segments = self.stt.transcribe(audio_float, language=lang)
                
                for segment in segments:
                    text = segment.text
                    if not text.strip(): continue
                    self.ui_update_queue.put((f'{pipeline_type}_transcription', text))

                    with self.nllb_lock:
                        src_lang = pipeline_config.get('source_language_nllb', 'eng_Latn')
                        tgt_lang = pipeline_config.get('target_language_nllb', 'eng_Latn')
                        self.nllb_tokenizer.src_lang = src_lang
                        inputs = self.nllb_tokenizer(text, return_tensors="pt").to(self.nllb_model.device)
                        translated_tokens = self.nllb_model.generate(**inputs, forced_bos_token_id=self.nllb_tokenizer.lang_code_to_id[tgt_lang])
                        translated_text = self.nllb_tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
                    self.ui_update_queue.put((f'{pipeline_type}_translation', translated_text))

                    with self.tts_lock:
                        lang_tts = pipeline_config.get('target_language_tts', 'en')
                        if pipeline_config['clone_voice'] and voice_registered:
                            wav_output = self.tts.tts(text=translated_text, speaker_wav=voice_file, language=lang_tts)
                        else:
                            wav_output = self.tts.tts(text=translated_text, speaker="Craig Gutsy", language=lang_tts)

                    original_duration_s = (segment.t1 - segment.t0) / 1000.0
                    if original_duration_s > 0.2: # Only stretch if the original audio is not too short
                        generated_duration_s = len(wav_output) / self.tts.synthesizer.output_sample_rate
                        # Define safety limits for the speed rate
                        MIN_RATE = 0.5  # Don't slow down more than half speed
                        MAX_RATE = 2.0  # Don't speed up more than double speed
                        
                        speed_rate = generated_duration_s / original_duration_s if original_duration_s > 0 else 1.0
                        
                        # Apply safety limits
                        if speed_rate > MAX_RATE:
                            speed_rate = MAX_RATE
                        elif speed_rate < MIN_RATE:
                            speed_rate = MIN_RATE

                        stretched_audio = librosa.effects.time_stretch(y=np.array(wav_output, dtype='float32'), rate=speed_rate)
                        self.audio_out_queue.put((stretched_audio, output_device))
                    else:
                        self.audio_out_queue.put((wav_output, output_device))

            except Exception as e:
                self.ui_update_queue.put(("log", f"Error in {pipeline_type} callback: {e}"))

        return Assistant(
            translation_callback, 
            input_device,
            vad_aggressiveness=vad_config.get('aggressiveness', 3),
            start_threshold=vad_config.get('start_threshold', 0.4),
            stop_threshold=vad_config.get('stop_threshold', 0.1)
        )
