import flet as ft
from engine import MentaleseEngine, find_device_index
import queue
import threading
import json
import sounddevice as sd

def check_vb_cable():
    try:
        devices = sd.query_devices()
        cable_input_found = any("CABLE Input" in d['name'] for d in devices if d['max_output_channels'] > 0)
        cable_output_found = any("CABLE Output" in d['name'] for d in devices if d['max_input_channels'] > 0)
        return cable_input_found and cable_output_found
    except Exception:
        return False

def main(page: ft.Page):
    page.title = "Mentalese Real-Time Translator"
    page.window_width = 600
    page.window_height = 700
    page.horizontal_alignment = ft.CrossAxisAlignment.CENTER

    if not check_vb_cable():
        page.title = "VB-CABLE Driver Not Found"
        page.add(ft.Text("VB-CABLE driver not found.", size=16, weight=ft.FontWeight.BOLD))
        page.add(ft.Text("Please install it to use Mentalese:"))
        page.add(ft.Text("1. Close this application."))
        page.add(ft.Text("2. Find 'VBCABLE_Setup_x64.exe' in the application folder."))
        page.add(ft.Text("3. Right-click it and select 'Run as administrator'."))
        page.add(ft.Text("4. Restart the application after installation."))
        page.update()
        return

    # --- Config Loading ---
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
    except Exception as e:
        page.add(ft.Text(f"Error loading config.json: {e}"))
        return

    # --- UI State & Engine ---
    ui_update_queue = queue.Queue()
    engine = MentaleseEngine(config, ui_update_queue)
    if not engine.load_language_lists():
        page.add(ft.Text("Error: Could not load language lists from models."))
        return
    
    is_running = ft.Ref[bool]()
    is_running.current = False

    # --- UI References ---
    log_view = ft.ListView(expand=True, spacing=10, auto_scroll=True)
    user_transcription_text = ft.Text("...")
    user_translation_text = ft.Text("...")
    other_transcription_text = ft.Text("...")
    other_translation_text = ft.Text("...")
    start_button = ft.Ref[ft.ElevatedButton]()
    stop_button = ft.Ref[ft.ElevatedButton]()
    loading_indicator = ft.Ref[ft.ProgressRing]()

    # --- UI Update Worker ---
    def ui_update_worker():
        while is_running.current:
            try:
                update_type, data = ui_update_queue.get(timeout=0.1)
                if update_type == "models_loaded":
                    loading_indicator.current.visible = False
                    stop_button.current.disabled = False
                elif update_type == "log":
                    log_view.controls.append(ft.Text(data, size=12))
                elif update_type == "user_transcription":
                    user_transcription_text.value = data
                elif update_type == "user_translation":
                    user_translation_text.value = data
                elif update_type == "other_transcription":
                    other_transcription_text.value = data
                elif update_type == "other_translation":
                    other_translation_text.value = data
                page.update()
            except queue.Empty:
                continue

    # --- Event Handlers ---
    def start_engine(e):
        is_running.current = True
        start_button.current.disabled = True
        loading_indicator.current.visible = True
        log_view.controls.clear()
        threading.Thread(target=engine.start, daemon=True).start()
        threading.Thread(target=ui_update_worker, daemon=True).start()
        page.update()

    def stop_engine(e):
        is_running.current = False
        start_button.current.disabled = False
        stop_button.current.disabled = True
        engine.stop()
        page.update()

    def save_settings(e):
        try:
            new_config = {
                "models": {
                    "whisper": whisper_model_dd.value,
                    "nllb": nllb_model_dd.value,
                    "tts": tts_model_tf.value
                },
                "devices": {
                    "user_mic_name": mic_dd.value,
                    "headphones_name": headphones_dd.value,
                    "cable_input_name": cable_in_dd.value,
                    "cable_output_name": cable_out_dd.value
                },
                "vad_settings": {
                    "aggressiveness": int(vad_aggressiveness_sl.value),
                    "start_threshold": round(vad_start_sl.value, 2),
                    "stop_threshold": round(vad_stop_sl.value, 2)
                },
                "user_pipeline": {
                    "clone_voice": clone_user_voice_sw.value,
                    "target_language_nllb": user_lang_nllb_dd.value,
                    "target_language_tts": user_lang_tts_dd.value
                },
                "other_pipeline": {
                    "clone_voice": clone_other_voice_sw.value,
                    "source_language_nllb": other_lang_nllb_dd.value,
                    "source_language_tts": other_lang_tts_dd.value
                }
            }
            with open("config.json", "w") as f:
                json.dump(new_config, f, indent=4)
            save_status.value = "Settings saved! Please restart the application."
            save_status.color = "green"
        except Exception as ex:
            save_status.value = f"Error saving settings: {ex}"
            save_status.color = "red"
        page.update()

    # --- UI Controls ---
    start_button.current = ft.ElevatedButton("Start Engine", on_click=start_engine, icon="play_arrow")
    stop_button.current = ft.ElevatedButton("Stop Engine", on_click=stop_engine, disabled=True, icon="stop")
    loading_indicator.current = ft.ProgressRing(width=16, height=16, stroke_width=2, visible=False)

    translator_view = ft.Container(
        content=ft.Column([
            ft.Row([start_button.current, stop_button.current, loading_indicator.current], alignment=ft.MainAxisAlignment.CENTER),
            ft.Divider(),
            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Your Voice", weight=ft.FontWeight.BOLD), ft.Text("You said:"), user_transcription_text, ft.Text("Translated for others:"), user_translation_text,
            ]), padding=10)),
            ft.Card(content=ft.Container(ft.Column([
                ft.Text("Their Voice", weight=ft.FontWeight.BOLD), ft.Text("They said:"), other_transcription_text, ft.Text("Translated for you:"), other_translation_text,
            ]), padding=10)),
            ft.Divider(), ft.Text("Logs:"),
            ft.Container(log_view, border=ft.border.all(1, "outline"), border_radius=ft.border_radius.all(5), padding=10, expand=True)
        ]), padding=20
    )

    vad_settings = config.get('vad_settings', {})
    vad_aggressiveness_sl = ft.Slider(min=0, max=3, divisions=3, label="Aggressiveness: {value}", value=vad_settings.get('aggressiveness', 3))
    vad_start_sl = ft.Slider(min=0, max=1, label="Start Threshold: {value}", value=vad_settings.get('start_threshold', 0.5))
    vad_stop_sl = ft.Slider(min=0, max=1, label="Stop Threshold: {value}", value=vad_settings.get('stop_threshold', 0.8))

    whisper_model_dd = ft.Dropdown(label="Whisper Model", value=config['models']['whisper'], options=[ft.dropdown.Option(m) for m in ["tiny", "base", "small", "medium", "large-v3"]])
    nllb_model_dd = ft.Dropdown(label="NLLB Model", value=config['models']['nllb'], options=[ft.dropdown.Option(m) for m in ["facebook/nllb-200-distilled-600M", "facebook/nllb-200-1.3B"]])
    tts_model_tf = ft.TextField(label="TTS Model", value=config['models']['tts'], read_only=True)

    devices = sd.query_devices()
    input_devices = [ft.dropdown.Option(d['name']) for d in devices if d['max_input_channels'] > 0]
    output_devices = [ft.dropdown.Option(d['name']) for d in devices if d['max_output_channels'] > 0]
    mic_dd = ft.Dropdown(label="Your Microphone", value=config['devices']['user_mic_name'], options=input_devices)
    headphones_dd = ft.Dropdown(label="Your Headphones", value=config['devices']['headphones_name'], options=output_devices)
    cable_in_dd = ft.Dropdown(label="Virtual Cable Input (Playback)", value=config['devices']['cable_input_name'], options=output_devices)
    cable_out_dd = ft.Dropdown(label="Virtual Cable Output (Recording)", value=config['devices']['cable_output_name'], options=input_devices)

    clone_user_voice_sw = ft.Switch(label="Clone Your Voice", value=config['user_pipeline']['clone_voice'])
    user_lang_nllb_dd = ft.Dropdown(label="Language They Hear (Translator)", value=config['user_pipeline']['target_language_nllb'], options=[ft.dropdown.Option(lang) for lang in engine.nllb_langs])
    user_lang_tts_dd = ft.Dropdown(label="Language They Hear (Voice)", value=config['user_pipeline']['target_language_tts'], options=[ft.dropdown.Option(lang) for lang in engine.tts_langs])

    clone_other_voice_sw = ft.Switch(label="Clone Their Voice", value=config['other_pipeline']['clone_voice'])
    other_lang_nllb_dd = ft.Dropdown(label="Language You Hear (Translator)", value=config['other_pipeline']['source_language_nllb'], options=[ft.dropdown.Option(lang) for lang in engine.nllb_langs])
    other_lang_tts_dd = ft.Dropdown(label="Language You Hear (Voice)", value=config['other_pipeline']['source_language_tts'], options=[ft.dropdown.Option(lang) for lang in engine.tts_langs])

    save_button = ft.ElevatedButton("Save Settings", on_click=save_settings, icon="save")
    save_status = ft.Text()

    settings_view = ft.Container(
        content=ft.Column([
            ft.Text("Models", weight=ft.FontWeight.BOLD), whisper_model_dd, nllb_model_dd, tts_model_tf,
            ft.Divider(), ft.Text("Audio Devices", weight=ft.FontWeight.BOLD), mic_dd, headphones_dd, cable_in_dd, cable_out_dd,
            ft.Divider(), ft.Text("VAD Settings", weight=ft.FontWeight.BOLD), vad_aggressiveness_sl, vad_start_sl, vad_stop_sl,
            ft.Divider(), ft.Text("Your Pipeline (What Others Hear)", weight=ft.FontWeight.BOLD), clone_user_voice_sw, user_lang_nllb_dd, user_lang_tts_dd,
            ft.Divider(), ft.Text("Their Pipeline (What You Hear)", weight=ft.FontWeight.BOLD), clone_other_voice_sw, other_lang_nllb_dd, other_lang_tts_dd,
            ft.Divider(), save_button, save_status,
            ft.Text("Note: A restart is required for settings to take effect.", size=11, italic=True)
        ], scroll=ft.ScrollMode.AUTO), padding=20
    )

    tabs = ft.Tabs(
        selected_index=0, animation_duration=300,
        tabs=[ft.Tab(text="Translator", content=translator_view), ft.Tab(text="Settings", content=settings_view)],
        expand=1,
    )

    page.add(tabs)
    page.update()

if __name__ == "__main__":
    ft.app(target=main)
