# Mentalese Real-Time Translator

This is currently a proof of concept, not a final product

Mentalese is a real-time, two-way voice translation application that runs on your local machine. It's designed to sit between you and a voice chat application (like Discord, Skype, or Google Meet) and translate your conversation in real-time.

## Features

- **Real-time, two-way translation:** Translates your voice for the other person and their voice for you.
- **Voice cloning:** Can clone your voice and the other person's voice for more natural-sounding translations.
- **Multiple languages supported:** Leverages Facebook's NLLB model for translation between many languages and Coqui TTS for voice synthesis in several languages.
- **GUI for configuration and monitoring:** A simple interface to control the engine, view transcriptions and translations, and configure settings.
- **Local and private:** All processing is done on your machine, ensuring privacy.

## How It Works

Mentalese uses a clever audio routing system with a virtual audio cable (VB-CABLE) to intercept and process audio streams.

1.  **Your Voice:**
    - Mentalese captures audio from your microphone.
    - It transcribes your speech to text using Whisper.
    - It translates the text to the target language using NLLB.
    - It synthesizes the translated text to speech using Coqui TTS (optionally cloning your voice).
    - The translated audio is sent to the **Virtual Cable Input**, which should be set as the microphone in your chat application.

2.  **Their Voice:**
    - You must set your chat application's audio output to the **Virtual Cable Output**.
    - Mentalese captures the other person's audio from the Virtual Cable Output.
    - It transcribes, translates, and synthesizes their speech, similar to your voice's pipeline.
    - The translated audio is sent directly to your headphones.

## Requirements

- **Python 3.10+**
- **PyTorch**
- **A CUDA-enabled GPU is highly recommended for decent performance.**
- **VB-CABLE Virtual Audio Device:** You need to have VB-CABLE installed.

## Installation

1.  **Install VB-CABLE:**
    - If you don't have it, you can find the installer in the project's root directory (`VBCABLE_Setup_x64.exe`).
    - Right-click `VBCABLE_Setup_x64.exe` and **Run as administrator**.

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download AI Models:**
    - The first time you run the application, the necessary AI models (Whisper, NLLB, and TTS) will be downloaded and cached in the `models` directory. This may take a while and requires a good internet connection.

## Usage

1.  **Configure `config.json`:**
    - Open `config.json` and configure your audio devices and language settings. See the Configuration section below for more details.

2.  **Run the application:**
    ```bash
    python gui.py
    ```

3.  **Configure your chat application:**
    - **Input Device:** Set to `CABLE Input (VB-Audio Virtual Cable)`.
    - **Output Device:** Set to `CABLE Output (VB-Audio Virtual Cable)`.

4.  **Start the engine:**
    - Click the "Start Engine" button in the Mentalese GUI.

## Configuration

The `config.json` file is the main configuration file for the application.

- **`models`**: Specifies the AI models to use.
- **`devices`**: The names of your audio devices.
  - `user_mic_name`: Your actual microphone.
  - `headphones_name`: Your headphones or speakers.
  - `cable_input_name`: Should be `CABLE Input (VB-Audio Virtual Cable)`.
  - `cable_output_name`: Should be `CABLE Output (VB-Audio Virtual Cable)`.
- **`vad_settings`**: Voice Activity Detection settings.
- **`user_pipeline`**: Settings for your voice's translation.
  - `clone_voice`: Set to `true` to enable voice cloning. The first time you speak, it will register your voice.
  - `target_language_nllb`: The language to translate your voice to for the other person (using NLLB language codes).
  - `target_language_tts`: The language for the synthesized voice (using TTS language codes).
- **`other_pipeline`**: Settings for the other person's voice translation.
  - `clone_voice`: Set to `true` to enable voice cloning for the other person.
  - `source_language_nllb`: The language the other person is speaking (using NLLB language codes).
  - `source_language_tts`: The language the other person is speaking (for Whisper).
