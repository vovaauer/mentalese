import subprocess
import os

# Use the pyinstaller from the virtual environment
pyinstaller_executable = os.path.join('.venv', 'Scripts', 'pyinstaller.exe')

# Paths to the VERSION files
tts_version_file = os.path.join('.venv', 'Lib', 'site-packages', 'TTS', 'VERSION')
trainer_version_file = os.path.join('.venv', 'Lib', 'site-packages', 'trainer', 'VERSION')

# Command to create the installer
pyinstaller_command = [
    pyinstaller_executable,
    '--name', 'Mentalese',
    '--onefile',
    '--windowed',
    '--icon', 'pin_in.ico',
    '--add-data', f'VBCABLE_Setup_x64.exe{os.pathsep}.',
    '--add-data', f'VBCABLE_Setup.exe{os.pathsep}.',
    '--add-data', f'config.json{os.pathsep}.',
    '--add-data', f'pin_in.ico{os.pathsep}.',
    '--add-data', f'pin_out.ico{os.pathsep}.',
    '--add-data', f'{tts_version_file}{os.pathsep}TTS',
    '--add-data', f'{trainer_version_file}{os.pathsep}trainer',
    'gui.py'
]

print("--- Running PyInstaller Command ---")
print(" ".join(pyinstaller_command))
print("-------------------------------------")

try:
    subprocess.run(pyinstaller_command, check=True)
    print("\nInstaller created successfully in the 'dist' folder.")
except FileNotFoundError:
    print(f"\nError: Could not find {pyinstaller_executable}.")
    print("Please make sure you are running this from the root of your project directory.")
except subprocess.CalledProcessError as e:
    print(f"\nAn error occurred while running PyInstaller: {e}")