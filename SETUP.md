# Setup & Installation Guide

## 1. Prerequisites
- **Python 3.12+**
- **Android Device**: Xiaomi Pad 6 or Android device connected via USB / Wi-Fi ADB.
- **ADB (Android Debug Bridge)**: Ensure `adb devices` shows your device in `device` state.
- **Scrcpy**: Installed via winget (`winget install Genymobile.scrcpy`) or downloaded to PATH.

## 2. Android Device Setup
1. Enable **Developer Options** and **USB Debugging** on the tablet.
2. Enable **USB Debugging (Security Settings)** if prompted to allow touch emulation.
3. Connect tablet via USB cable. Run `adb devices` to verify connection.

## 3. Running the Bot
```powershell
# 1. Launch Desktop App (GUI with live stream and controls)
python main.py

# 2. Run Headless Autonomous Daemon
python main.py --headless --turn-screen-off

# 3. Run in Copilot Advisory Mode
python main.py --mode COPILOT
```
