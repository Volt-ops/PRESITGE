# Prestige AI HUD

A translucent, frameless desktop AI automation HUD built with Python, PyQt6, and the Google Gemini API.

Prestige provides an overlay interface for real-time visual screen capture, voice synthesis, and dynamic AI model interaction directly from your desktop.

---

## Project Screenshot

![Project Screenshot](./images/Screenshot-Prstige.png)

## Features

- **Frosted Glass HUD Layout:** Lightweight, transparent PyQt6 desktop overlay designed for unobtrusive background operation.
- **Multimodal Perception:** Instant screen capture and visual analysis powered by `mss` and `pyautogui`.
- **Voice Output Engine:** Asynchronous text-to-speech voice output using `edge-tts`.
- **Dynamic AI Core:** Integrated with the official `google-genai` SDK, supporting dynamic model discovery across Gemini model tiers.

---

## System Requirements

- **Operating System:** Windows 10 / 11
- **Python Version:** Python 3.10 or higher
- **API Key:** A valid Google Gemini API key (get one from [Google AI Studio](https://aistudio.google.com/app/apikey))

---

## Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Volt-ops/PRESITGE.git
cd PRESITGE
```

### 2. Create and Activate a Virtual Environment

PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

> For Command Prompt (cmd.exe), use `venv\Scripts\activate.bat` instead.

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Environment Variables

1. Copy the example file to create your local `.env`:

```powershell
   Copy-Item .env.example .env
```

2. Open `.env` in a text editor:

```powershell
   notepad .env
```

3. Replace the placeholder with your actual Gemini API key:

```
   GEMINI_API_KEY=your_actual_gemini_api_key_here
```

4. Save and close the file.

> **Never commit your `.env` file or share your API key.** The `.env` file is listed in `.gitignore`.

---

## Usage

With your virtual environment active, run:

```powershell
python main.py
```

The Prestige HUD overlay will launch on your desktop. Use the HUD interface or the designated keybindings to trigger screen capture, text input, or audio playback.

---

## Security

- Keep your API key only in `.env`, which is excluded from version control.
- `.env.example` contains placeholder values only.
- If you ever expose a key, revoke it immediately in the Google Cloud / AI Studio console and create a new one.
