import sys
import os
import io
import re
import time
import json
import asyncio
import subprocess
import webbrowser
import traceback
import ctypes
from ctypes import wintypes

import mss
from PIL import Image
from dotenv import load_dotenv

import pyautogui
pyautogui.FAILSAFE = True

from google import genai
from google.genai import types

import edge_tts

from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QUrl, QEvent, QObject, QAbstractNativeEventFilter
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextEdit, QFrame, QSizeGrip,
    QCheckBox, QLabel, QComboBox, QStackedWidget, QListWidget
)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

# Load environment variables before initializing client
load_dotenv()

# Initialize Google GenAI Client
try:
    client = genai.Client()
except Exception as e:
    print(f"[Initialization Warning] Client init failed: {e}")

# Explicit Active Model Definitions
PRIMARY_MODEL = 'gemini-3.8-flash'
FALLBACK_MODELS = ['gemini-3.1-pro-preview', 'gemini-3.0-flash']

def get_available_models():
    """Queries Google API for currently active generateContent models on your key."""
    try:
        all_models = list(client.models.list())
        supported = [
            m.name.replace('models/', '') for m in all_models
            if hasattr(m, 'supported_generation_methods') and 'generateContent' in m.supported_generation_methods
        ]
        
        flash_models = [m for m in supported if 'flash' in m]
        pro_models = [m for m in supported if 'pro' in m]
        
        if flash_models:
            print(f"[API Connected] Available Flash models: {flash_models}")
            return flash_models[0], flash_models[1:] + pro_models
        elif supported:
            print(f"[API Connected] Available models: {supported}")
            return supported[0], supported[1:]
    except Exception as err:
        print(f"[Model Discovery Warning] Could not list models automatically: {err}")
    
    return PRIMARY_MODEL, FALLBACK_MODELS

# Dynamically set active models from your Google API key on startup
PRIMARY_MODEL, FALLBACK_MODELS = get_available_models()
print(f"[Active Config] Primary: {PRIMARY_MODEL} | Fallbacks: {FALLBACK_MODELS}")

FEMALE_VOICE = 'en-US-AriaNeural'

WM_HOTKEY = 0x0312
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
VK_P = 0x50
HOTKEY_ID = 1001


def clean_text_for_speech(text: str) -> str:
    text = re.sub(r'```[\s\S]*?```', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'[\*_]{1,2}', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'^\s*[\-\*\+]\s+', '', text, flags=re.MULTILINE)
    return re.sub(r'\s+', ' ', text).strip()


def extract_json_payload(text: str):
    if not text:
        return None
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\})\s*```', text, re.IGNORECASE)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'(\{[\s\S]*"action"[\s\S]*\})', text)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


class ActionExecutor:
    @staticmethod
    def execute(action_dict, screen_width=1920, screen_height=1080):
        action_type = action_dict.get("action")
        results = []

        try:
            if action_type == "open_url":
                url = action_dict.get("url", "https://google.com")
                webbrowser.open(url, new=2)
                results.append(f"Opened URL: {url}")

            elif action_type == "run_command":
                cmd = action_dict.get("command", "")
                if cmd:
                    res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10)
                    out = res.stdout or res.stderr or "Executed successfully."
                    results.append(f"Command Output:\n{out[:300]}")

            elif action_type == "click_coordinate":
                px = action_dict.get("x_percent", 50)
                py = action_dict.get("y_percent", 50)
                target_x = int((px / 100.0) * screen_width)
                target_y = int((py / 100.0) * screen_height)
                pyautogui.click(target_x, target_y)
                results.append(f"Clicked screen coordinates ({target_x}, {target_y})")

            elif action_type == "hotkey":
                keys = action_dict.get("keys", [])
                if keys:
                    pyautogui.hotkey(*keys)
                    results.append(f"Triggered hotkey: {keys}")

            elif action_type == "type_text":
                text = action_dict.get("text", "")
                if text:
                    pyautogui.write(text, interval=0.01)
                    results.append(f"Typed text: {text}")

        except Exception as e:
            results.append(f"Action execution error: {str(e)}")

        return "\n".join(results)


class AutonomousAIWorker(QThread):
    chunk_received = pyqtSignal(str)
    finished = pyqtSignal(str, str, dict)

    def __init__(self, query, enable_tts=True, mode="Concise"):
        super().__init__()
        self.query = query
        self.enable_tts = enable_tts
        self.mode = mode

    def capture_screen_internal(self):
        """Thread-safe screen capture executed inside worker lifecycle using mss.MSS()."""
        with mss.MSS() as sct:
            monitor = sct.monitors[1]
            sct_img = sct.grab(monitor)
            img = Image.frombytes("RGB", sct_img.size, sct_img.bgra, "raw", "BGRX")
            img.thumbnail((1280, 720), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=75)
            return buffer.getvalue(), monitor["width"], monitor["height"]

    def run(self):
        full_text = ""
        models = [PRIMARY_MODEL] + FALLBACK_MODELS

        # 1. Desktop Screenshot Capture
        try:
            screen_bytes, sw, sh = self.capture_screen_internal()
        except Exception as e:
            print(f"[Worker Error] Screenshot failure: {e}")
            traceback.print_exc()
            self.finished.emit(f"Capture error: {e}", "", {})
            return

        depth_instruction = "Keep explanation to 1 short sentence." if self.mode == "Concise" else "Provide full breakdown."

        system_instruction = (
            "You are an active PC Automation Agent. "
            "Output the automation JSON block FIRST at the very start, then brief text.\n\n"
            "Format strictly:\n"
            "```json\n"
            "{\n"
            '  "action": "open_url" | "run_command" | "click_coordinate" | "hotkey" | "type_text",\n'
            '  "url": "[https://google.com](https://google.com)",\n'
            '  "command": "dir",\n'
            '  "x_percent": 50,\n'
            '  "y_percent": 50,\n'
            '  "keys": ["ctrl", "t"],\n'
            '  "text": "sample text"\n'
            "}\n"
            "```\n"
            f"{depth_instruction}"
        )

        try:
            contents = [
                types.Part.from_bytes(data=screen_bytes, mime_type='image/jpeg'),
                f"{system_instruction}\n\nTask: {self.query}"
            ]
        except Exception as e:
            print(f"[Worker Error] Input preparation failure: {e}")
            self.finished.emit(f"Payload error: {e}", "", {})
            return

        # 2. Execution Stream across available active models
        for model_name in models:
            try:
                response = client.models.generate_content_stream(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        max_output_tokens=600,
                        temperature=0.0,
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
                    )
                )
                for chunk in response:
                    if chunk and hasattr(chunk, 'text') and chunk.text:
                        full_text += chunk.text
                        self.chunk_received.emit(full_text)
                
                if full_text:
                    break
            except Exception as err:
                print(f"[Model Failure] Execution failed on {model_name}: {err}")
                continue

        if not full_text:
            self.finished.emit("Failed to process request. Check terminal logs for detailed traceback.", "", {})
            return

        action_dict = extract_json_payload(full_text) or {}
        display_text = re.sub(r'```(?:json)?\s*\{[\s\S]*?\}\s*```', '', full_text, flags=re.IGNORECASE).strip()

        # 3. Thread-Safe Text-To-Speech Synthesis
        audio_path = ""
        if self.enable_tts and display_text:
            spoken_text = clean_text_for_speech(display_text)
            audio_path = os.path.abspath("temp_response.mp3")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                communicate = edge_tts.Communicate(spoken_text, FEMALE_VOICE)
                loop.run_until_complete(communicate.save(audio_path))
                loop.close()
            except Exception as tts_err:
                print(f"[TTS Failure - Continuing without audio]: {tts_err}")
                audio_path = ""

        self.finished.emit(display_text, audio_path, action_dict)


class WinHotkeyFilter(QAbstractNativeEventFilter):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam == HOTKEY_ID:
                self.callback()
                return True, 0
        return False, 0


class FluentGlassHUD(QWidget):
    BORDER_WIDTH = 10

    def __init__(self):
        super().__init__()
        self.drag_pos = None
        self.resize_dir = None
        self.is_processing = False
        self.hotkey_registered = False
        self.action_executed = False

        self.screen_width = 1920
        self.screen_height = 1080

        self.audio_output = QAudioOutput()
        self.audio_output.setVolume(1.0)
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self.audio_output)

        self.init_ui()

    def showEvent(self, event):
        super().showEvent(event)
        if not self.hotkey_registered:
            self._register_native_hotkey()

    def stop_audio(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.stop()

    def _register_native_hotkey(self):
        hwnd = int(self.winId())
        if hwnd:
            user32 = ctypes.windll.user32
            if user32.RegisterHotKey(hwnd, HOTKEY_ID, MOD_CONTROL | MOD_SHIFT, VK_P):
                self.win_filter = WinHotkeyFilter(self.on_hotkey_pressed)
                QApplication.instance().installNativeEventFilter(self.win_filter)
                self.hotkey_registered = True

    def on_hotkey_pressed(self):
        if self.is_processing:
            return

        if self.isHidden() or self.isMinimized():
            self.showNormal()
            self.raise_()
            self.activateWindow()
        else:
            self.on_capture()

    def init_ui(self):
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        
        self.setMinimumSize(360, 300)
        self.resize(540, 440)

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(6, 6, 6, 6)

        self.container = QFrame(self)
        self.container.setObjectName("GlassMain")
        self.container.setStyleSheet("""
            QFrame#GlassMain {
                background-color: rgba(18, 20, 29, 0.95);
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 16px;
            }
            QLabel, QCheckBox {
                color: #ffffff;
                font-family: 'Segoe UI', sans-serif;
            }
        """)

        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(16, 12, 16, 12)
        container_layout.setSpacing(10)

        top_bar = QHBoxLayout()

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setFixedSize(26, 26)
        self.settings_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.settings_btn.setStyleSheet("""
            QPushButton {
                background: rgba(255, 255, 255, 0.08);
                color: rgba(255, 255, 255, 0.85);
                border: 1px solid rgba(255, 255, 255, 0.15);
                border-radius: 13px;
            }
            QPushButton:hover { background: rgba(255, 255, 255, 0.25); }
        """)
        self.settings_btn.clicked.connect(self.toggle_settings)
        top_bar.addWidget(self.settings_btn)

        self.history_btn = QPushButton("📋 Log")
        self.history_btn.setFixedSize(50, 26)
        self.history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.history_btn.setStyleSheet(self.settings_btn.styleSheet())
        self.history_btn.clicked.connect(self.toggle_history)
        top_bar.addWidget(self.history_btn)

        self.handle = QWidget()
        self.handle.setFixedSize(38, 4)
        self.handle.setStyleSheet("background-color: rgba(255, 255, 255, 0.35); border-radius: 2px;")
        top_bar.addStretch()
        top_bar.addWidget(self.handle)
        top_bar.addStretch()

        min_btn = QPushButton("—")
        min_btn.setFixedSize(26, 26)
        min_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        min_btn.setStyleSheet(self.settings_btn.styleSheet())
        min_btn.clicked.connect(self.showMinimized)
        top_bar.addWidget(min_btn)

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(26, 26)
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet(self.settings_btn.styleSheet())
        close_btn.clicked.connect(self.close)
        top_bar.addWidget(close_btn)

        container_layout.addLayout(top_bar)

        self.stacked = QStackedWidget()

        main_view = QWidget()
        main_layout = QVBoxLayout(main_view)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(10)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter command (e.g., 'open google.com', 'click login')...")
        self.input_field.setStyleSheet("""
            QLineEdit {
                background-color: rgba(255, 255, 255, 0.09);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.18);
                border-radius: 10px;
                padding: 8px 12px;
            }
        """)
        self.input_field.returnPressed.connect(self.on_capture)
        main_layout.addWidget(self.input_field)

        self.btn_capture = QPushButton("Analyze & Auto-Execute (Ctrl+Shift+P)")
        self.btn_capture.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_capture.setStyleSheet("""
            QPushButton {
                background-color: rgba(255, 255, 255, 0.16);
                color: #ffffff;
                font-weight: 600;
                border-radius: 8px;
                padding: 8px 16px;
                border: 1px solid rgba(255, 255, 255, 0.22);
            }
            QPushButton:hover { background-color: rgba(255, 255, 255, 0.28); }
        """)
        self.btn_capture.clicked.connect(self.on_capture)
        main_layout.addWidget(self.btn_capture)

        self.output_area = QTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setPlaceholderText("Ready. Press Ctrl+Shift+P to trigger anywhere.")
        self.output_area.setStyleSheet("""
            QTextEdit {
                background-color: rgba(0, 0, 0, 0.45);
                color: #ffffff;
                border: 1px solid rgba(255, 255, 255, 0.12);
                border-radius: 10px;
                padding: 10px;
            }
        """)
        main_layout.addWidget(self.output_area, 1)
        self.stacked.addWidget(main_view)

        settings_view = QWidget()
        settings_layout = QVBoxLayout(settings_view)

        lbl_title = QLabel("Settings")
        lbl_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        settings_layout.addWidget(lbl_title)

        self.chk_audio = QCheckBox("Enable Text-to-Speech Audio")
        self.chk_audio.setChecked(True)
        settings_layout.addWidget(self.chk_audio)

        lbl_mode = QLabel("Response Mode:")
        settings_layout.addWidget(lbl_mode)
        
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Concise", "Detailed"])
        self.combo_mode.setStyleSheet("background: rgba(255,255,255,0.1); color: white; padding: 4px;")
        settings_layout.addWidget(self.combo_mode)

        settings_layout.addStretch()
        btn_back = QPushButton("← Back to HUD")
        btn_back.setStyleSheet(self.btn_capture.styleSheet())
        btn_back.clicked.connect(self.toggle_settings)
        settings_layout.addWidget(btn_back)

        self.stacked.addWidget(settings_view)

        history_view = QWidget()
        history_layout = QVBoxLayout(history_view)

        lbl_hist_title = QLabel("Execution Log")
        lbl_hist_title.setStyleSheet("font-size: 14px; font-weight: bold;")
        history_layout.addWidget(lbl_hist_title)

        self.history_list = QListWidget()
        self.history_list.setStyleSheet("background-color: rgba(0, 0, 0, 0.45); color: #ffffff;")
        history_layout.addWidget(self.history_list, 1)

        btn_hist_back = QPushButton("← Back to HUD")
        btn_hist_back.setStyleSheet(self.btn_capture.styleSheet())
        btn_hist_back.clicked.connect(self.toggle_history)
        history_layout.addWidget(btn_hist_back)

        self.stacked.addWidget(history_view)

        container_layout.addWidget(self.stacked, 1)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch()
        grip = QSizeGrip(self)
        bottom_layout.addWidget(grip)
        container_layout.addLayout(bottom_layout)

        root_layout.addWidget(self.container)

    def toggle_settings(self):
        self.stacked.setCurrentIndex(1 if self.stacked.currentIndex() != 1 else 0)

    def toggle_history(self):
        self.stacked.setCurrentIndex(2 if self.stacked.currentIndex() != 2 else 0)

    def on_capture(self):
        if self.is_processing:
            return

        self.stop_audio()
        self.is_processing = True
        self.action_executed = False
        
        query = self.input_field.text().strip() or "Open browser."
        self.input_field.clear()
        self.btn_capture.setEnabled(False)
        self.btn_capture.setText("Capturing & Processing...")
        
        self.hide()
        QTimer.singleShot(150, lambda: self._start_worker(query))

    def _start_worker(self, query):
        self.show()
        self.raise_()
        self.activateWindow()
        self.output_area.setText("Analyzing layout...")

        mode = "Concise" if self.combo_mode.currentIndex() == 0 else "Detailed"
        self.worker = AutonomousAIWorker(
            query, enable_tts=self.chk_audio.isChecked(), mode=mode
        )
        self.worker.chunk_received.connect(self.on_chunk)
        self.worker.finished.connect(self.on_finished)
        self.worker.start()

    def on_chunk(self, live_text):
        clean_live = re.sub(r'```(?:json)?\s*\{[\s\S]*?\}\s*```', '', live_text, flags=re.IGNORECASE).strip()
        self.output_area.setText(clean_live if clean_live else "Executing automation plan...")

        if not self.action_executed:
            action_dict = extract_json_payload(live_text)
            if action_dict:
                self.action_executed = True
                QTimer.singleShot(10, lambda: self.run_system_action(action_dict))

    def on_finished(self, text, audio_path, action_dict):
        self.output_area.setText(text if text else "Action completed.")
        self.btn_capture.setEnabled(True)
        self.btn_capture.setText("Analyze & Auto-Execute (Ctrl+Shift+P)")
        self.is_processing = False

        if action_dict and not self.action_executed:
            self.action_executed = True
            self.run_system_action(action_dict)

        if audio_path and os.path.exists(audio_path) and self.chk_audio.isChecked():
            file_url = QUrl.fromLocalFile(audio_path)
            self.player.stop()
            self.player.setSource(file_url)
            self.player.play()

    def run_system_action(self, action_dict):
        action_type = action_dict.get("action")
        
        if action_type in ("hotkey", "click_coordinate", "type_text"):
            self.hide()
            QApplication.processEvents()
            time.sleep(0.2)

        execution_log = ActionExecutor.execute(
            action_dict, screen_width=self.screen_width, screen_height=self.screen_height
        )

        if action_type in ("hotkey", "click_coordinate", "type_text"):
            self.show()

        if execution_log:
            self.history_list.addItem(f"[{time.strftime('%H:%M:%S')}] {execution_log}")

    def closeEvent(self, event):
        self.stop_audio()
        if self.hotkey_registered:
            hwnd = int(self.winId())
            ctypes.windll.user32.UnregisterHotKey(hwnd, HOTKEY_ID)
        super().closeEvent(event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    hud = FluentGlassHUD()
    hud.show()
    sys.exit(app.exec())