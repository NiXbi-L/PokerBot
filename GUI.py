import tkinter as tk
from tkinter import filedialog
import pyautogui
from pynput import keyboard
from pynput.keyboard import Key
import threading
import time
import ttkbootstrap as ttk
from ttkbootstrap import Style
import pygetwindow as gw

# Создаем окно и применяем тему
style = Style(theme="darkly")  # Альтернативный вариант

# Остальной код класса Application остается без изменений
class Application(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("AI Assistant")
        self.geometry("400x350")
        self.running = False
        self.hotkey_start = (Key.alt, 'a')  # Было: "alt+a"
        self.hotkey_stop = (Key.alt, 's')
        self.window_rect = None
        self.model_path = ""
        self.mode = "autoplay"  # autoplay/hints
        self.listener = None

        # Инициализация дескрипторов здесь
        self.start_hotkey_handle = None
        self.stop_hotkey_handle = None

        self.create_widgets()
        self.register_hotkeys()

    def create_widgets(self):
        # Model file selection
        ttk.Button(self, text="Select .pth Model", command=self.select_model).pack(pady=5)
        self.model_label = ttk.Label(self, text="No model selected")
        self.model_label.pack()

        # Window selection
        ttk.Button(self, text="Select Target Window", command=self.select_window).pack(pady=5)
        self.window_label = ttk.Label(self, text="No window selected")
        self.window_label.pack()

        # Mode selection
        self.mode_var = tk.StringVar(value="autoplay")
        ttk.Radiobutton(self, text="Autoplay", variable=self.mode_var,
                        value="autoplay").pack()
        ttk.Radiobutton(self, text="Hints", variable=self.mode_var,
                        value="hints").pack()

        # Hotkey configuration
        ttk.Label(self, text="Start Hotkey:").pack()
        self.start_hotkey_entry = ttk.Entry(self)
        self.start_hotkey_entry.insert(0, self.hotkey_start)
        self.start_hotkey_entry.pack()

        ttk.Label(self, text="Stop Hotkey:").pack()
        self.stop_hotkey_entry = ttk.Entry(self)
        self.stop_hotkey_entry.insert(0, self.hotkey_stop)
        self.stop_hotkey_entry.pack()

        ttk.Button(self, text="Apply Hotkeys", command=self.update_hotkeys).pack(pady=5)

        # Start/Stop buttons
        self.start_btn = ttk.Button(self, text="Start", command=self.start)
        self.start_btn.pack(side=tk.LEFT, padx=20)
        self.stop_btn = ttk.Button(self, text="Stop", command=self.stop, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.RIGHT, padx=20)

        # Status
        self.status_label = ttk.Label(self, text="Status: Stopped")
        self.status_label.pack(pady=10)

    def select_model(self):
        file_path = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if file_path:
            self.model_path = file_path
            self.model_label.config(text=file_path.split("/")[-1])

    def select_window(self):
        self.withdraw()
        time.sleep(0.5)
        # Используйте pygetwindow или другой метод
        windows = pyautogui.getWindowsWithTitle("")
        if windows:
            window = windows[0]
            self.window_rect = (window.left, window.top, window.width, window.height)
            self.window_label.config(text=f"{window.title} ({window.width}x{window.height})")
        self.deiconify()

    def update_hotkeys(self):
        # Преобразуйте строки из полей ввода в формат pynput
        start_str = self.start_hotkey_entry.get().lower()
        stop_str = self.stop_hotkey_entry.get().lower()

        # Пример преобразования "alt+a" в (<Key.alt>, 'a')
        try:
            mod1, key1 = start_str.split('+')
            self.hotkey_start = (getattr(Key, mod1), key1)
        except Exception:
            print("Invalid start hotkey format")

        try:
            mod2, key2 = stop_str.split('+')
            self.hotkey_stop = (getattr(Key, mod2), key2)
        except Exception:
            print("Invalid stop hotkey format")

        self.register_hotkeys()

    def register_hotkeys(self):
        if self.listener:
            self.listener.stop()

        # Создаем словарь с кортежами клавиш
        hotkeys = {
            '<alt>+a': self.start,  # Синтаксис pynput для хоткеев
            '<alt>+s': self.stop
        }

        self.listener = keyboard.GlobalHotKeys(hotkeys)
        self.listener.start()

    def start(self):
        if not self.running:
            self.running = True
            self.status_label.config(text="Status: Running")
            self.start_btn.config(state=tk.DISABLED)
            self.stop_btn.config(state=tk.NORMAL)

            # Start processing thread
            self.thread = threading.Thread(target=self.process)
            self.thread.daemon = True
            self.thread.start()

    def stop(self):
        self.running = False
        self.status_label.config(text="Status: Stopped")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def process(self):
        while self.running:
            if self.window_rect:
                # Capture window region
                x, y, w, h = self.window_rect
                screenshot = pyautogui.screenshot(region=(x, y, w, h))

                # Here you would add your model processing code
                # Example:
                # prediction = model.predict(screenshot)

                # Implement your mode logic
                if self.mode_var.get() == "autoplay":
                    pass  # Add autoplay logic
                else:
                    pass  # Add hints logic

            time.sleep(0.1)  # Adjust processing interval

    def on_close(self):
        if self.listener:
            self.listener.stop()
        self.destroy()


if __name__ == "__main__":
    app = Application()  # Создается только одно окно
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()