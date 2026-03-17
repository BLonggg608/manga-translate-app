import sys
import os
import subprocess
import gc
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QComboBox, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QApplication)
from PySide6.QtCore import Qt

from app.core.ai_worker import ModelLoaderWorker, AITranslatorWorker

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Translator Pro")
        self.setMinimumSize(600, 400) 
        
        self.worker = None
        self.loader_worker = None
        self.loaded_models = None 
        self.current_loaded_gpu = None 
        self.is_running = False
        
        # Security flag: Once PyTorch is loaded in this process, we CANNOT switch GPUs cleanly.
        self.pytorch_tainted = False 

        # Check if the app was restarted to automatically select the requested GPU
        self.startup_gpu = None
        if "--gpu" in sys.argv:
            idx = sys.argv.index("--gpu")
            if idx + 1 < len(sys.argv):
                self.startup_gpu = sys.argv[idx + 1]

        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25) 
        layout.setSpacing(15) 
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # --- Row 1: Input Directory ---
        input_layout = QHBoxLayout()
        self.input_entry = QLineEdit()
        self.input_entry.setPlaceholderText("Select raw manga folder (Japanese)...")
        self.input_btn = QPushButton("Browse...")
        self.input_btn.setCursor(Qt.PointingHandCursor)
        self.input_btn.clicked.connect(self.browse_input)
        input_layout.addWidget(self.input_entry)
        input_layout.addWidget(self.input_btn)
        layout.addLayout(input_layout)

        # --- Row 2: Output Directory ---
        output_layout = QHBoxLayout()
        self.output_entry = QLineEdit()
        self.output_entry.setPlaceholderText("Select output folder...")
        
        default_out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "output")
        self.output_entry.setText(default_out)
        
        self.output_btn = QPushButton("Browse...")
        self.output_btn.setCursor(Qt.PointingHandCursor)
        self.output_btn.clicked.connect(self.browse_output)
        output_layout.addWidget(self.output_entry)
        output_layout.addWidget(self.output_btn)
        layout.addLayout(output_layout)

        # --- Row 3: Translation Options (Genre & Language) ---
        options_layout = QHBoxLayout()
        
        options_layout.addWidget(QLabel("Genre:"))
        self.genre_entry = QLineEdit()
        self.genre_entry.setText("General Manga")
        self.genre_entry.setPlaceholderText("e.g., Fantasy, Romance...")
        options_layout.addWidget(self.genre_entry)

        options_layout.addSpacing(20) 

        options_layout.addWidget(QLabel("Language:"))
        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Vietnamese"])
        options_layout.addWidget(self.lang_combo)

        layout.addLayout(options_layout)

        # --- Row 4: GPU Selection Dropdown ---
        gpu_layout = QHBoxLayout()
        self.gpu_combo = QComboBox()
        gpu_layout.addWidget(QLabel("AI Device (GPU):"))
        gpu_layout.addWidget(self.gpu_combo)
        layout.addLayout(gpu_layout)

        # --- Row 5: Status Label & Progress Bar ---
        layout.addSpacing(10)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        # --- Row 6: Dynamic Action Button ---
        layout.addSpacing(10)
        self.action_btn = QPushButton("START TRANSLATION")
        self.action_btn.setObjectName("actionBtn")
        self.action_btn.setMinimumHeight(50)
        self.action_btn.setCursor(Qt.PointingHandCursor)
        self.action_btn.clicked.connect(self.handle_action_button)
        layout.addWidget(self.action_btn)

        # Scan for GPUs
        self.load_gpus()
        
        # Apply custom styles
        self.apply_styles()

    def apply_styles(self):
        qss = """
        QMainWindow {
            background-color: #1E1E2E;
        }

        QLabel {
            color: #CDD6F4;
            font-size: 13px;
            font-weight: bold;
        }

        QLineEdit {
            background-color: #313244;
            color: #CDD6F4;
            border: 2px solid #45475A;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 13px;
        }
        QLineEdit:focus {
            border: 2px solid #89B4FA;
        }

        QComboBox {
            background-color: #45475A;
            color: #CDD6F4;
            border: 2px solid #313244;
            border-radius: 8px;
            padding: 8px 12px;
            font-size: 13px;
            font-weight: bold;
        }
        QComboBox:hover {
            background-color: #585B70;
        }
        QComboBox::drop-down {
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 30px;
            border-left: 2px solid #313244; 
        }
        
        QComboBox::down-arrow {
            width: 6px;
            height: 6px;
            background-color: #CDD6F4;
            border-radius: 3px; 
            margin-right: 3px;
        }
        
        QComboBox::down-arrow:hover {
            background-color: #FFFFFF;
        }
        
        QComboBox QAbstractItemView {
            background-color: #313244;
            color: #CDD6F4;
            selection-background-color: #585B70;
            border-radius: 4px;
        }

        QPushButton {
            background-color: #45475A;
            color: #CDD6F4;
            border: none;
            border-radius: 8px;
            padding: 8px 15px;
            font-size: 13px;
            font-weight: bold;
        }
        QPushButton:hover {
            background-color: #585B70;
        }
        QPushButton:pressed {
            background-color: #313244;
        }
        QPushButton:disabled {
            background-color: #313244;
            color: #585B70;
        }

        QProgressBar {
            background-color: #313244;
            border: none;
            border-radius: 10px;
            color: white;
            font-weight: bold;
            text-align: center;
            height: 20px;
        }
        QProgressBar::chunk {
            background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #89B4FA, stop:1 #B4BEFE);
            border-radius: 10px;
        }

        QLabel#statusLabel {
            color: #A6E3A1;
            font-size: 14px;
        }

        QPushButton#actionBtn {
            background-color: #89B4FA;
            color: #1E1E2E;
            border-radius: 10px;
            font-size: 14px;
            font-weight: 900;
        }
        QPushButton#actionBtn:hover {
            background-color: #B4BEFE;
        }
        QPushButton#actionBtn:pressed {
            background-color: #74C7EC;
        }
        
        QPushButton#actionBtn[state="cancel"] {
            background-color: #F38BA8;
            color: #1E1E2E;
        }
        QPushButton#actionBtn[state="cancel"]:hover {
            background-color: #eba0b5;
        }
        QPushButton#actionBtn[state="cancel"]:pressed {
            background-color: #e64553;
        }
        QPushButton#actionBtn:disabled {
            background-color: #45475A;
            color: #7F849C;
        }
        """
        self.setStyleSheet(qss)

    def load_gpus(self):
        has_gpu = False
        try:
            result = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=index,name,memory.free', '--format=csv,nounits,noheader'], 
                encoding='utf-8'
            )
            for line in result.strip().split('\n'):
                if line:
                    gpu_id, name, free_mem = [x.strip() for x in line.split(',')]
                    self.gpu_combo.addItem(f"GPU {gpu_id}: {name} ({free_mem}MB Free)", gpu_id)
                    has_gpu = True
        except Exception:
            pass
        
        if not has_gpu:
            self.gpu_combo.addItem("No NVIDIA GPU Found", "-1")
            self.action_btn.setEnabled(False)
            self.action_btn.setText("NVIDIA GPU REQUIRED")
            return

        # Restore the previously selected GPU if the app was restarted
        if self.startup_gpu is not None:
            index = self.gpu_combo.findData(self.startup_gpu)
            if index >= 0:
                self.gpu_combo.setCurrentIndex(index)

    def browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Input Folder")
        if folder: self.input_entry.setText(folder)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder: self.output_entry.setText(folder)

    def handle_action_button(self, checked=False):
        if not self.is_running:
            self.prepare_and_start_translation()
        else:
            self.cancel_translation()

    def restart_app(self, target_gpu=None):
        """Forcefully replaces the current process at the OS level to guarantee VRAM release."""
        self.status_label.setText("Status: Restarting application to clear VRAM...")
        QApplication.processEvents() 
        
        # Clean up sys.argv so we don't have duplicate --gpu flags
        new_args = []
        skip_next = False
        for arg in sys.argv:
            if skip_next:
                skip_next = False
                continue
            if arg == "--gpu":
                skip_next = True
                continue
            new_args.append(arg)
            
        if target_gpu is not None:
            new_args.extend(["--gpu", str(target_gpu)])
            
        os.execl(sys.executable, sys.executable, *new_args)

    def prepare_and_start_translation(self):
        input_dir = self.input_entry.text().strip()
        output_dir = self.output_entry.text().strip()
        
        if not input_dir or not output_dir:
            QMessageBox.warning(self, "Error", "Please select both Input and Output folders!")
            return

        selected_gpu = self.gpu_combo.currentData()

        # The Golden Rule: If PyTorch was touched, we MUST restart to change GPU or recover from OOM
        if self.pytorch_tainted and (self.loaded_models is None or self.current_loaded_gpu != selected_gpu):
            reply = QMessageBox.question(
                self, 
                "Restart Required", 
                "To switch GPUs or safely recover from a previous task, the application must restart to clear VRAM completely.\n\nDo you want to restart now?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.restart_app(target_gpu=selected_gpu)
            else:
                if self.current_loaded_gpu:
                    idx = self.gpu_combo.findData(self.current_loaded_gpu)
                    if idx >= 0: self.gpu_combo.setCurrentIndex(idx)
            return

        self.is_running = True
        self.genre_entry.setEnabled(False)
        self.lang_combo.setEnabled(False)
        self.gpu_combo.setEnabled(False)
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #89B4FA, stop:1 #B4BEFE); border-radius: 10px; }")

        if self.loaded_models is None:
            self.load_models_to_gpu(selected_gpu)
        else:
            self.run_translation_worker()

    def load_models_to_gpu(self, gpu_id):
        self.pytorch_tainted = True # Mark process as having initialized PyTorch
        
        self.action_btn.setEnabled(False)
        self.action_btn.setText("LOADING MODELS... (DO NOT CLOSE)")
        
        self.loader_worker = ModelLoaderWorker(gpu_id)
        self.loader_worker.progress_updated.connect(self.update_progress)
        self.loader_worker.finished.connect(self.on_models_loaded)
        self.loader_worker.error_occurred.connect(self.on_error)
        self.loader_worker.start()

    def on_models_loaded(self, models):
        self.loaded_models = models
        self.current_loaded_gpu = self.gpu_combo.currentData()
        self.run_translation_worker()

    def run_translation_worker(self):
        self.action_btn.setEnabled(True)
        self.action_btn.setText("CANCEL")
        self.action_btn.setProperty("state", "cancel")
        self.action_btn.style().unpolish(self.action_btn)
        self.action_btn.style().polish(self.action_btn)

        config = {
            'input_dir': self.input_entry.text().strip(),
            'output_dir': self.output_entry.text().strip(),
            'gpu_id': self.current_loaded_gpu,
            'genre': self.genre_entry.text().strip() or "General Manga",
            'target_lang': self.lang_combo.currentText().strip() or "Vietnamese"
        }

        self.worker = AITranslatorWorker(config, self.loaded_models)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.finished.connect(self.on_finished)
        self.worker.error_occurred.connect(self.on_error)
        self.worker.cancelled.connect(self.on_cancelled)
        
        self.worker.start()

    def cancel_translation(self):
        reply = QMessageBox.question(
            self, 
            "Confirm Cancel", 
            "Are you sure you want to stop the translation process?",
            QMessageBox.Yes | QMessageBox.No, 
            QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            self.status_label.setText("Status: Cancelling... Waiting for current task to finish.")
            self.action_btn.setEnabled(False)
            self.action_btn.setText("CANCELLING...")
            
            if self.worker is not None and self.worker.isRunning():
                self.worker.stop_gracefully()

    def update_progress(self, percent, message):
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)

    def on_finished(self, message):
        self.progress_bar.setValue(100)
        self.status_label.setText("All Done!")
        self.unlock_ui()
        QMessageBox.information(self, "Success", message)

    def on_error(self, message):
        self.status_label.setText("ERROR!")
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #F38BA8, stop:1 #eba0b5); border-radius: 10px; }")
        self.unlock_ui()
        
        self.loaded_models = None 
        
        QMessageBox.critical(self, "System Error", message)

    def on_cancelled(self, message):
        self.status_label.setText("Status: Cancelled by user")
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: #F38BA8; border-radius: 10px; }")
        self.unlock_ui()
        QMessageBox.warning(self, "Cancelled", message)

    def unlock_ui(self):
        self.is_running = False
        
        self.action_btn.setEnabled(True)
        self.action_btn.setText("START TRANSLATION")
        self.action_btn.setProperty("state", "")
        self.action_btn.style().unpolish(self.action_btn)
        self.action_btn.style().polish(self.action_btn)
        
        self.genre_entry.setEnabled(True)
        self.lang_combo.setEnabled(True)
        self.gpu_combo.setEnabled(True)