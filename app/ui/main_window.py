import sys
import os
import subprocess
import gc
import shutil
from PySide6.QtWidgets import (QGridLayout, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QPushButton, QLabel, QComboBox, QLineEdit, QFileDialog, 
                               QProgressBar, QMessageBox, QApplication, QScrollArea, QDialog)
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QIcon, QPixmap

from app.core.ai_worker import ModelLoaderWorker, AITranslatorWorker

class ImagePreviewDialog(QDialog):
    """A popup dialog to display the full resolution image with scrolling and zooming capabilities."""
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Preview - Full Resolution")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("background-color: #1E1E2E;")
        
        self.scale_factor = 1.0
        self.original_pixmap = QPixmap(image_path)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: #1E1E2E;
                width: 14px;
            }
            QScrollBar:horizontal {
                background: #1E1E2E;
                height: 14px;
            }
            QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
                background: #45475A;
                border-radius: 7px;
            }
            QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {
                background: #585B70;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
            }
        """)
        
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        
        # Calculate initial scale factor to fit screen if the image is too large
        screen_size = QApplication.primaryScreen().availableSize()
        screen_width = screen_size.width() - 40
        screen_height = screen_size.height() - 40
        
        if self.original_pixmap.width() > screen_width or self.original_pixmap.height() > screen_height:
            width_ratio = screen_width / self.original_pixmap.width()
            height_ratio = screen_height / self.original_pixmap.height()
            self.scale_factor = min(width_ratio, height_ratio)

        self.update_image()
        
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

    def update_image(self):
        """Scales the original image by the current scale factor and updates the label."""
        if not self.original_pixmap.isNull():
            new_width = int(self.original_pixmap.width() * self.scale_factor)
            new_height = int(self.original_pixmap.height() * self.scale_factor)
            
            scaled_pixmap = self.original_pixmap.scaled(
                new_width, 
                new_height, 
                Qt.KeepAspectRatio, 
                Qt.SmoothTransformation
            )
            
            self.image_label.setPixmap(scaled_pixmap)
            self.image_label.resize(scaled_pixmap.size())

    def wheelEvent(self, event):
        """Intercept mouse wheel scrolling to perform zoom if Ctrl is held."""
        if event.modifiers() == Qt.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def zoom_in(self):
        self.scale_image(1.2)

    def zoom_out(self):
        self.scale_image(0.8)

    def scale_image(self, factor):
        """Adjusts the scale factor and updates the image and scrollbars."""
        new_scale = self.scale_factor * factor
        
        if new_scale < 0.1 or new_scale > 10.0:
            return

        self.scale_factor = new_scale
        self.update_image()
        self.adjust_scrollbars(factor)

    def adjust_scrollbars(self, factor):
        """Ensures the view stays centered on the current area when zooming."""
        self.adjust_scrollbar(self.scroll_area.horizontalScrollBar(), factor)
        self.adjust_scrollbar(self.scroll_area.verticalScrollBar(), factor)

    def adjust_scrollbar(self, scrollbar, factor):
        """Calculates the new position for a specific scrollbar."""
        current_value = scrollbar.value()
        page_step = scrollbar.pageStep()
        new_value = int(factor * current_value + ((factor - 1) * page_step / 2))
        scrollbar.setValue(new_value)

class ClickableThumbnail(QWidget):
    """A custom widget containing a thumbnail and a delete button."""
    delete_requested = Signal(QWidget, str)

    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.image_path = image_path
        self.setFixedSize(150, 210)
        self.setCursor(Qt.PointingHandCursor)

        # Image Label acting as the background
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(150, 210)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #45475A; 
                border-radius: 8px; 
                background-color: #313244;
            }
            QLabel:hover {
                border: 2px solid #89B4FA;
            }
        """)
        
        pixmap = QPixmap(image_path)
        scaled_pixmap = pixmap.scaled(146, 206, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        # Delete Button floating on top right
        self.delete_btn = QPushButton("X", self)
        self.delete_btn.setGeometry(120, 5, 25, 25)
        self.delete_btn.setCursor(Qt.PointingHandCursor)
        self.delete_btn.setStyleSheet("""
            QPushButton {
                background-color: rgba(243, 139, 168, 0.85);
                color: #1E1E2E;
                border-radius: 12px;
                font-weight: bold;
                font-size: 14px;
                border: none;
                padding: 0px;
                text-align: center;
                padding-bottom: 2px;
            }
            QPushButton:hover {
                background-color: rgba(230, 69, 83, 1.0);
                color: white;
            }
        """)
        self.delete_btn.clicked.connect(self.request_delete)

    def request_delete(self):
        """Emits signal to main window to remove this widget and delete the cached file."""
        self.delete_requested.emit(self, self.image_path)

    def mousePressEvent(self, event):
        """Opens the preview dialog if the user clicks anywhere on the image (excluding the delete button)."""
        if event.button() == Qt.LeftButton:
            dialog = ImagePreviewDialog(self.image_path, self.window())
            dialog.exec()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manga Translator")
        self.setMinimumSize(800, 650) 

        if getattr(sys, 'frozen', False):
            app_root = os.path.dirname(sys.executable)
        else:
            app_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        icon_path = os.path.join(app_root, "assets", "images", "favicon.ico")
        self.setWindowIcon(QIcon(icon_path))
        
        self.worker = None
        self.loader_worker = None
        self.loaded_models = None 
        self.current_loaded_gpu = None 
        self.is_running = False
        self.pytorch_tainted = False 
        
        self.selected_images = []
        self.translated_images_cache = []

        self.startup_gpu = None
        self.startup_engine = "Local LLM" # Default engine

        # Parse arguments for GPU and Engine state after restart
        args = sys.argv
        if "--gpu" in args:
            idx = args.index("--gpu")
            if idx + 1 < len(args): self.startup_gpu = args[idx + 1]
            
        if "--engine" in args:
            idx = args.index("--engine")
            if idx + 1 < len(args): self.startup_engine = args[idx + 1]

        main_widget = QWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(25, 25, 25, 25) 
        layout.setSpacing(15) 
        main_widget.setLayout(layout)
        self.setCentralWidget(main_widget)

        # --- Row 1: Image Selection ---
        input_layout = QHBoxLayout()
        self.image_info_label = QLabel("No images selected.")
        self.image_info_label.setStyleSheet("color: #A6E3A1;")
        
        self.select_images_btn = QPushButton("Select Images...")
        self.select_images_btn.setCursor(Qt.PointingHandCursor)
        self.select_images_btn.clicked.connect(self.browse_images)
        
        input_layout.addWidget(self.image_info_label)
        input_layout.addStretch()
        input_layout.addWidget(self.select_images_btn)
        layout.addLayout(input_layout)

        # Create a single Grid Layout to keep Row 2 and Row 3 perfectly aligned
        grid_layout = QGridLayout()

        # Add a 20px gap between the left components (Genre/API Key) and right components (Language/Engine)
        grid_layout.setHorizontalSpacing(20)

        # Force Column 1 (the text input fields) to expand and absorb all available horizontal space
        grid_layout.setColumnStretch(1, 1)

        # --- Row 2: Translation Options ---
        # Placed at grid row 0
        grid_layout.addWidget(QLabel("Genre:"), 0, 0)

        self.genre_entry = QLineEdit()
        self.genre_entry.setText("General Manga")
        grid_layout.addWidget(self.genre_entry, 0, 1)

        grid_layout.addWidget(QLabel("Language:"), 0, 2)

        self.lang_combo = QComboBox()
        self.lang_combo.addItems(["English", "Vietnamese"])
        grid_layout.addWidget(self.lang_combo, 0, 3)


        # --- Row 3: Engine Selection ---
        # Placed at grid row 1 (perfectly aligned with row 0)
        self.api_key_label = QLabel("API Key:")
        self.api_key_entry = QLineEdit()
        self.api_key_entry.setPlaceholderText("Enter Gemini API Key here...")
        self.api_key_entry.setEchoMode(QLineEdit.Password) # Hide text like a password

        grid_layout.addWidget(self.api_key_label, 1, 0)
        grid_layout.addWidget(self.api_key_entry, 1, 1)

        grid_layout.addWidget(QLabel("Engine:"), 1, 2)

        self.engine_combo = QComboBox()
        self.engine_combo.addItems(["Local LLM", "Gemini API"])
        self.engine_combo.setCurrentText(self.startup_engine)
        self.engine_combo.currentTextChanged.connect(self.on_engine_changed)
        grid_layout.addWidget(self.engine_combo, 1, 3)


        # --- API Key Visibility Setup ---
        # Set initial visibility based on the selected startup engine
        is_gemini = (self.startup_engine == "Gemini API")
        self.api_key_label.setVisible(is_gemini)
        self.api_key_entry.setVisible(is_gemini)

        # Add the perfectly aligned grid block to the main layout
        layout.addLayout(grid_layout)

        # --- Row 4: GPU Selection ---
        gpu_layout = QHBoxLayout()
        self.gpu_combo = QComboBox()
        gpu_layout.addWidget(QLabel("Device (GPU):"))
        gpu_layout.addWidget(self.gpu_combo)
        layout.addLayout(gpu_layout)

        # --- Row 5: Status & Progress ---
        layout.addSpacing(10)
        self.status_label = QLabel("Status: Ready")
        self.status_label.setObjectName("statusLabel")
        layout.addWidget(self.status_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.progress_bar)

        # --- Row 6: Image Preview Filmstrip ---
        layout.addWidget(QLabel("Live Preview (Click image to expand):"))
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFixedHeight(250)
        self.scroll_area.setStyleSheet("""
            QScrollArea { 
                border: 2px solid #313244; 
                border-radius: 8px; 
                background-color: #1E1E2E; 
            }
            QScrollBar:horizontal {
                background: #1E1E2E;
                height: 12px;
            }
            QScrollBar::handle:horizontal {
                background: #45475A;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal:hover {
                background: #585B70;
            }
            QScrollBar::add-line, QScrollBar::sub-line {
                background: none;
            }
        """)
        
        self.preview_container = QWidget()
        self.preview_container.setStyleSheet("background-color: transparent;")
        self.preview_layout = QHBoxLayout()
        self.preview_layout.setAlignment(Qt.AlignLeft)
        self.preview_layout.setSpacing(15)
        self.preview_container.setLayout(self.preview_layout)
        self.scroll_area.setWidget(self.preview_container)
        layout.addWidget(self.scroll_area)

        # --- Row 7: Action Buttons ---
        button_layout = QHBoxLayout()
        
        self.action_btn = QPushButton("START TRANSLATION")
        self.action_btn.setObjectName("actionBtn")
        self.action_btn.setMinimumHeight(50)
        self.action_btn.setCursor(Qt.PointingHandCursor)
        self.action_btn.clicked.connect(self.handle_action_button)
        
        self.save_btn = QPushButton("SAVE RESULTS")
        self.save_btn.setObjectName("saveBtn")
        self.save_btn.setMinimumHeight(50)
        self.save_btn.setCursor(Qt.PointingHandCursor)
        self.save_btn.setEnabled(False)
        self.save_btn.clicked.connect(self.save_translated_images)
        
        button_layout.addWidget(self.action_btn, stretch=2)
        button_layout.addWidget(self.save_btn, stretch=1)
        layout.addLayout(button_layout)

        self.load_gpus()
        self.apply_styles()

    def apply_styles(self):
        qss = """
        QMainWindow { background-color: #1E1E2E; }
        QLabel { color: #CDD6F4; font-size: 13px; font-weight: bold; }
        QLineEdit { background-color: #313244; color: #CDD6F4; border: 2px solid #45475A; border-radius: 8px; padding: 8px 12px; font-size: 13px; }
        QLineEdit:focus { border: 2px solid #89B4FA; }
        QComboBox { background-color: #45475A; color: #CDD6F4; border: 2px solid #313244; border-radius: 8px; padding: 8px 12px; font-size: 13px; font-weight: bold; }
        QComboBox:hover { background-color: #585B70; }
        QComboBox::drop-down { subcontrol-origin: padding; subcontrol-position: top right; width: 30px; border-left: 2px solid #313244; }
        QComboBox::down-arrow { width: 6px; height: 6px; background-color: #CDD6F4; border-radius: 3px; margin-right: 3px; }
        QComboBox::down-arrow:hover { background-color: #FFFFFF; }
        QComboBox QAbstractItemView { background-color: #313244; color: #CDD6F4; selection-background-color: #585B70; border-radius: 4px; }
        QPushButton { background-color: #45475A; color: #CDD6F4; border: none; border-radius: 8px; padding: 8px 15px; font-size: 13px; font-weight: bold; }
        QPushButton:hover { background-color: #585B70; }
        QPushButton:pressed { background-color: #313244; }
        QPushButton:disabled { background-color: #313244; color: #585B70; }
        QProgressBar { background-color: #313244; border: none; border-radius: 10px; color: white; font-weight: bold; text-align: center; height: 20px; }
        QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #89B4FA, stop:1 #B4BEFE); border-radius: 10px; }
        QLabel#statusLabel { color: #A6E3A1; font-size: 14px; }
        QPushButton#actionBtn { background-color: #89B4FA; color: #1E1E2E; border-radius: 10px; font-size: 14px; font-weight: 900; }
        QPushButton#actionBtn:hover { background-color: #B4BEFE; }
        QPushButton#actionBtn:pressed { background-color: #74C7EC; }
        QPushButton#actionBtn[state="cancel"] { background-color: #F38BA8; color: #1E1E2E; }
        QPushButton#actionBtn[state="cancel"]:hover { background-color: #eba0b5; }
        QPushButton#actionBtn[state="cancel"]:pressed { background-color: #e64553; }
        QPushButton#actionBtn:disabled { background-color: #45475A; color: #7F849C; }
        
        QPushButton#saveBtn { background-color: #A6E3A1; color: #1E1E2E; border-radius: 10px; font-size: 14px; font-weight: 900; }
        QPushButton#saveBtn:hover { background-color: #94E2D5; }
        QPushButton#saveBtn:pressed { background-color: #74C7EC; }
        QPushButton#saveBtn:disabled { background-color: #45475A; color: #7F849C; }
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

        if self.startup_gpu is not None:
            index = self.gpu_combo.findData(self.startup_gpu)
            if index >= 0:
                self.gpu_combo.setCurrentIndex(index)

    def browse_images(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, 
            "Select Manga Images", 
            "", 
            "Images (*.png *.jpg *.jpeg *.webp)"
        )
        if files:
            self.selected_images = files
            self.image_info_label.setText(f"{len(files)} image(s) selected ready for translation.")

    def handle_action_button(self, checked=False):
        if not self.is_running:
            self.prepare_and_start_translation()
        else:
            self.cancel_translation()

    def restart_app(self, target_gpu=None, target_engine=None):
        self.status_label.setText("Status: Restarting application to clear VRAM...")
        QApplication.processEvents() 
        new_args = []
        
        # Strip old arguments
        skip_next = False
        for arg in sys.argv:
            if skip_next:
                skip_next = False
                continue
            if arg in ["--gpu", "--engine"]:
                skip_next = True
                continue
            new_args.append(arg)
            
        # Append new arguments
        if target_gpu is not None:
            new_args.extend(["--gpu", str(target_gpu)])
        if target_engine is not None:
            new_args.extend(["--engine", str(target_engine)])
            
        os.execl(sys.executable, sys.executable, *new_args)

    def prepare_and_start_translation(self):
        if not self.selected_images:
            QMessageBox.warning(self, "Error", "Please select at least one image to translate!")
            return

        # Validate API Key if Gemini is selected
        selected_engine = self.engine_combo.currentText()
        if selected_engine == "Gemini API" and not self.api_key_entry.text().strip():
            QMessageBox.warning(self, "Error", "Please enter a valid Gemini API Key!")
            return

        selected_gpu = self.gpu_combo.currentData()

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
        self.engine_combo.setEnabled(False) # Lock engine combo
        self.api_key_entry.setEnabled(False) # Lock api key entry
        self.select_images_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        
        self.progress_bar.setValue(0)
        self.progress_bar.setStyleSheet("QProgressBar::chunk { background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:0, stop:0 #89B4FA, stop:1 #B4BEFE); border-radius: 10px; }")
        
        self.clear_preview_area()
        self.translated_images_cache = []

        if self.loaded_models is None:
            self.load_models_to_gpu(selected_gpu, selected_engine) # Pass engine here
        else:
            self.run_translation_worker()

    def on_engine_changed(self, new_engine):
        """Handle UI changes and VRAM clearing when switching engines."""
        is_gemini = (new_engine == "Gemini API")
        self.api_key_label.setVisible(is_gemini)
        self.api_key_entry.setVisible(is_gemini)
        
        # If the user switches from Local LLM to Gemini AFTER models are loaded, force a restart to clear VRAM
        if is_gemini and self.loaded_models is not None and self.loaded_models.get('translator_model') is not None:
            reply = QMessageBox.question(
                self, 
                "Restart Required", 
                "Switching to Gemini API requires restarting the app to free up the VRAM used by the Local LLM.\n\nRestart now?",
                QMessageBox.Yes | QMessageBox.No, 
                QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                self.restart_app(target_gpu=self.gpu_combo.currentData(), target_engine=new_engine)
            else:
                # Revert selection
                self.engine_combo.setCurrentText("Local LLM")

    def clear_preview_area(self):
        """Removes all thumbnails from the preview layout"""
        while self.preview_layout.count():
            item = self.preview_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def load_models_to_gpu(self, gpu_id, engine):
        self.pytorch_tainted = True 
        self.action_btn.setEnabled(False)
        self.action_btn.setText("LOADING MODELS... (DO NOT CLOSE)")
        
        # Pass both GPU and Engine to the loader
        self.loader_worker = ModelLoaderWorker(gpu_id, engine) 
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
            'image_paths': self.selected_images,
            'gpu_id': self.current_loaded_gpu,
            'genre': self.genre_entry.text().strip() or "General Manga",
            'target_lang': self.lang_combo.currentText().strip() or "Vietnamese",
            'engine': self.engine_combo.currentText(), # Add engine
            'api_key': self.api_key_entry.text().strip() # Add API Key
        }

        self.worker = AITranslatorWorker(config, self.loaded_models)
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.image_translated.connect(self.on_image_translated)
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

    def scroll_to_end(self):
        """Helper to force the scrollbar to the far right"""
        scrollbar = self.scroll_area.horizontalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def on_image_translated(self, cached_image_path):
        """Triggered every time a single image finishes processing"""
        self.translated_images_cache.append(cached_image_path)
        
        thumbnail = ClickableThumbnail(cached_image_path)
        # Connect the custom delete signal to handle removal
        thumbnail.delete_requested.connect(self.remove_thumbnail)
        self.preview_layout.addWidget(thumbnail)
        
        self.save_btn.setEnabled(True)
        QTimer.singleShot(50, self.scroll_to_end)

    def remove_thumbnail(self, widget, image_path):
        """Handles the removal of a specific image from the UI and cache."""
        if image_path in self.translated_images_cache:
            self.translated_images_cache.remove(image_path)
            
        self.preview_layout.removeWidget(widget)
        widget.deleteLater()
        
        # Disable save button if no images are left
        if not self.translated_images_cache:
            self.save_btn.setEnabled(False)

    def on_finished(self, message):
        self.progress_bar.setValue(100)
        self.status_label.setText("All Done! Please review and save your images.")
        self.unlock_ui()
        if self.translated_images_cache:
            self.save_btn.setEnabled(True)
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

    def save_translated_images(self):
        """Allows user to select a folder and copies cached images to it."""
        if not self.translated_images_cache:
            return
            
        dest_dir = QFileDialog.getExistingDirectory(self, "Select Folder to Save Translated Images")
        if dest_dir:
            try:
                for img_path in self.translated_images_cache:
                    img_name = os.path.basename(img_path)
                    target_path = os.path.join(dest_dir, img_name)
                    shutil.copy2(img_path, target_path)
                QMessageBox.information(self, "Saved", f"Successfully saved {len(self.translated_images_cache)} images to:\n{dest_dir}")
            except Exception as e:
                QMessageBox.critical(self, "Save Error", f"Could not save images: {str(e)}")

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
        self.select_images_btn.setEnabled(True)
        self.engine_combo.setEnabled(True)
        self.api_key_entry.setEnabled(True)
        self.select_images_btn.setEnabled(True)