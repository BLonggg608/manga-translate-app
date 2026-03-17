import sys
import os
from PySide6.QtWidgets import QApplication

# Append the root directory to sys.path so Python can find the 'app' module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui.main_window import MainWindow

def main():
    # Initialize the Qt Application
    app = QApplication(sys.argv)
    
    # Apply modern 'Fusion' style across all platforms (Windows/Mac/Linux)
    app.setStyle("Fusion") 
    
    # Create and display the main window
    window = MainWindow()
    window.show()
    
    # Start the application event loop
    sys.exit(app.exec())

if __name__ == "__main__":
    main()  