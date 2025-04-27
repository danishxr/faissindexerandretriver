import sys
import os
import base64
from pathlib import Path
import PyPDF2
# In the imports section
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLineEdit, QPushButton, QTextBrowser, QFileDialog, QLabel, QProgressBar,
    QMessageBox, QRadioButton, QButtonGroup
)
from PySide6.QtCore import Qt, QThread, Signal, QByteArray, QBuffer, QIODevice
from PySide6.QtGui import QPixmap
import qtawesome as qta

# Add parent directory to path to import backend modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from backend.main import RAGSystem, FAISSVectorStore
from backend.gemini_client import GeminiClient
from utils.logger import setup_logger

# Initialize logger
logger = setup_logger('app')


class IndexingWorker(QThread):
    """Worker thread for indexing documents to avoid UI freezing"""
    finished = Signal(str)
    progress = Signal(int)
    
    def __init__(self, folder_path):
        super().__init__()
        self.folder_path = folder_path
        logger.info(f"Initializing IndexingWorker with folder: {folder_path}")
        # Create a vector store with paths in the selected folder
        index_file = os.path.join(self.folder_path, "faiss_index.idx")
        metadata_file = os.path.join(self.folder_path, "metadata.pkl")
        logger.info(f"Setting up vector store with index file: {index_file} and metadata file: {metadata_file}")
        vector_store = FAISSVectorStore(index_file=index_file, metadata_file=metadata_file)
        self.rag_system = RAGSystem(vector_store=vector_store)
        
    def run(self):
        try:
            logger.info(f"Starting indexing process for folder: {self.folder_path}")
            # Get all PDF files in the folder
            pdf_files = []
            for root, _, files in os.walk(self.folder_path):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_files.append(os.path.join(root, file))
            
            logger.info(f"Found {len(pdf_files)} PDF files in folder")
            
            if not pdf_files:
                logger.warning(f"No PDF files found in folder: {self.folder_path}")
                self.finished.emit("No PDF files found in the selected folder.")
                return
                
            # Process each PDF file
            for i, pdf_path in enumerate(pdf_files):
                try:
                    logger.info(f"Processing PDF {i+1}/{len(pdf_files)}: {pdf_path}")
                    # Extract text from PDF using PyPDF2 instead
                    with open(pdf_path, 'rb') as file:
                        reader = PyPDF2.PdfReader(file)
                        document_text = ""
                        for page_num in range(len(reader.pages)):
                            document_text += reader.pages[page_num].extract_text() + "\n"
                    
                    # Create metadata with filepath and filename
                    metadata = {
                        "filepath": pdf_path,
                        "filename": os.path.basename(pdf_path)
                    }
                    
                    # Add document to RAG system with metadata
                    logger.info(f"Adding document to RAG system: {os.path.basename(pdf_path)}")
                    self.rag_system.add_documents([document_text], [metadata])
                    
                    # Update progress
                    progress_value = int((i + 1) / len(pdf_files) * 100)
                    self.progress.emit(progress_value)
                except Exception as e:
                    logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
                    self.finished.emit(f"Error processing {os.path.basename(pdf_path)}: {str(e)}")
                    return
            
            logger.info(f"Successfully indexed {len(pdf_files)} PDF documents")
            self.finished.emit(f"Successfully indexed {len(pdf_files)} PDF documents.")
        except Exception as e:
            logger.error(f"Error during indexing: {str(e)}", exc_info=True)
            self.finished.emit(f"Error during indexing: {str(e)}")


class SearchApp(QMainWindow):
    def __init__(self):
        super().__init__()
        logger.info("Initializing SearchApp")
        self.init_ui()
        
    def init_ui(self):
        logger.info("Setting up UI components")
        self.setWindowTitle("Document Search and Indexing")
        # Add a settings icon to the window
        self.setWindowIcon(qta.icon('fa5s.cog'))
        self.setMinimumSize(600, 400)
        
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        
        # Search section
        search_label = QLabel("Enter your search query:")
        self.search_input = QLineEdit()
        search_button = QPushButton("Search")
        search_button.setIcon(qta.icon('mdi.magnify'))
        search_button.clicked.connect(self.search_documents)
        
        search_layout = QHBoxLayout()
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_button)
        
        # Folder selection section (combined for both indexing and searching)
        folder_label = QLabel("Select folder:")
        self.folder_input = QLineEdit()
        self.folder_input.setReadOnly(True)
        browse_button = QPushButton("Browse")
        browse_button.setIcon(qta.icon('mdi.folder-open'))
        browse_button.clicked.connect(self.browse_folder)
        
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(self.folder_input)
        folder_layout.addWidget(browse_button)
        
        # Radio buttons for operation selection
        operation_layout = QHBoxLayout()
        self.index_radio = QRadioButton("Index this folder")
        self.search_radio = QRadioButton("Search using this folder's index")
        self.index_radio.setChecked(True)  # Default to index operation
        
        # Group radio buttons
        self.operation_group = QButtonGroup()
        self.operation_group.addButton(self.index_radio)
        self.operation_group.addButton(self.search_radio)
        
        operation_layout.addWidget(self.index_radio)
        operation_layout.addWidget(self.search_radio)
        
        # Action button (will be either Index or Search based on radio selection)
        self.action_button = QPushButton("Index Folder")
        self.action_button.setIcon(qta.icon('mdi.database-plus'))
        self.action_button.clicked.connect(self.perform_action)
        
        # Progress bar for indexing
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        
        # Results area
        results_label = QLabel("Results:")
        self.results_area = QTextBrowser()
        self.results_area.setReadOnly(True)
        
        # Add all widgets to main layout
        main_layout.addWidget(search_label)
        main_layout.addLayout(search_layout)
        main_layout.addSpacing(20)
        main_layout.addWidget(folder_label)
        main_layout.addLayout(folder_layout)
        main_layout.addLayout(operation_layout)
        main_layout.addWidget(self.action_button)
        main_layout.addWidget(self.progress_bar)
        main_layout.addSpacing(20)
        main_layout.addWidget(results_label)
        main_layout.addWidget(self.results_area)
        
        # Connect radio button changes to update action button text
        self.index_radio.toggled.connect(self.update_action_button)
        
        self.setCentralWidget(main_widget)
        logger.info("UI setup complete")
    
    def update_action_button(self):
        """Update the action button text based on selected operation"""
        if self.index_radio.isChecked():
            self.action_button.setText("Index Folder")
            self.action_button.setIcon(qta.icon('mdi.database-plus'))
        else:
            self.action_button.setText("Search")
            self.action_button.setIcon(qta.icon('mdi.magnify'))
    
    def browse_folder(self):
        logger.info("Browse folder button clicked")
        folder_path = QFileDialog.getExistingDirectory(
            self, "Select Folder", os.path.expanduser("~")
        )
        if folder_path:
            logger.info(f"Selected folder: {folder_path}")
            self.folder_input.setText(folder_path)
            
            # If search is selected, check if the folder has index files
            if self.search_radio.isChecked():
                self.validate_index_folder(folder_path, show_success=False)
    
    def validate_index_folder(self, folder_path, show_success=True):
        """Validate that the folder contains required index files"""
        index_file = os.path.join(folder_path, "faiss_index.idx")
        metadata_file = os.path.join(folder_path, "metadata.pkl")
        
        if not os.path.exists(index_file) or not os.path.exists(metadata_file):
            logger.warning(f"Missing index files in folder: {folder_path}")
            QMessageBox.warning(
                self,
                "Missing Index Files",
                f"The selected folder is missing required index files.\n"
                f"Please select a folder containing both 'faiss_index.idx' and 'metadata.pkl' files,\n"
                f"or switch to 'Index this folder' to create new index files.",
                QMessageBox.Ok
            )
            return False
        
        if show_success:
            logger.info(f"Valid index folder selected: {folder_path}")
            QMessageBox.information(
                self,
                "Valid Index Folder",
                f"The selected folder contains valid index files.",
                QMessageBox.Ok
            )
        return True
    
    def perform_action(self):
        """Perform the selected action (index or search)"""
        folder_path = self.folder_input.text()
        if not folder_path:
            logger.warning("No folder selected")
            self.results_area.setText("Please select a folder first.")
            return
        
        if self.index_radio.isChecked():
            self.index_folder()
        else:
            # For search operation, validate the folder has index files
            if self.validate_index_folder(folder_path):
                query = self.search_input.text()
                if query:
                    self.search_documents()
                else:
                    logger.warning("Empty search query")
                    self.results_area.setText("Please enter a search query.")
    
    def index_folder(self):
        folder_path = self.folder_input.text()
        if not folder_path:
            logger.warning("No folder selected for indexing")
            self.results_area.setText("Please select a folder first.")
            return
        
        logger.info(f"Starting indexing process for folder: {folder_path}")
        # Show progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        
        # Create and start worker thread
        self.indexing_worker = IndexingWorker(folder_path)
        self.indexing_worker.progress.connect(self.update_progress)
        self.indexing_worker.finished.connect(self.indexing_finished)
        self.indexing_worker.start()
        
        # Disable UI elements during indexing
        self.folder_input.setEnabled(False)
        self.results_area.setText("Indexing in progress...")
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
        logger.debug(f"Indexing progress: {value}%")
    
    def indexing_finished(self, message):
        logger.info(f"Indexing finished: {message}")
        # Re-enable UI elements
        self.folder_input.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Show result message
        self.results_area.setText(message)
        
        # Check if the message indicates no PDF files were found
        if "No PDF files found" in message:
            logger.warning("No PDF files found warning displayed")
            QMessageBox.warning(
                self,
                "No PDF Files Found",
                "No PDF files were found in the selected folder.",
                QMessageBox.Ok
            )
    
    def search_documents(self):
        query = self.search_input.text()
        if not query:
            logger.warning("Empty search query")
            self.results_area.setText("Please enter a search query.")
            return
        
        logger.info(f"Search query: {query}")
        
        try:
            # Get the folder path from the folder input
            folder_path = self.folder_input.text()
            if not folder_path:
                logger.warning("No folder selected for search")
                self.results_area.setText("Please select a folder first.")
                return
                
            logger.info(f"Using folder: {folder_path}")
            # Create a vector store with paths in the selected folder
            index_file = os.path.join(folder_path, "faiss_index.idx")
            metadata_file = os.path.join(folder_path, "metadata.pkl")
            
            # Check if index exists in the folder
            if not os.path.exists(index_file) or not os.path.exists(metadata_file):
                logger.error(f"Missing index files in folder: {folder_path}")
                QMessageBox.critical(
                    self,
                    "Missing Index Files",
                    f"The selected folder is missing required index files.\n"
                    f"Please select a folder containing both 'faiss_index.idx' and 'metadata.pkl' files,\n"
                    f"or switch to 'Index this folder' to create new index files.",
                    QMessageBox.Ok
                )
                self.results_area.setText(f"Missing index files in folder: {folder_path}")
                return
                
            logger.info(f"Loading vector store from {index_file} and {metadata_file}")
            vector_store = FAISSVectorStore(index_file=index_file, metadata_file=metadata_file)
            rag_system = RAGSystem(vector_store=vector_store)
            
            # Use the RAG system to query
            logger.info("Performing similarity search")
            results = rag_system.vector_store.similarity_search(query, k=1)  # Only get 1 result
            
            # Check if any results were found
            if not results:
                logger.warning(f"No search results found for query: {query}")
                # Show dialog box for no results
                QMessageBox.information(
                    self,
                    "No Results Found",
                    f"No documents found with the search text: '{query}'",
                    QMessageBox.Ok
                )
                self.results_area.setText("No Files found")
                return
            
            logger.info(f"Found {len(results)} search results")
            
            # Get the first (and only) result
            result = results[0]
            filename = result["metadata"].get("filename", "Unknown")
            filepath = result["metadata"].get("filepath", "")
            
            # Store the filepath as a property of the results_area for later use
            self.results_area.setProperty("current_pdf_path", filepath)
            
            # Create HTML with PDF icon and filename as a clickable link
            pdf_icon = qta.icon('mdi.file-pdf').pixmap(32, 32).toImage()
            pdf_icon_base64 = self._pixmap_to_base64(pdf_icon)
            
            result_html = f"""
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <img src="data:image/png;base64,{pdf_icon_base64}" style="margin-right: 10px;" width="32" height="32"/>
                <a href="file://{filepath}" style="font-size: 16px; text-decoration: none; color: #0066cc;">{filename}</a>
            </div>
            """
            
            # Set the HTML content
            self.results_area.setHtml(result_html)
            
            # Connect the anchorClicked signal to open_pdf method
            self.results_area.setOpenLinks(False)
            self.results_area.anchorClicked.connect(self.open_pdf)
            
            logger.info("Search completed successfully")
            
        except Exception as e:
            logger.error(f"Error during search: {str(e)}", exc_info=True)
            self.results_area.setText(f"Error during search: {str(e)}")


    def _pixmap_to_base64(self, image):
        """
        Convert a QImage to base64 string.
        
        Args:
            image: QImage to convert
            
        Returns:
            Base64 encoded string of the image
        """
        byte_array = QByteArray()
        buffer = QBuffer(byte_array)
        buffer.open(QIODevice.WriteOnly)
        image.save(buffer, "PNG")
        return base64.b64encode(byte_array).decode('utf-8')
    
    def open_pdf(self, url):
        """
        Open the PDF file when clicked.
        
        Args:
            url: The URL of the PDF file to open
        """
        try:
            # Get the file path from the URL
            file_path = self.results_area.property("current_pdf_path")
            if file_path and os.path.exists(file_path):
                logger.info(f"Opening PDF file: {file_path}")
                # Use the platform-specific method to open the file
                if sys.platform == 'darwin':  # macOS
                    os.system(f"open '{file_path}'")
                elif sys.platform == 'win32':  # Windows
                    os.system(f'start "" "{file_path}"')
                else:  # Linux and other Unix-like
                    os.system(f"xdg-open '{file_path}'")
            else:
                logger.warning(f"PDF file not found: {file_path}")
                QMessageBox.warning(
                    self,
                    "File Not Found",
                    f"The PDF file could not be found at the specified location.",
                    QMessageBox.Ok
                )
        except Exception as e:
            logger.error(f"Error opening PDF file: {str(e)}", exc_info=True)
            QMessageBox.critical(
                self,
                "Error Opening File",
                f"An error occurred while trying to open the PDF file: {str(e)}",
                QMessageBox.Ok
            )


def main():
    logger.info("Starting application")
    app = QApplication(sys.argv)
    window = SearchApp()
    window.show()
    logger.info("Application UI displayed")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
