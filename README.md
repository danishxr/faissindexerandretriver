# Document Search and Indexing Application
This application provides a user-friendly interface for indexing and searching PDF documents using vector embeddings. It leverages FAISS (Facebook AI Similarity Search) for efficient similarity search and Google's Gemini API for generating high-quality embeddings.

## Features
- PDF Indexing : Index PDF documents in a selected folder to create searchable vector embeddings
- Semantic Search : Search through indexed documents using natural language queries
- User-Friendly Interface : Simple PySide6-based UI for easy interaction
- Cross-Platform : Works on Windows, macOS, and Linux
## Prerequisites
- Python 3.8 or higher
- Google API key for Gemini (stored in a .env file)
## Installation
1. Clone the repository:

```bash
git clone <repository-url>
cd faiss_testing
```


2. Install the required dependencies:
```bash
cd my_ui/pyside6_ui
pip install -r requirements.txt
```
3. Create a `.env` file in the `my_ui/pyside6_ui/backend` directory with your Google API key:
```bash
GOOGLE_API_KEY=your_google_api_key_here
```

## Running the Application
Navigate to the application directory and run the main script:

```bash
cd my_ui/pyside6_ui
python app.py
```
## How to Use
1. Index Documents :
   
   - Select "Index this folder" option
   - Click "Browse" to select a folder containing PDF files
   - Click "Index Folder" to start the indexing process
   - Wait for the indexing to complete
2. Search Documents :
   
   - Select "Search using this folder's index" option
   - Click "Browse" to select a folder containing index files ( `faiss_index.idx` and `metadata.pkl` )
   - Enter your search query in the text field
   - Click "Search" to find relevant documents
   - Click on the search results to open the corresponding PDF files
## Creating Executable with PyInstaller
### For Windows, MacOS
```bash
cd my_ui/pyside6_ui
pip install pyinstaller
pyinstaller --name DocumentSearch --windowed --add-data "backend:backend" --add-data "utils:utils" app.py
```
The executable will be created in the `dist/DocumentSearch` directory.


The application bundle will be created in the dist directory.


## Important Notes
- When distributing the application, users will still need to provide their own Google API key in a .env file
- The application requires internet access to use the Gemini API for generating embeddings
- Large PDF collections may require significant memory and disk space for indexing
## Troubleshooting
- If you encounter issues with missing dependencies when running the executable, try adding them explicitly to the PyInstaller command with --hidden-import
- For large PDF collections, consider increasing the memory limit when building with PyInstaller using --hidden-import=numpy and other relevant packages