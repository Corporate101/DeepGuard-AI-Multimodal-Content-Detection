import os
import magic
from PyPDF2 import PdfReader
from docx import Document
import tempfile

class FileProcessor:
    def __init__(self):
        self.mime = magic.Magic(mime=True)
    
    def detect_file_type(self, filename):
        """Detect file type based on extension and content"""
        extension = os.path.splitext(filename)[1].lower()
        
        text_extensions = ['.txt', '.pdf', '.docx', '.odt']
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
        audio_extensions = ['.mp3', '.wav', '.flac', '.m4a', '.aac']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        
        if extension in text_extensions:
            return "text"
        elif extension in image_extensions:
            return "image"
        elif extension in audio_extensions:
            return "audio"
        elif extension in video_extensions:
            return "video"
        else:
            return "unknown"
    
    def extract_text(self, file_path):
        """Extract text from various file formats"""
        file_type = self.detect_file_type(file_path)
        
        try:
            if file_path.endswith('.pdf'):
                return self._extract_from_pdf(file_path)
            elif file_path.endswith('.docx'):
                return self._extract_from_docx(file_path)
            elif file_path.endswith('.txt'):
                return self._extract_from_txt(file_path)
            else:
                return None
        except Exception as e:
            print(f"Error extracting text from {file_path}: {str(e)}")
            return None
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF files"""
        text = ""
        with open(file_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX files"""
        doc = Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    
    def _extract_from_txt(self, file_path):
        """Extract text from TXT files"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    
    def get_file_info(self, file_path):
        """Get comprehensive file information"""
        file_stats = os.stat(file_path)
        
        return {
            'size': file_stats.st_size,
            'modified_time': file_stats.st_mtime,
            'file_type': self.detect_file_type(file_path),
            'mime_type': self.mime.from_file(file_path)
        }
