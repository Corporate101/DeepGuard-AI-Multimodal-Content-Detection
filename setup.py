from setuptools import setup, find_packages

setup(
    name="deepguard-ai",
    version="1.0.0",
    description="Multi-Modal Content Authenticity Detection System",
    author="Michael Dempsey",
    author_email="michael.dempsey237@gmail.com",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.28.0",
        "numpy>=1.24.3",
        "pandas>=2.0.3",
        "scikit-learn>=1.3.0",
        "torch>=2.0.1",
        "transformers>=4.30.0",
        "Pillow>=10.0.0",
        "opencv-python>=4.8.0.74",
        "librosa>=0.10.0",
        "pydub>=0.25.1",
        "python-docx>=0.8.11",
        "PyPDF2>=3.0.1",
        "plotly>=5.15.0",
        "python-magic>=0.4.27",
    ],
    python_requires=">=3.8",
)
