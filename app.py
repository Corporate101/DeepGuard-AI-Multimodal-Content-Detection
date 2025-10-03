import streamlit as st
import os
import tempfile
from src.file_processor import FileProcessor
from src.text_detector import TextDetector
from src.image_detector import ImageDetector
from src.audio_detector import AudioDetector
import plotly.graph_objects as go
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="DeepGuard AI - Content Authenticity Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

class DeepGuardApp:
    def __init__(self):
        self.file_processor = FileProcessor()
        self.text_detector = TextDetector()
        self.image_detector = ImageDetector()
        self.audio_detector = AudioDetector()
        
    def run(self):
        # Header
        col1, col2 = st.columns([1, 4])
        with col1:
            st.image("assets/logo.png", width=80)
        with col2:
            st.title("🔍 DeepGuard AI")
            st.markdown("**Multi-Modal Content Authenticity Detection System**")
        
        st.markdown("---")
        
        # Sidebar
        st.sidebar.title("Navigation")
        app_mode = st.sidebar.selectbox(
            "Choose Analysis Mode",
            ["🏠 Dashboard", "📄 Text Analysis", "🖼️ Image Analysis", "🎵 Audio Analysis", "📊 Results History"]
        )
        
        if app_mode == "🏠 Dashboard":
            self.show_dashboard()
        elif app_mode == "📄 Text Analysis":
            self.text_analysis()
        elif app_mode == "🖼️ Image Analysis":
            self.image_analysis()
        elif app_mode == "🎵 Audio Analysis":
            self.audio_analysis()
        elif app_mode == "📊 Results History":
            self.results_history()
    
    def show_dashboard(self):
        st.header("Welcome to DeepGuard AI")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **🔤 Text Analysis**
            - Detect AI-generated text content
            - Support for PDF, DOCX, TXT formats
            - Advanced NLP-based detection
            """)
        
        with col2:
            st.info("""
            **🖼️ Image Analysis**
            - Identify AI-generated images
            - Multiple image formats supported
            - Computer vision-based analysis
            """)
        
        with col3:
            st.info("""
            **🎵 Audio Analysis**
            - Detect synthetic audio content
            - Support for MP3, WAV formats
            - Audio pattern recognition
            """)
        
        st.markdown("---")
        
        # Quick start section
        st.subheader("🚀 Quick Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload a file for instant analysis",
            type=['txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'mp3', 'wav'],
            help="Supported formats: Text, PDF, DOCX, Images, Audio"
        )
        
        if uploaded_file is not None:
            self.quick_analysis(uploaded_file)
    
    def quick_analysis(self, uploaded_file):
        st.info(f"Analyzing: {uploaded_file.name}")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        try:
            file_type = self.file_processor.detect_file_type(uploaded_file.name)
            
            if file_type == "text":
                result = self.text_analysis_file(tmp_path)
            elif file_type == "image":
                result = self.image_analysis_file(tmp_path)
            elif file_type == "audio":
                result = self.audio_analysis_file(tmp_path)
            else:
                st.error("Unsupported file type for quick analysis")
                return
            
            self.display_results(result, uploaded_file.name)
            
        finally:
            os.unlink(tmp_path)
    
    def text_analysis(self):
        st.header("📄 Text Content Analysis")
        
        tab1, tab2, tab3 = st.tabs(["📝 Direct Input", "📁 File Upload", "⚙️ Advanced Settings"])
        
        with tab1:
            text_input = st.text_area(
                "Enter text to analyze",
                height=200,
                placeholder="Paste your text here for authenticity analysis..."
            )
            if st.button("Analyze Text", key="text_direct"):
                if text_input.strip():
                    with st.spinner("Analyzing text content..."):
                        result = self.text_detector.analyze_text(text_input)
                        self.display_text_results(result, "Direct Input")
                else:
                    st.warning("Please enter some text to analyze")
        
        with tab2:
            uploaded_file = st.file_uploader(
                "Upload text file",
                type=['txt', 'pdf', 'docx'],
                key="text_upload"
            )
            if uploaded_file is not None:
                if st.button("Analyze Uploaded File", key="text_file"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        with st.spinner("Processing and analyzing file..."):
                            extracted_text = self.file_processor.extract_text(tmp_path)
                            if extracted_text:
                                result = self.text_detector.analyze_text(extracted_text)
                                self.display_text_results(result, uploaded_file.name)
                            else:
                                st.error("Could not extract text from the file")
                    finally:
                        os.unlink(tmp_path)
        
        with tab3:
            st.subheader("Detection Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.5,
                max_value=0.95,
                value=0.75,
                help="Higher values require more confidence for AI detection"
            )
            st.info("Advanced model settings can be configured in the configuration files.")
    
    def image_analysis(self):
        st.header("🖼️ Image Content Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload an image for analysis",
            type=['png', 'jpg', 'jpeg', 'bmp', 'tiff'],
            key="image_upload"
        )
        
        if uploaded_file is not None:
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            
            with col2:
                if st.button("Analyze Image", key="image_analyze"):
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        tmp_path = tmp_file.name
                    
                    try:
                        with st.spinner("Analyzing image features..."):
                            result = self.image_detector.analyze_image(tmp_path)
                            self.display_image_results(result, uploaded_file.name)
                    finally:
                        os.unlink(tmp_path)
    
    def audio_analysis(self):
        st.header("🎵 Audio Content Analysis")
        
        uploaded_file = st.file_uploader(
            "Upload an audio file",
            type=['mp3', 'wav', 'flac', 'm4a'],
            key="audio_upload"
        )
        
        if uploaded_file is not None:
            st.audio(uploaded_file, format=uploaded_file.type)
            
            if st.button("Analyze Audio", key="audio_analyze"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                try:
                    with st.spinner("Analyzing audio patterns..."):
                        result = self.audio_detector.analyze_audio(tmp_path)
                        self.display_audio_results(result, uploaded_file.name)
                finally:
                    os.unlink(tmp_path)
    
    def results_history(self):
        st.header("📊 Analysis History")
        st.info("This feature would connect to a database in a production environment.")
        st.write("In a full implementation, this would show:")
        st.write("- Previous analysis results")
        st.write("- Statistical trends")
        st.write("- Export capabilities")
    
    def display_text_results(self, result, source_name):
        st.subheader(f"Analysis Results: {source_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence = result.get('confidence', 0)
            st.metric("Confidence Score", f"{confidence:.2%}")
        
        with col2:
            is_ai = result.get('is_ai_generated', False)
            status = "🤖 AI-Generated" if is_ai else "👤 Human-Written"
            st.metric("Prediction", status)
        
        with col3:
            st.metric("Analysis Time", f"{result.get('processing_time', 0):.2f}s")
        
        # Confidence visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = confidence * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Authenticity Confidence"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightcoral"},
                    {'range': [30, 70], 'color': "lightyellow"},
                    {'range': [70, 100], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
        
        # Feature analysis
        if 'features' in result:
            st.subheader("Feature Analysis")
            features = result['features']
            fig = px.bar(
                x=list(features.keys()),
                y=list(features.values()),
                title="Text Feature Scores"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def display_image_results(self, result, source_name):
        st.subheader(f"Image Analysis: {source_name}")
        
        col1, col2 = st.columns(2)
        
        with col1:
            confidence = result.get('confidence', 0)
            st.metric("AI Probability", f"{confidence:.2%}")
            
            is_ai = result.get('is_ai_generated', False)
            status = "🤖 AI-Generated" if is_ai else "👤 Authentic Photo"
            st.metric("Prediction", status)
        
        with col2:
            st.metric("Analysis Time", f"{result.get('processing_time', 0):.2f}s")
            if 'artifacts_detected' in result:
                st.metric("Artifacts Found", result['artifacts_detected'])
        
        # Display feature analysis if available
        if 'features' in result:
            st.subheader("Detection Features")
            for feature, value in result['features'].items():
                st.progress(value, text=f"{feature}: {value:.2f}")
    
    def display_audio_results(self, result, source_name):
        st.subheader(f"Audio Analysis: {source_name}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            confidence = result.get('confidence', 0)
            st.metric("Synthetic Probability", f"{confidence:.2%}")
        
        with col2:
            is_ai = result.get('is_ai_generated', False)
            status = "🤖 AI-Generated" if is_ai else "👤 Human Voice"
            st.metric("Prediction", status)
        
        with col3:
            st.metric("Analysis Time", f"{result.get('processing_time', 0):.2f}s")
        
        # Audio features visualization
        if 'audio_features' in result:
            st.subheader("Audio Feature Analysis")
            features = result['audio_features']
            fig = px.bar(
                x=list(features.keys()),
                y=list(features.values()),
                title="Audio Feature Scores"
            )
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    app = DeepGuardApp()
    app.run()
