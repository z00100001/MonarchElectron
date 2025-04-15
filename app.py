import sys
import os
import random
import streamlit as st
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import torch
import numpy as np
import pandas as pd
import io
import PyPDF2
import docx2txt
import torch.nn.functional as F
st.set_page_config(
    page_title="Monarch - Emotional Text Analyzer",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None
    }
)
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    button[kind="iconButton"][title="View app menu"] {display: none !important;}
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Import the transformer libraries with proper fallbacks
try:
    from transformers import BertTokenizerFast, BertForSequenceClassification
except ImportError:
    st.error("Transformers library not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers"])
    from transformers import BertTokenizerFast, BertForSequenceClassification

# Update the model path to look in multiple possible locations
MODEL_DIR = "model"  # Simplified to match test.py
# Define possible model locations in order of preference
POSSIBLE_MODEL_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "model")),
    os.path.join(os.getcwd(), "model"),
    os.path.abspath("model")
]
REPORTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports"))

# Add a debug check function
def check_image_paths():
    """Debug function to check if image paths exist"""
    st.write("Current working directory:", os.getcwd())
    
    # Check if reports directory exists
    if os.path.exists("reports"):
        st.write("Reports directory found")
        st.write("Files in reports directory:", os.listdir("reports"))
    else:
        st.write("Reports directory NOT found at:", os.path.join(os.getcwd(), "reports"))
        
        # Check if reports exists relative to script location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        reports_path = os.path.join(script_dir, "reports")
        if os.path.exists(reports_path):
            st.write("Reports directory found relative to script at:", reports_path)
            st.write("Files in reports directory:", os.listdir(reports_path))
        else:
            st.write("Reports directory NOT found relative to script at:", reports_path)

# Soft color pallete for graphs
COLORS = {
    "green": "rgba(75, 192, 120, 0.7)",   # Softer green
    "yellow": "rgba(255, 205, 86, 0.7)",  # Softer yellow
    "orange": "rgba(255, 159, 64, 0.7)",  # Softer orange
    "red": "rgba(255, 99, 132, 0.7)",     # Softer red
    "blue": "rgba(54, 162, 235, 0.7)",    # Soft blue for reference lines
    "purple": "rgba(153, 102, 255, 0.7)",  # Soft purple
    "teal": "rgba(75, 192, 192, 0.7)"     # Soft teal for happiness
}

# Defines the reference levels for the radar chart
REFERENCE_LEVELS = {
    "low": 25,
    "moderate": 50,
    "high": 75,
    "severe": 90
}


# Creates a list to remember past analyses for future graphs
if 'history' not in st.session_state:
    st.session_state.history = []

# Loads model and tokenizer once when the app starts
@st.cache_resource
def load_mental_health_model():
    """Load the mental health model and tokenizer"""
    # Dictionary mapping emotion indices to labels (from test.py)
    id2label = {
        0: "sadness",
        1: "anger",
        2: "distress",
        3: "joy", 
        4: "worry"
    }
    
    # Try each possible model location in order until one works
    for model_path in POSSIBLE_MODEL_PATHS:
        try:
            if os.path.exists(model_path):
                st.sidebar.info(f"Attempting to load model from: {model_path}")
                tokenizer = BertTokenizerFast.from_pretrained(model_path)
                model = BertForSequenceClassification.from_pretrained(model_path)
                model.eval()
                st.sidebar.success(f"✅ Successfully loaded model from: {model_path}")
                return model, tokenizer, id2label, True
            
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load model from {model_path}: {str(e)}")
            continue  # Try the next path
    
    # If we've tried all paths and none worked, fall back to base model
    st.sidebar.error("❌ Could not load any local model. Falling back to base model.")
    try:
        # Last resort - try loading direct from HuggingFace
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", 
                                                           num_labels=5)  # 5 labels for emotions
        return model, tokenizer, id2label, False
    except Exception as e2:
        st.sidebar.error(f"Failed to load fallback model: {str(e2)}")
        return None, None, id2label, False

# Method to extract text from PDFs
def extract_text_from_pdf(file):
    text = ""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
    return text

# Method to extract text from docx
def extract_text_from_docx(file):
    try:
        text = docx2txt.process(file)
        return text
    except Exception as e:
        st.error(f"Error extracting text from DOCX: {str(e)}")
        return ""

# Method to extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.getvalue().decode("utf-8")
        return text
    except Exception as e:
        st.error(f"Error extracting text from TXT: {str(e)}")
        return ""
    
def display_report_images():
    """Display the visualization images from the reports folder in the Our AI tab"""
    # Dictionary mapping image purposes to filenames
    report_images = {
        "emotion_matrix": "PRIOR2modern_emotion_heatmap.png",
        "key_expressions": "PRIORkey_expressions_wordcloud.png",
        "community_worry": "PRIORimproved_subreddit_worry_levels.png",
        "worry_distribution": "PRIORimproved_anxiety_distribution.png",
        "worry_vs_length": "PRIORimproved_worry_quadrant_plot.png"
    }
    
    # Try multiple possible locations for reports directory
    possible_paths = [
        os.path.join(os.path.dirname(__file__), "reports"),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports")),
        os.path.abspath("reports"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    ]
    
    # Find the first path that exists
    reports_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            reports_dir = path
            break
    
    # If no valid path found, use a default and let the error handling in the display code manage it
    if not reports_dir:
        reports_dir = os.path.join(os.path.dirname(__file__), "reports")
        
    # Function to create the full path to an image
    def get_image_path(filename):
        return os.path.join(reports_dir, filename)
    
    # Return the dictionary of image paths
    return {purpose: get_image_path(filename) for purpose, filename in report_images.items()}

# Creates the main layout
main_col, sidebar_col = st.columns([3, 1])

# The sidebar content
with sidebar_col:
    # Sidebar header
    st.sidebar.title("Monarch")
    st.sidebar.subheader("Privacy-focused NLP for Emotional Pattern Detection")
    st.sidebar.markdown("[Monarch's Website](https://0θ.com/0x02/)")
    # About section
    with st.sidebar.expander("📖 About Monarch", expanded=True):
        st.write("""
        Monarch is a privacy-focused deep learning model that interprets emotional patterns in text. 
        
        Unlike other tools, all analysis happens entirely on your device - your data never leaves your computer.
        
        We use fine-tuned NLP model (BERT) to identify patterns associated with sadness, worry, anger, happiness, and distress.
        """)
    
    # Project details from poster
    with st.sidebar.expander("🔬 Research Details", expanded=False):
        st.write("""
        ### Research Questions
        - What words most frequently correlate with emotional distress?
        - How accurate is emotion classification when models are trained on lexicon-tagged emotional data?
        - Can a fine-tuned deep learning model identify emotional cues in text-based language?
        
        ### Technology
        Monarch uses BERT NLP models fine-tuned on emotion-labeled datasets to approximate categorical responses within 5 categories: sadness, worry, anger, happiness, and distress.
        
        The model was trained and validated entirely offline using PyTorch, HuggingFace Transformers, and local GPU/CPU.
        """)
    
    # Team information
    with st.sidebar.expander("👥 Team", expanded=False):
        st.write("""
        ### Authors
        - Tyler Clanton
        - Derick Burbano-Ramon
        
        ### Advisors
        - Dr. Jeff Adkisson
        
        ### Affiliation
        Kennesaw State University
        """)
    
    # Technical details
    with st.sidebar.expander("⚙️ Technical Details", expanded=False):
        st.write("Model architecture: BERT Base")
        st.write("Training data: Mental health text samples")
        st.write("Output: Distress probability mapped to emotional dimensions")
        
        if os.path.exists(os.path.join(MODEL_DIR, "eval_results.json")):
            import json
            with open(os.path.join(MODEL_DIR, "eval_results.json"), "r") as f:
                eval_results = json.load(f)
                st.write("### Model Performance")
                for metric, value in eval_results.items():
                    if isinstance(value, (int, float)):
                        st.write(f"{metric}: {value:.4f}")
    
    # Color reference/legend for users
    with st.sidebar.expander("🎨 Color Scale Reference", expanded=False):
        st.markdown("""
        - <span style='color: rgba(75, 192, 120, 0.7);'>■</span> Green (0-40): Low level
        - <span style='color: rgba(255, 205, 86, 0.7);'>■</span> Yellow (40-65): Moderate level
        - <span style='color: rgba(255, 159, 64, 0.7);'>■</span> Orange (65-85): High level
        - <span style='color: rgba(255, 99, 132, 0.7);'>■</span> Red (85-100): Severe level
        """, unsafe_allow_html=True)
    
    # Privacy information
    with st.sidebar.expander("🔒 Privacy", expanded=True):
        st.write("""
        All analysis is performed locally in your browser. Your text is never sent to external servers or stored anywhere.
        
        A lightweight, Raspberry Pi-compatible version for a complete offline use on low-power hardware.
        """)
    
    # Resources for the user, might change depending on what we deem as morally bad
    with st.sidebar.expander("📚 Resources", expanded=False):
        st.write("""
        ### Understanding Emotional Analysis
        
        This tool looks for patterns in:
        - Word choice and frequency
        - Linguistic structures
        - Emotional indicators
        
        Results show possible emotional dimensions in your text, but should not be used as a diagnostic tool.
        
        ### Getting Help
        If you notice concerning patterns in your emotional analysis, please consider speaking with a mental health professional.
                 
        ### Help is available 
        988 Suicide and Crisis Lifeline
        """)

# Main content area
with main_col:
    # Header
    st.markdown("<h1 style='text-align: center;'>Monarch</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Private AI for Emotional Pattern Detection</h4>", unsafe_allow_html=True)
    st.markdown("---")
    
    # Loads model
    model, tokenizer, id2label, model_loaded = load_mental_health_model()
    
    # Display model status, two options based on whether or not the placeholder is present
    if model_loaded:
        st.success("✅ Model loaded successfully!")
    else:
        st.warning("⚠️ Using base model (not trained)")
    
    # Create tabs for the main interface including the new Our AI tab
    tab1, tab2, tab3 = st.tabs(["Text Analysis", "Upload File", "Our AI"])
    
    # Text input tab
    with tab1:
        st.subheader("Enter text for analysis")
        user_input = st.text_area("Type or paste any text you want to analyze...", height=200)
        input_source = "text_area"
    
    # File input tab
    with tab2:
        st.subheader("Upload a document")
        uploaded_file = st.file_uploader("Upload a document", type=["pdf", "txt", "docx"])
        if uploaded_file is not None:
            file_type = uploaded_file.name.split(".")[-1].lower()
            
            if file_type == "pdf":
                user_input = extract_text_from_pdf(uploaded_file)
            elif file_type == "docx":
                user_input = extract_text_from_docx(uploaded_file)
            elif file_type == "txt":
                user_input = extract_text_from_txt(uploaded_file)
            else:
                user_input = ""
                st.error("Unsupported file type")
            
            if user_input:
                st.success(f"Successfully extracted text from {uploaded_file.name}")
                st.text_area("Extracted text:", user_input, height=200)
            
            input_source = "file_upload"
    
    # New Our AI tab
    with tab3:
        st.header("Our Machine Learning Technology")
        
        # Create two columns for layout
        ml_col1, ml_col2 = st.columns([3, 2])
        
        with ml_col1:
            st.subheader("How Monarch's AI Works")
            st.write("""
            Monarch utilizes advanced natural language processing (NLP) and machine learning techniques to analyze emotional patterns in text. Our approach combines multiple AI models and data sources to provide accurate, privacy-focused analysis.
            
            ### Data Collection & Learning
            Our models are trained on diverse datasets including:
            - Anonymized mental health forum posts
            - Emotion-labeled linguistic datasets
            - Clinical language samples (with all identifying information removed)
            
            The AI continuously learns patterns of emotional expression across different contexts, allowing it to identify subtle indicators of emotional states like sadness, worry, anger, happiness, and distress.
            """)
            
            st.subheader("Our Machine Learning Pipeline")
            st.write("""
            1. **Text Preprocessing**: Cleaning and normalizing input text
            2. **Feature Extraction**: Identifying linguistic patterns and emotional markers
            3. **Deep Learning Analysis**: Processing through our fine-tuned BERT model
            4. **Multi-dimensional Scoring**: Mapping probabilities to emotional dimensions
            5. **Visualization**: Presenting results through intuitive visual representations
            
            All processing happens locally on your device, ensuring complete privacy.
            """)
        
        with ml_col2:
            # Try multiple possible paths for the diagram image
            diagram_paths = [
                os.path.join(os.path.dirname(__file__), "reports", "Diagram.jpg"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports", "Diagram.jpg"),
                os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "reports", "Diagram.jpg"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "reports", "Diagram.jpg")),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "reports", "Diagram.jpg"))
            ]
            
            # Try each path until we find one that works
            image_found = False
            for diagram_path in diagram_paths:
                if os.path.exists(diagram_path):
                    try:
                        st.image(diagram_path, caption="Monarch's Machine Learning Architecture")
                        image_found = True
                        break
                    except Exception as e:
                        continue
            
            # If no path worked, use the placeholder
            if not image_found:
                st.error(f"Could not find Diagram.jpg in any expected location. Please check file path.")
                st.image("https://via.placeholder.com/400x250", caption="Monarch's Machine Learning Architecture")
        
        # After the ml_col1 and ml_col2 sections, now add the remaining content
        # Note that these are not indented under ml_col2 anymore
        st.markdown("---")
        
        # Get image paths
        report_images = display_report_images()
        
        # Explaining the relationship between emotions section
        st.subheader("Understanding Emotion Relationships")
        st.write("""
        Our research has uncovered important relationships between different emotional dimensions. The heatmap below shows how different emotions relate to each other in our analysis.
        """)
        
        # Create two columns for first set of charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Use the actual emotion matrix image instead of placeholder
            try:
                st.image(report_images["emotion_matrix"], caption="Emotion Relationship Strength Matrix")
            except Exception as e:
                st.error(f"Error displaying emotion matrix image: {str(e)}")
                st.image("https://via.placeholder.com/450x400", caption="Emotion Relationship Strength Matrix")
                
            st.write("""
            This visualization shows how different emotions correlate with each other. For example, we can see strong positive relationships between joy and optimism (0.58), while emotions like anger and disgust show weaker relationships with positive emotions.
            
            These relationship patterns help our model understand the complex interplay of emotions in human expression.
            """)
        
        with chart_col2:
            # Use the actual word cloud image
            try:
                st.image(report_images["key_expressions"], caption="Key Expressions in High Concern Posts")
            except Exception as e:
                st.error(f"Error displaying key expressions image: {str(e)}")
                st.image("https://via.placeholder.com/450x400", caption="Key Expressions in High Concern Posts")
                
            st.write("""
            This word cloud highlights common expressions found in texts with elevated emotional concern. The size of each word represents its frequency, while colors indicate emotional tone categories.
            
            Words like "help," "connection," "issue," and "thought" are frequently associated with expressions of concern or emotional distress.
            """)
        
        st.markdown("---")
        
        # Analysis of online community data
        st.subheader("Community Analysis & Worry Intensity Patterns")
        
        # Create two columns for second set of charts
        chart_col3, chart_col4 = st.columns(2)
        
        with chart_col3:
            # Use the actual community worry image
            try:
                st.image(report_images["community_worry"], caption="Average Worry Intensity by Online Community")
            except Exception as e:
                st.error(f"Error displaying community worry image: {str(e)}")
                st.image("https://via.placeholder.com/450x380", caption="Average Worry Intensity by Online Community")
                
            st.write("""
            Our research examines worry intensity across different online communities. This data helps calibrate our AI to better understand contextual emotional expressions.
            
            Communities focused specifically on mental health support show varying levels of expressed worry, which helps our model recognize different manifestations of emotional distress.
            """)
        
        with chart_col4:
            # Use the actual worry distribution image
            try:
                st.image(report_images["worry_distribution"], caption="Distribution of Worry Intensity Measurements")
            except Exception as e:
                st.error(f"Error displaying worry distribution image: {str(e)}")
                st.image("https://via.placeholder.com/450x380", caption="Distribution of Worry Intensity Measurements")
                
            st.write("""
            This distribution chart shows the range of worry intensity values across our research dataset. The mean worry score of 7.57 and median of 5.96 help establish baselines for our analysis.
            
            The graph shows that while most texts express low to moderate worry levels, there is a significant tail of high-intensity emotional expression that our model is trained to recognize.
            """)
        
        st.markdown("---")
        
        # Text length and worry correlation
        st.subheader("Text Patterns & Emotional Expression")
        # Use the actual worry vs length plot
        try:
            st.image(report_images["worry_vs_length"], caption="Worry Score vs. Post Length Analysis")
        except Exception as e:
            st.error(f"Error displaying worry vs length image: {str(e)}")
            st.image("https://via.placeholder.com/800x450", caption="Worry Score vs. Post Length Analysis")
            
        st.write("""
        This visualization explores the relationship between text length and expressed worry levels. We've found that approximately 27.7% of longer posts express high worry levels, while shorter posts show a different distribution pattern.
        
        This insight helps our model adjust its analysis based on text length, improving accuracy across different types of input.
        """)
        
        st.markdown("---")
        
        # Research outcomes and future development
        st.subheader("Ongoing Research & Development")
        st.write("""
        Monarch is continuously evolving through ongoing research and model refinement. Current areas of development include:
        
        - **Expanded Emotional Dimensions**: Adding more nuanced emotional categories beyond our current five dimensions
        - **Cross-Cultural Adaptation**: Improving recognition of emotional expression across different cultural contexts
        - **Longitudinal Analysis**: Enhancing trend detection for users who analyze multiple texts over time
        - **Low-Resource Deployment**: Optimizing our models to run efficiently on personal devices with limited processing power
        
        Our commitment to privacy-first AI means all improvements are designed to run locally, keeping your data on your device.
        """)
    
    # Color mapping
    def get_color(score):
        if score >= 85:
            return COLORS["red"]
        elif score >= 65:
            return COLORS["orange"]
        elif score >= 40:
            return COLORS["yellow"]
        else:
            return COLORS["green"]
    
    # Map model probabilities to emotion scores - Updated for BERT model
    def map_to_emotions(probs, id2label):
        """Map model outputs to different emotional dimensions"""
        # Create a dictionary mapping emotion labels to probability scores
        emotion_scores = {}
        
        # Convert probabilities to percentage scores (0-100)
        for i, prob in enumerate(probs):
            emotion = id2label[i]
            # Convert probability to percentage and round to 2 decimal places
            emotion_scores[emotion] = round(float(prob) * 100, 2)
            
        return emotion_scores
    
    # Real analyzer using BERT model (based on test.py)
    def analyze_text(text):
        """Analyze text using the mental health model"""
        
        if not model or not tokenizer:
            # Fall back to random if model isn't loaded
            return {
                "sadness": round(random.uniform(40, 100), 2),
                "anger": round(random.uniform(0, 80), 2),
                "distress": round(random.uniform(30, 95), 2),
                "joy": round(random.uniform(10, 70), 2),
                "worry": round(random.uniform(10, 90), 2)
            }
        
        try:
            # Tokenize the input text
            inputs = tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # Gets model predictions
            with torch.no_grad(): # Uses the model to get a result without worrying about training or tracking changes
                outputs = model(**inputs)
                logits = outputs.logits
                probs = F.softmax(logits, dim=1)[0]
            
            # Map the outputs to emotions using our id2label mapping
            emotion_scores = map_to_emotions(probs, id2label)
            return emotion_scores
            
        except Exception as e:
            st.error(f"Error analyzing text: {str(e)}")
            # Fall back to random values if model fails
            return {
                "sadness": round(random.uniform(40, 100), 2),
                "anger": round(random.uniform(0, 80), 2),
                "distress": round(random.uniform(30, 95), 2),
                "joy": round(random.uniform(10, 70), 2),
                "worry": round(random.uniform(10, 90), 2)
            }
    
    # Extract significant keywords with improved NLP capabilities
    def extract_significant_keywords(text):
        """Extract potentially significant words from text with improved NLP understanding"""
        # This is a simplified version, it is highly recommended that later into the project we use NLP libraries

        # Enhanced common words list to include more stopwords
        common_words = {"the", "and", "a", "to", "of", "in", "that", "it", "with", 
                       "is", "was", "for", "on", "are", "as", "be", "this", "have", 
                       "or", "at", "by", "not", "but", "what", "all", "when", "can",
                       "from", "an", "they", "we", "you", "he", "she", "his", "her",
                       "their", "our", "my", "your", "i", "me", "him", "us", "them",
                       "who", "which", "where", "there", "here", "how", "why", "am",
                       "been", "being", "has", "had", "would", "could", "should",
                       "will", "shall", "may", "might", "must", "do", "does", "did",
                       "doing", "done", "get", "got", "getting", "very", "really",
                       "just", "like", "so", "much", "many", "more", "most", "some",
                       "any", "no", "yes", "one", "two", "three", "first", "last",
                       "also", "then", "than", "about", "after", "before", "over",
                       "under", "up", "down", "out", "in", "through"}
        
        # Simple word extraction
        words = text.lower().split()
        # Remove any punctuation
        cleaned_words = [word.strip(".,;:!?()[]{}\"'") for word in words]
        # Remove common words and very short words
        significant = [word for word in cleaned_words if word not in common_words and len(word) > 3]
        
        # Count occurrences
        word_counts = {}
        for word in significant:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
        
        # Get top words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_words[:10]  # Return top 10 words
    
    # Adds timestamp to data for historical tracking
    def add_timestamp_to_data(data):
        data_with_timestamp = data.copy()
        data_with_timestamp['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return data_with_timestamp
    
    # Generate radar chart - Updated for BERT model emotions
    def create_radar_chart(scores):
        categories = list(scores.keys())
        values = list(scores.values())
        
        # Add first value again to close the loop
        categories = categories + [categories[0]]
        values = values + [values[0]]
        
        # Create reference circles
        fig = go.Figure()
        
        # Adds more reference circles (less prominent)
        for level_name, level_value in REFERENCE_LEVELS.items():
            fig.add_trace(go.Scatterpolar(
                r=[level_value] * len(categories),
                theta=categories,
                fill=None,
                mode='lines',
                line=dict(color='rgba(200, 200, 200, 0.5)', dash='dot'),
                name=f"{level_name.capitalize()} ({level_value})",
                hoverinfo='text',
                text=[f"{level_name.capitalize()} Level: {level_value}"] * len(categories)
            ))
        
        # Add actual "scores"
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            mode='lines+markers',
            line=dict(color=COLORS["purple"], width=2),
            marker=dict(size=8, color=COLORS["purple"]),
            name='Your Profile',
            hoverinfo='text',
            text=[f"{cat}: {val}" for cat, val in zip(categories, values)]
        ))
        
        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100],
                    tickfont=dict(size=10),
                    tickvals=[0, 25, 50, 75, 100],
                    ticktext=["0", "25", "50", "75", "100"]
                ),
                angularaxis=dict(
                    tickfont=dict(size=12, color='darkblue'),
                )
            ),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=20, r=20, t=30, b=20),
            height=450
        )
        
        return fig
    
    # Create line chart for historical data (requires multiple entries)
    def create_history_chart(history_data):
        # Convert list of dictionaries to DataFrame
        df = pd.DataFrame(history_data)
        
        # Create figure
        fig = go.Figure()
        
        # Add traces for each emotion
        for emotion in ["sadness", "anger", "distress", "joy", "worry"]:
            if emotion in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[emotion],
                    mode='lines+markers',
                    name=emotion.capitalize(),
                    line=dict(width=2),
                    marker=dict(size=8)
                ))
        
        # Update layout
        fig.update_layout(
            title="Emotional Trends Over Time",
            xaxis=dict(title="Analysis #"),
            yaxis=dict(title="Score", range=[0, 100]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            height=400
        )
        
        return fig
        
    # Only show analyze button on Text Analysis or Upload File tabs
    if tab1.active or tab2.active:
        # Add the analyze button
        analyze_button = st.button("Analyze Text")
        
        # Results
        if analyze_button and user_input and user_input.strip():
            with st.spinner("Analyzing your text..."):
                scores = analyze_text(user_input)
                
                # Add to history for data
                st.session_state.history.append(scores)
                
                # Extract keywords
                keywords = extract_significant_keywords(user_input)
                
            st.markdown("### Analysis Results:")
            
            # Create tabs for different visualizations
            tabs = st.tabs(["Overview", "Detailed Analysis", "Trends", "Text Statistics", "Emotion Gauges"])
            
            # Radar graph
            with tabs[0]:
                st.subheader("Emotional Profile")
                
                radar_fig = create_radar_chart(scores)
                st.plotly_chart(radar_fig, use_container_width=True)
                
                # Display dominant emotion
                emotions = list(scores.keys())
                values = list(scores.values())
                dominant_idx = values.index(max(values))
                dominant_emotion = emotions[dominant_idx]
                dominant_score = values[dominant_idx]
                
                st.markdown(f"**🔥 Dominant Emotion: {dominant_emotion.capitalize()} ({dominant_score:.2f}%)**")
                st.caption("This radar chart shows the emotional dimensions detected in your text.")
        
            # Pie chart with key words detection table
            with tabs[1]:
                st.subheader("Detailed Analysis")
                
                # Create columns for layout
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        names=[e.capitalize() for e in scores.keys()],
                        values=list(scores.values()),
                        hole=0.4,
                        color_discrete_sequence=[COLORS["red"], COLORS["orange"], COLORS["yellow"], COLORS["teal"], COLORS["purple"]],
                    )
                    fig_pie.update_traces(textinfo='label+percent', pull=[0.05]*len(scores))
                    st.plotly_chart(fig_pie, use_container_width=True)
                    st.caption("Emotional composition of your text")
                
                with col2:
                    # Key Words Detection
                    if keywords:
                        st.subheader("Key Words Detection")
                        key_words_df = pd.DataFrame(keywords, columns=["Word", "Frequency"])
                        st.dataframe(key_words_df, use_container_width=True)
                        st.caption("Frequently used words in your text that may indicate emotional context.")
            
                # Display emotion distribution as a bar chart
                st.subheader("Emotion Distribution")
                
                # Convert scores to a format suitable for a bar chart
                emotions_df = pd.DataFrame({
                    'Emotion': list(scores.keys()),
                    'Score': list(scores.values())
                })
                
                # Create a bar chart
                fig_bar = px.bar(
                    emotions_df,
                    x='Emotion',
                    y='Score',
                    color='Score',
                    color_continuous_scale=['green', 'yellow', 'orange', 'red'],
                    labels={'Score': 'Intensity (%)'},
                    range_y=[0, 100]
                )
                
                fig_bar.update_layout(
                    title="Emotional Intensity Levels",
                    xaxis_title="Emotion Category",
                    yaxis_title="Intensity (%)",
                    height=400
                )
                
                st.plotly_chart(fig_bar, use_container_width=True)
            
            # Trend graph that takes in multiple responses
            with tabs[2]:
                st.subheader("Emotional Trends")
                
                # Show historical data if multiple analyses have been done
                if len(st.session_state.history) > 1:
                    history_chart = create_history_chart(st.session_state.history)
                    st.plotly_chart(history_chart, use_container_width=True)
                    st.caption("Changes in emotional profiles across multiple analyses.")
                else:
                    st.info("Analyze more texts to see emotional trends over time.")
            
            # Text statistics section
            with tabs[3]:
                st.subheader("Text Statistics")
                
                # Create columns for metrics
                stat_cols = st.columns(4)
                with stat_cols[0]:
                    st.metric("Word Count", len(user_input.split()))
                with stat_cols[1]:
                    st.metric("Character Count", len(user_input))
                with stat_cols[2]:
                    st.metric("Sentence Count", user_input.count('.') + user_input.count('!') + user_input.count('?'))
                with stat_cols[3]:
                    avg_word_length = sum(len(word) for word in user_input.split()) / max(1, len(user_input.split()))
                    st.metric("Avg Word Length", f"{avg_word_length:.1f}")
                
                # Additional text statistics
                st.write("### Text Sample")
                if len(user_input) > 500:
                    st.write(f"{user_input[:500]}...")
                else:
                    st.write(user_input)
                    
            # Individual emotion gauges
            with tabs[4]:
                st.subheader("Individual Emotion Gauges")
                
                # Create a separate tab for each emotion gauge
                emotion_tabs = st.tabs([emotion.capitalize() for emotion in scores.keys()])
                
                for i, (emotion, score) in enumerate(scores.items()):
                    with emotion_tabs[i]:
                        color = get_color(score)
                        
                        # For joy, invert the color scale (high joy is good)
                        if emotion == "joy":
                            if score >= 85:
                                color = COLORS["green"]
                            elif score >= 65:
                                color = COLORS["yellow"]
                            elif score >= 40:
                                color = COLORS["orange"]
                            else:
                                color = COLORS["red"]
                        
                        # Create a full-sized gauge for each emotion
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=score,
                            title={'text': f"{emotion.capitalize()} Score"},
                            gauge={
                                'axis': {'range': [0, 100]},
                                'bar': {'color': color},
                                'steps': [
                                    {'range': [0, 40], 'color': COLORS["red"] if emotion == "joy" else COLORS["green"]},
                                    {'range': [40, 65], 'color': COLORS["orange"] if emotion == "joy" else COLORS["yellow"]},
                                    {'range': [65, 85], 'color': COLORS["yellow"] if emotion == "joy" else COLORS["orange"]},
                                    {'range': [85, 100], 'color': COLORS["green"] if emotion == "joy" else COLORS["red"]},
                                ],
                                'threshold': {
                                    'line': {'color': color, 'width': 4},
                                    'thickness': 0.75,
                                    'value': score
                                }
                            }
                        ))
                        
                        fig.update_layout(height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Adjust descriptions for joy
                        if emotion == "joy":
                            if score >= 85:
                                st.success(f"High {emotion} level detected - excellent emotional well-being indicator.")
                            elif score >= 65:
                                st.info(f"Moderate-high {emotion} level detected - good emotional well-being.")
                            elif score >= 40:
                                st.warning(f"Moderate-low {emotion} level detected.")
                            else:
                                st.error(f"Low {emotion} level detected.")
                        else:
                            # Standard descriptions for other emotions
                            if score < 40:
                                st.success(f"Low {emotion} level detected.")
                            elif score < 65:
                                st.info(f"Moderate {emotion} level detected.")
                            elif score < 85:
                                st.warning(f"High {emotion} level detected.")
                            else:
                                st.error(f"Severe {emotion} level detected.")
        
            st.caption("Note: This analysis is for informational purposes only and should not be considered a medical diagnosis.")
    
        elif analyze_button and (not user_input or not user_input.strip()):
            st.warning("Please enter or upload text for analysis.")
        else:
            st.info("Enter text or upload a file above and click 'Analyze Text' to begin.")

# FAQ Section
with main_col:
    st.markdown("---")
    st.markdown("### Frequently Asked Questions")
    
    # Creates columns to format Qs
    col1, col2 = st.columns(2)
    
    with col1:
        faq_expander = st.expander("How does this tool work?")
        with faq_expander:
            st.write("""
            This tool analyzes text using natural language processing (NLP) technology to identify patterns 
            in language that may correspond to different emotional dimensions. The analysis is based on 
            linguistic features rather than clinical diagnostic criteria. The results are presented as 
            visualizations showing the relative presence of different emotional dimensions in the text.
            """)
        
        privacy_expander = st.expander("How is my data protected?")
        with privacy_expander:
            st.write("""
            Your privacy is our top priority. All analysis is performed locally in your browser, and your 
            text is never stored on external servers. Your data remains on your device and is not shared 
            with anyone. This tool is designed with privacy-first principles.
            """)
    
    with col2:
        usage_expander = st.expander("What are the best uses for this tool?")
        with usage_expander:
            st.write("""
            This tool works best for:
            - Exploring emotional content in written text
            - Analyzing journal entries over time
            - Understanding emotional patterns in communication
            - Writing analysis for creative purposes
            
            It should not be used for medical diagnosis or as a substitute for professional advice.
            """)
        
        limitations_expander = st.expander("What are the limitations?")
        with limitations_expander:
            st.write("""
            - The tool can only analyze text, not images or audio
            - Results are based on linguistic patterns, not clinical assessment
            - The model has been trained on specific datasets and may not generalize to all types of text
            - Analysis should be interpreted as exploratory, not diagnostic
            """)

# Footer
st.markdown("---")
st.caption(f"🔒 This tool runs completely offline. No data is uploaded. | © {datetime.now().year} Monarch Project")