"""
Functions for loading and preprocessing document data.
"""
import os
import re
import pandas as pd
from docx import Document
import jieba
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK data (uncomment if needed)
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

def extract_text_from_docx(file_path):
    """Extract text content from a .docx file."""
    try:
        doc = Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            if para.text.strip():  # Skip empty paragraphs
                full_text.append(para.text)
        return '\n'.join(full_text)
    except Exception as e:
        print(f"Error extracting text from {file_path}: {e}")
        return ""

def load_data(directory_path, language="chinese", subdirs=None):
    """
    Load documents from specified directories.
    
    Args:
        directory_path: Path to the directory containing documents
        language: 'chinese' or 'english'
        subdirs: List of subdirectories (for English mode)
        
    Returns:
        pandas.DataFrame with document filenames and text
    """
    essays = []
    
    if language == "english" and subdirs:
        # Process multiple subdirectories for English
        for subdir in subdirs:
            full_path = os.path.join(directory_path, subdir)
            _process_directory(full_path, essays)
    elif language == 'chinese':
        # Process single directory (for Chinese or if no subdirs specified)
        _process_directory(directory_path, essays)
    
    # Create DataFrame
    df = pd.DataFrame(essays)
    print(f"Loaded {len(df)} {language} documents")
    return df

def _process_directory(directory, essays_list):
    """Process documents in a directory and add them to the essays list."""
    for file in os.listdir(directory):
        if file.endswith('.docx') and not file.startswith('~$'):
            try:
                file_path = os.path.join(directory, file)
                text = extract_text_from_docx(file_path)
                essays_list.append({'filename': file, 'text': text})
            except Exception as e:
                print(f"Error processing {file}: {e}")

#! This function is not called
def preprocess_data(text, language="chinese"):
    """
    Preprocess text based on language.
    
    Args:
        text: Input text
        language: 'chinese' or 'english'
        
    Returns:
        list of tokens/words
    """
    if language == "chinese":
        return preprocess_chinese(text)
    elif language == "english":
        return preprocess_english(text)

def preprocess_chinese(text):
    """Preprocess Chinese text for word embeddings."""
    # Check for empty or non-string input
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Common Chinese stopwords
    stopwords = {'的', '了', '和', '是', '就', '都', '而', '及', '与', '着',
                '或', '一个', '没有', '我们', '你们', '他们', '她们', '这个',
                '那个', '这些', '那些', '不', '在', '人', '上', '来', '到'}
    
    # Remove punctuation and segment with jieba
    text = re.sub(r'[^\w\s\u4e00-\u9fff]', '', text)
    words = list(jieba.cut(text))
    
    # Remove stopwords and empty strings
    words = [word for word in words if word.strip() and word not in stopwords]
    
    return words

def preprocess_english(text):
    """Preprocess English text for word embeddings."""
    # Check for empty input
    if not isinstance(text, str) or not text.strip():
        return []
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Tokenize, lowercase, and remove punctuation
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = word_tokenize(text)
    
    # Remove stopwords and empty strings
    words = [word for word in words if word.strip() and word not in stop_words]
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return words