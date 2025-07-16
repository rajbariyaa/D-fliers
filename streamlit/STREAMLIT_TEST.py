import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
import warnings
import os
import sys
from pathlib import Path
import time
import traceback

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuration
GITHUB_REPO = "rajbariyaa/D-fliers"
RELEASE_TAG = "123"
DATA_FILENAME = "merged_flights_weather.csv"
LOCAL_DATA_DIR = Path("data")
LOCAL_DATA_FILE = LOCAL_DATA_DIR / DATA_FILENAME

# Create data directory if it doesn't exist
try:
    LOCAL_DATA_DIR.mkdir(exist_ok=True)
except Exception as e:
    st.error(f"Could not create data directory: {e}")


def download_with_direct_url(repo, tag, filename, local_path):
    """
    Download using direct GitHub URL instead of API (avoids rate limits)
    """
    try:
        # Construct direct download URL
        # Format: https://github.com/user/repo/releases/download/tag/filename
        direct_url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
        
        st.info(f"Downloading {filename} from direct URL...")
        st.info(f"URL: {direct_url}")
        
        # Try to download with direct URL
        response = requests.get(direct_url, stream=True, timeout=120)
        
        if response.status_code == 404:
            return False, f"File not found at {direct_url}. Please check:\n1. Release tag '{tag}' exists\n2. File '{filename}' is uploaded to that release\n3. Release is published (not draft)"
        
        response.raise_for_status()
        
        # Get file size from headers if available
        file_size = response.headers.get('content-length')
        if file_size:
            file_size = int(file_size)
            st.info(f"File size: {file_size / (1024*1024):.1f} MB")
        
        # Download with progress
        downloaded_size = 0
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if file_size:
                        progress = min(downloaded_size / file_size, 1.0)
                        progress_bar.progress(progress)
                        status_text.text(f"Downloaded: {downloaded_size / (1024*1024):.1f} MB")
        
        progress_bar.progress(1.0)
        status_text.text("Download completed!")
        return True, "Download successful"
        
    except requests.exceptions.RequestException as e:
        return False, f"Network error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"


def download_with_github_token(repo, tag, filename, local_path, token=None):
    """
    Download using GitHub API with authentication token
    """
    try:
        headers = {}
        if token:
            headers['Authorization'] = f'token {token}'
            headers['Accept'] = 'application/vnd.github.v3+json'
        
        # Get release information
        api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
        
        response = requests.get(api_url, headers=headers, timeout=30)
        
        if response.status_code == 404:
            return False, f"Release '{tag}' not found in repository '{repo}'"
        elif response.status_code == 403:
            return False, "API rate limit exceeded. Please provide a GitHub token or try direct download."
        
        response.raise_for_status()
        release_data = response.json()
        
        # Find the asset
        download_url = None
        file_size = None
        
        for asset in release_data.get('assets', []):
            if asset['name'] == filename:
                download_url = asset['browser_download_url']
                file_size = asset['size']
                break
        
        if not download_url:
            return False, f"File '{filename}' not found in release assets"
        
        # Download the file
        file_response = requests.get(download_url, stream=True, headers=headers, timeout=120)
        file_response.raise_for_status()
        
        downloaded_size = 0
        progress_bar = st.progress(0)
        
        with open(local_path, 'wb') as f:
            for chunk in file_response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded_size += len(chunk)
                    
                    if file_size:
                        progress = min(downloaded_size / file_size, 1.0)
                        progress_bar.progress(progress)
        
        progress_bar.progress(1.0)
        return True, "Download successful"
        
    except Exception as e:
        return False, f"Error: {str(e)}"


def safe_download_file_from_github_release(repo, tag, filename, local_path, github_token=None):
    """
    Try multiple download methods to avoid rate limits
    """
    st.info(f"Attempting to download {filename} from {repo}/releases/tag/{tag}")
    
    # Method 1: Direct URL (fastest, avoids API rate limits)
    st.info("🔄 Trying direct download URL...")
    success, message = download_with_direct_url(repo, tag, filename, local_path)
    if success:
        return True, message
    else:
        st.warning(f"Direct download failed: {message}")
    
    # Method 2: GitHub API with token (if provided)
    if github_token:
        st.info("🔄 Trying GitHub API with token...")
        success, message = download_with_github_token(repo, tag, filename, local_path, github_token)
        if success:
            return True, message
        else:
            st.warning(f"API download with token failed: {message}")
    
    # Method 3: GitHub API without token (last resort)
    st.info("🔄 Trying GitHub API without token...")
    success, message = download_with_github_token(repo, tag, filename, local_path, None)
    if success:
        return True, message
    else:
        st.error(f"All download methods failed. Last error: {message}")
    
    return False, message


def load_model_safely():
    """Load the saved model with error handling"""
    try:
        model_paths = [
            'models/test02.pkl',
            'test02.pkl',
            './models/test02.pkl',
            './test02.pkl'
        ]
        
        for path in model_paths:
            if os.path.exists(path):
                st.info(f"Found model at: {path}")
                with open(path, 'rb') as f:
                    model_package = pickle.load(f)
                
                return (
                    model_package.get('model'),
                    model_package.get('encoders', {}),
                    model_package.get('feature_columns', []),
                    model_package.get('created_date', 'Unknown'),
                    model_package.get('version', 'Unknown')
                )
        
        st.error("Model file not found. Please ensure test02.pkl is in the repository.")
        return None, None, None, None, None

    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None


def load_historical_data_safely():
    """
    Load historical data with multiple download methods
    """
    try:
        # Check if file exists locally
        if LOCAL_DATA_FILE.exists():
            try:
                st.info("Checking cached data...")
                df_test = pd.read_csv(LOCAL_DATA_FILE, nrows=5)
                df = pd.read_csv(LOCAL_DATA_FILE)
                st.success(f"✅ Loaded cached data: {len(df):,} rows")
                return df
            except Exception as e:
                st.warning(f"Cached file appears corrupted: {str(e)}")
                try:
                    LOCAL_DATA_FILE.unlink()
                except:
                    pass
        
        # Get GitHub token from environment or user input
        github_token = os.environ.get('GITHUB_TOKEN')
        
        if not github_token:
            st.info("💡 For better download reliability, you can provide a GitHub token")
            github_token = st.text_input(
                "GitHub Token (optional)", 
                type="password",
                help="Generate at https://github.com/settings/tokens"
            )
        
        # Download from GitHub
        st.info(f"📥 Downloading {DATA_FILENAME} from GitHub releases...")
        
        success, message = safe_download_file_from_github_release(
            GITHUB_REPO, 
            RELEASE_TAG, 
            DATA_FILENAME, 
            LOCAL_DATA_FILE,
            github_token if github_token else None
        )
        
        if success:
            try:
                df = pd.read_csv(LOCAL_DATA_FILE)
                st.success(f"✅ Downloaded and loaded data: {len(df):,} rows")
                return df
            except Exception as e:
                st.error(f"❌ Error reading downloaded file: {str(e)}")
        else:
            st.error(f"❌ Failed to download data: {message}")
            
            # Show troubleshooting info
            st.markdown("""
            ### 🔧 Troubleshooting Steps:
            
            1. **Check your release**: Go to https://github.com/rajbariyaa/D-fliers/releases/tag/123
            2. **Verify file exists**: Make sure `merged_flights_weather.csv` is uploaded to that release
            3. **Check if release is published**: Draft releases are not accessible
            4. **Try with GitHub token**: Create a token at https://github.com/settings/tokens
            5. **Alternative**: Use direct link or smaller file
            """)
            
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()


def main():
    """Main application"""
    try:
        st.markdown("# 🛫 Flight Delay Predictor")
        
        # Show current configuration
        with st.expander("📋 Configuration"):
            st.code(f"""
Repository: {GITHUB_REPO}
Release Tag: {RELEASE_TAG}
Data File: {DATA_FILENAME}
Direct URL: https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{DATA_FILENAME}
            """)
        
        # Load model
        model, encoders, feature_columns, created_date, version = load_model_safely()
        
        if model is None:
            st.error("Could not load the model.")
            return
        
        # Load data
        df = load_historical_data_safely()
        
        # Show status
        col1, col2 = st.columns(2)
        with col1:
            if model:
                st.success("✅ Model loaded")
            else:
                st.error("❌ Model failed")
        
        with col2:
            if not df.empty:
                st.success(f"✅ Data loaded ({len(df):,} rows)")
            else:
                st.warning("⚠️ No data (app will work in limited mode)")
        
        # Simple interface for testing
        st.markdown("### Quick Test")
        if st.button("Test Prediction"):
            st.info("Model is ready for predictions!")
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
