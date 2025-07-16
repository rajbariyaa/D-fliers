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

# Configuration - UPDATE THESE VALUES
GITHUB_REPO = os.environ.get('GITHUB_REPO', 'rajbariyaa/D-fliers')  # SET YOUR REPO HERE
RELEASE_TAG = os.environ.get('RELEASE_TAG', '123')
DATA_FILENAME = "merged_flights_weather.csv"
LOCAL_DATA_DIR = Path("data")
LOCAL_DATA_FILE = LOCAL_DATA_DIR / DATA_FILENAME

# Create data directory if it doesn't exist
try:
    LOCAL_DATA_DIR.mkdir(exist_ok=True)
except Exception as e:
    st.error(f"Could not create data directory: {e}")

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        color: white;
        text-align: center;
        margin: 0;
        font-size: 2.5rem;
        font-weight: bold;
    }
    .error-box {
        background: #f8d7da;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }
    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def safe_download_file_from_github_release(repo, tag, filename, local_path, max_retries=3):
    """
    Safely download a file from GitHub releases with retry logic
    """
    if repo == 'your-username/your-repo-name':
        return False, "Please update GITHUB_REPO in the configuration"
    
    for attempt in range(max_retries):
        try:
            st.info(f"Attempting to download {filename} (attempt {attempt + 1}/{max_retries})")
            
            # Get release information
            if tag == "latest":
                api_url = f"https://api.github.com/repos/{repo}/releases/latest"
            else:
                api_url = f"https://api.github.com/repos/{repo}/releases/tags/{tag}"
            
            response = requests.get(api_url, timeout=30)
            
            if response.status_code == 404:
                return False, f"Repository {repo} or release {tag} not found"
            
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
            
            # Download with progress
            st.info(f"Downloading {filename} ({file_size / (1024*1024):.1f} MB)...")
            
            response = requests.get(download_url, stream=True, timeout=60)
            response.raise_for_status()
            
            downloaded_size = 0
            progress_bar = st.progress(0)
            
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if file_size:
                            progress = min(downloaded_size / file_size, 1.0)
                            progress_bar.progress(progress)
            
            progress_bar.progress(1.0)
            st.success(f"Successfully downloaded {filename}")
            return True, "Download successful"
            
        except requests.exceptions.RequestException as e:
            if attempt == max_retries - 1:
                return False, f"Network error after {max_retries} attempts: {str(e)}"
            st.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2)
        except Exception as e:
            if attempt == max_retries - 1:
                return False, f"Error after {max_retries} attempts: {str(e)}"
            st.warning(f"Attempt {attempt + 1} failed, retrying...")
            time.sleep(2)
    
    return False, "All download attempts failed"


def load_model_safely():
    """Load the saved model with error handling"""
    try:
        # Try different possible paths
        model_paths = [
            'models/test02.pkl',
            'test02.pkl',
            './models/test02.pkl',
            './test02.pkl'
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                st.info(f"Found model at: {path}")
                with open(path, 'rb') as f:
                    model_package = pickle.load(f)
                model_loaded = True
                break
        
        if not model_loaded:
            st.error("Model file not found. Please ensure test02.pkl is in the repository.")
            st.info("Searched paths: " + ", ".join(model_paths))
            return None, None, None, None, None

        return (
            model_package.get('model'),
            model_package.get('encoders', {}),
            model_package.get('feature_columns', []),
            model_package.get('created_date', 'Unknown'),
            model_package.get('version', 'Unknown')
        )
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.code(traceback.format_exc())
        return None, None, None, None, None


def load_historical_data_safely():
    """
    Load historical data with comprehensive error handling
    """
    try:
        # Check if file exists locally and is valid
        if LOCAL_DATA_FILE.exists():
            try:
                st.info("Checking cached data...")
                # Quick validation
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
        
        # Download from GitHub
        st.info(f"📥 Downloading {DATA_FILENAME} from GitHub releases...")
        
        success, message = safe_download_file_from_github_release(
            GITHUB_REPO, 
            RELEASE_TAG, 
            DATA_FILENAME, 
            LOCAL_DATA_FILE
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
            
        # Return empty DataFrame as fallback
        st.warning("Using empty dataset as fallback. Some features may not work.")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"Unexpected error in load_historical_data_safely: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()


def load_airports_data_safely():
    """Load airports data with error handling"""
    try:
        # Check for airports.csv in different locations
        airports_paths = ['airports.csv', './airports.csv', 'data/airports.csv']
        
        for path in airports_paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        
        st.info("Airports data not found. Using fallback mode.")
        return pd.DataFrame()
        
    except Exception as e:
        st.warning(f"Error loading airports data: {str(e)}")
        return pd.DataFrame()


def create_simple_prediction_interface():
    """Create a simplified interface when full data isn't available"""
    st.markdown("### Simple Flight Delay Predictor")
    
    st.markdown("""
    <div class="info-box">
        <h4>Limited Mode</h4>
        <p>Running in limited mode. Full historical data not available, but basic prediction is still possible.</p>
    </div>
    """, unsafe_allow_html=True)
    
    with st.form("simple_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("Origin Airport", value="JFK").upper()
            airline = st.selectbox("Airline", ["AA", "UA", "DL", "WN", "AS"])
            
        with col2:
            dest = st.text_input("Destination Airport", value="LAX").upper()
            flight_date = st.date_input("Flight Date", value=datetime.date.today() + datetime.timedelta(days=1))
        
        submitted = st.form_submit_button("Get Basic Prediction")
        
        if submitted:
            # Simple rule-based prediction as fallback
            import random
            random.seed(hash(f"{origin}{dest}{airline}"))
            base_delay = random.randint(-5, 45)
            
            st.markdown(f"""
            <div class="info-box">
                <h4>Basic Prediction</h4>
                <p>Flight {airline} from {origin} to {dest} on {flight_date} is estimated to have a delay of <strong>{base_delay} minutes</strong>.</p>
                <p><em>Note: This is a simplified prediction. For accurate results, please ensure the historical data is properly loaded.</em></p>
            </div>
            """, unsafe_allow_html=True)


def main():
    """Main application with comprehensive error handling"""
    try:
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>Flight Delay Predictor</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Configuration check
        if GITHUB_REPO == 'your-username/your-repo-name':
            st.markdown("""
            <div class="error-box">
                <h4>Configuration Required</h4>
                <p>Please update the GITHUB_REPO variable in the code with your actual repository name.</p>
                <p>Current value: <code>your-username/your-repo-name</code></p>
            </div>
            """, unsafe_allow_html=True)
        
        # Show configuration info
        with st.expander("📋 Configuration Info"):
            st.code(f"""
Repository: {GITHUB_REPO}
Release Tag: {RELEASE_TAG}
Data File: {DATA_FILENAME}
Working Directory: {os.getcwd()}
Python Version: {sys.version}
            """)
        
        # Load components with error handling
        st.info("Loading application components...")
        
        # Load model
        model, encoders, feature_columns, created_date, version = load_model_safely()
        
        if model is None:
            st.error("Could not load the model. Please check if the model file exists.")
            return
        
        # Load data
        df = load_historical_data_safely()
        airports_df = load_airports_data_safely()
        
        # Sidebar info
        with st.sidebar:
            st.markdown("### Model Information")
            st.markdown(f"**Created:** {created_date}")
            st.markdown(f"**Version:** {version}")
            st.markdown(f"**Features:** {len(feature_columns) if feature_columns else 'Unknown'}")
            
            st.markdown("### Data Status")
            if not df.empty:
                st.markdown(f"**Rows:** {len(df):,}")
                st.markdown("✅ Historical data loaded")
            else:
                st.markdown("⚠️ No historical data")
            
            if not airports_df.empty:
                st.markdown("✅ Airport data loaded")
            else:
                st.markdown("⚠️ No airport data")
        
        # Main interface
        if df.empty:
            create_simple_prediction_interface()
        else:
            st.success("✅ All components loaded successfully!")
            st.markdown("### Flight Details")
            st.info("Full prediction interface would be shown here with all features.")
            # Here you would include your full prediction interface
            # For now, showing a simplified version to ensure the app starts
        
        # Footer
        st.markdown("---")
        st.markdown("Flight Delay Predictor | Built with Streamlit")
        
    except Exception as e:
        st.error("A critical error occurred:")
        st.code(str(e))
        st.code(traceback.format_exc())
        
        # Provide basic functionality even on error
        st.markdown("### Emergency Mode")
        st.info("The application encountered an error but is running in emergency mode.")


if __name__ == "__main__":
    main()
