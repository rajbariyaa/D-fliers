import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
import warnings
import os
import sys
import gc
from pathlib import Path
import time
import traceback

# Suppress warnings
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

# Memory management settings
MAX_MEMORY_GB = 1.5  # Conservative limit for Streamlit Cloud
CHUNK_SIZE = 50000   # Process in chunks if needed

def get_memory_usage():
    """Get current memory usage using built-in methods"""
    try:
        # Simple memory tracking using garbage collector stats
        import resource
        memory_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # On Linux, this is in KB, on macOS it's in bytes
        if sys.platform == 'darwin':  # macOS
            memory_mb = memory_kb / 1024 / 1024
        else:  # Linux
            memory_mb = memory_kb / 1024
        return memory_mb
    except:
        # Fallback: estimate based on DataFrame sizes if resource unavailable
        return 0

def log_memory_usage(message=""):
    """Log current memory usage"""
    memory_mb = get_memory_usage()
    if memory_mb > 0:
        st.write(f"🧠 Memory: {memory_mb:.1f} MB {message}")
    else:
        st.write(f"🧠 Memory monitoring unavailable {message}")
    return memory_mb

# Create data directory
try:
    LOCAL_DATA_DIR.mkdir(exist_ok=True)
except Exception as e:
    st.error(f"Could not create data directory: {e}")

def download_with_direct_url(repo, tag, filename, local_path):
    """Download using direct GitHub URL"""
    try:
        direct_url = f"https://github.com/{repo}/releases/download/{tag}/{filename}"
        st.info(f"📥 Downloading from: {direct_url}")
        
        # Check if URL is accessible
        head_response = requests.head(direct_url, timeout=10)
        if head_response.status_code == 404:
            st.error(f"❌ File not found at {direct_url}")
            st.info("Please verify:")
            st.info(f"1. Release '{tag}' exists at https://github.com/{repo}/releases")
            st.info(f"2. File '{filename}' is uploaded to that release")
            st.info("3. Release is published (not draft)")
            return False, "File not found"
        
        response = requests.get(direct_url, stream=True, timeout=300)
        response.raise_for_status()
        
        # Get file size
        file_size = head_response.headers.get('content-length') or response.headers.get('content-length')
        if file_size:
            file_size = int(file_size)
            file_size_mb = file_size / (1024 * 1024)
            st.info(f"📊 File size: {file_size_mb:.1f} MB")
            
            # Check if we have enough space
            available_memory = (MAX_MEMORY_GB * 1024) - get_memory_usage()
            if file_size_mb > available_memory:
                st.warning(f"⚠️ File size ({file_size_mb:.1f} MB) may exceed available memory ({available_memory:.1f} MB)")
        
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
        status_text.text("✅ Download completed!")
        return True, "Download successful"
        
    except requests.exceptions.Timeout:
        return False, "Download timeout - file may be too large"
    except requests.exceptions.RequestException as e:
        return False, f"Network error: {str(e)}"
    except Exception as e:
        return False, f"Error: {str(e)}"

def load_model_safely():
    """Load model with comprehensive error handling"""
    st.info("🔄 Loading model...")
    
    model_paths = [
        'models/test02.pkl',
        'test02.pkl',
        './models/test02.pkl',
        './test02.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            st.info(f"📂 Found model at: {path}")
            try:
                with open(path, 'rb') as f:
                    model_package = pickle.load(f)
                
                model = model_package.get('model')
                encoders = model_package.get('encoders', {})
                feature_columns = model_package.get('feature_columns', [])
                
                # Test model
                if model and hasattr(model, 'predict'):
                    st.success("✅ Model loaded and validated")
                    return model, encoders, feature_columns, 'Loaded', '1.0'
                else:
                    st.error("❌ Model object is invalid")
                    
            except Exception as e:
                st.error(f"❌ Model loading failed: {str(e)}")
                if "xgboost" in str(e).lower():
                    st.error("🔧 XGBoost version compatibility issue detected")
                    st.info("💡 Try updating the model with current XGBoost version")
    
    st.error("❌ No valid model found")
    return None, {}, [], 'None', 'None'

def load_data_progressively():
    """Load data with memory management and progressive loading"""
    st.info("🔄 Loading historical data...")
    
    try:
        # Check if file exists locally
        if LOCAL_DATA_FILE.exists():
            file_size_mb = LOCAL_DATA_FILE.stat().st_size / (1024 * 1024)
            st.info(f"📊 Local file size: {file_size_mb:.1f} MB")
            
            # Conservative memory check - if file is larger than 500MB, use chunked loading
            # This is conservative since Streamlit Cloud has limited memory
            if file_size_mb > 500:
                st.warning(f"⚠️ Large file detected ({file_size_mb:.1f} MB). Using smart loading strategy...")
                return load_data_in_chunks()
            else:
                st.info("✅ File size manageable. Loading full dataset...")
                return load_full_dataset()
        else:
            # Download first
            st.info("📥 File not found locally. Downloading...")
            success, message = download_with_direct_url(
                GITHUB_REPO, RELEASE_TAG, DATA_FILENAME, LOCAL_DATA_FILE
            )
            
            if success:
                return load_data_progressively()  # Recursive call after download
            else:
                st.error(f"❌ Download failed: {message}")
                return pd.DataFrame()
    
    except Exception as e:
        st.error(f"❌ Error in progressive loading: {str(e)}")
        st.code(traceback.format_exc())
        return pd.DataFrame()

def load_full_dataset():
    """Load the complete dataset"""
    try:
        st.info("📖 Reading full dataset...")
        log_memory_usage("before loading")
        
        # Load with optimized dtypes
        df = pd.read_csv(LOCAL_DATA_FILE, low_memory=False)
        
        log_memory_usage("after loading")
        
        # Optimize memory usage
        st.info("🔧 Optimizing memory usage...")
        df = optimize_dataframe_memory(df)
        
        log_memory_usage("after optimization")
        
        st.success(f"✅ Loaded full dataset: {len(df):,} rows, {len(df.columns)} columns")
        return df
        
    except MemoryError:
        st.error("❌ Out of memory loading full dataset. Switching to chunked loading...")
        return load_data_in_chunks()
    except Exception as e:
        st.error(f"❌ Error loading full dataset: {str(e)}")
        return load_data_in_chunks()

def load_data_in_chunks():
    """Load data in chunks to manage memory"""
    try:
        st.info("📚 Loading data in chunks...")
        
        # Get total rows first
        total_rows = sum(1 for _ in open(LOCAL_DATA_FILE)) - 1  # -1 for header
        st.info(f"📊 Total rows: {total_rows:,}")
        
        # Conservative approach: load a reasonable sample
        # For large datasets, use 100k rows as a good balance between performance and completeness
        max_rows = min(100000, total_rows)
        
        st.info(f"📊 Loading sample of {max_rows:,} rows for optimal performance")
        
        if max_rows < total_rows:
            # Load random sample
            skip_rows = sorted(np.random.choice(range(1, total_rows + 1), 
                                              size=total_rows - max_rows, 
                                              replace=False))
            
            df = pd.read_csv(LOCAL_DATA_FILE, skiprows=skip_rows, low_memory=False)
        else:
            # Load all rows if dataset is small
            df = pd.read_csv(LOCAL_DATA_FILE, low_memory=False)
        
        df = optimize_dataframe_memory(df)
        
        st.success(f"✅ Loaded sample: {len(df):,} rows")
        
        if len(df) < total_rows:
            st.info(f"📈 This represents {len(df)/total_rows*100:.1f}% of the full dataset")
        
        return df
        
    except Exception as e:
        st.error(f"❌ Chunked loading failed: {str(e)}")
        st.info("🔄 Falling back to minimal dataset...")
        return load_minimal_dataset()

def load_minimal_dataset():
    """Load minimal dataset as last resort"""
    try:
        st.info("📝 Loading minimal dataset (first 5000 rows)...")
        df = pd.read_csv(LOCAL_DATA_FILE, nrows=5000, low_memory=False)
        df = optimize_dataframe_memory(df)
        st.warning(f"⚠️ Using minimal dataset: {len(df):,} rows")
        return df
    except Exception as e:
        st.error(f"❌ Even minimal loading failed: {str(e)}")
        return pd.DataFrame()

def optimize_dataframe_memory(df):
    """Optimize DataFrame memory usage"""
    try:
        initial_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        for col in df.columns:
            col_type = df[col].dtype
            
            if col_type != 'object':
                c_min = df[col].min()
                c_max = df[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                
                elif str(col_type)[:5] == 'float':
                    if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
        
        final_memory = df.memory_usage(deep=True).sum() / 1024 / 1024
        reduction = (initial_memory - final_memory) / initial_memory * 100
        
        if reduction > 0:
            st.info(f"🎯 Memory optimized: {reduction:.1f}% reduction ({initial_memory:.1f}→{final_memory:.1f} MB)")
        
        return df
        
    except Exception as e:
        st.warning(f"Memory optimization failed: {str(e)}")
        return df

def create_prediction_interface(model, encoders, feature_columns):
    """Create the prediction interface"""
    st.markdown("### ✈️ Flight Delay Prediction")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            origin = st.text_input("Origin Airport", value="JFK", max_chars=3).upper()
            airline = st.selectbox("Airline", ["AA", "UA", "DL", "WN", "AS", "B6"])
            departure_time = st.time_input("Departure Time", value=datetime.time(14, 0))
        
        with col2:
            destination = st.text_input("Destination Airport", value="LAX", max_chars=3).upper()
            flight_date = st.date_input("Flight Date", value=datetime.date.today() + datetime.timedelta(days=1))
            arrival_time = st.time_input("Arrival Time", value=datetime.time(17, 0))
        
        submitted = st.form_submit_button("🔮 Predict Delay", use_container_width=True)
        
        if submitted:
            if len(origin) != 3 or len(destination) != 3:
                st.error("Airport codes must be exactly 3 letters!")
                return
            
            try:
                # Create prediction input
                input_data = {
                    'SCHEDULED_DEPARTURE': departure_time.hour * 100 + departure_time.minute,
                    'SCHEDULED_ARRIVAL': arrival_time.hour * 100 + arrival_time.minute,
                    'ORIGIN_AIRPORT': origin,
                    'DESTINATION_AIRPORT': destination,
                    'AIRLINE': airline
                }
                
                input_df = pd.DataFrame([input_data])
                
                # Handle missing features
                for col in feature_columns:
                    if col not in input_df.columns:
                        input_df[col] = 0
                
                input_df = input_df.reindex(columns=feature_columns, fill_value=0)
                
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Display result with better formatting
                if prediction <= 0:
                    st.success(f"🎉 **ON TIME** - Arrives {abs(prediction):.0f} min early!")
                elif prediction <= 15:
                    st.info(f"✈️ **MINOR DELAY** - {prediction:.0f} minutes late")
                elif prediction <= 30:
                    st.warning(f"⚠️ **MODERATE DELAY** - {prediction:.0f} minutes late")
                else:
                    st.error(f"🚨 **MAJOR DELAY** - {prediction:.0f} minutes late")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")

def main():
    """Main application with robust error handling"""
    try:
        # Header
        st.title("🛫 Flight Delay Predictor")
        st.markdown("*AI-powered flight delay predictions*")
        
        # System information
        with st.expander("🔧 System Status"):
            log_memory_usage("at startup")
            st.write(f"**Python:** {sys.version}")
            st.write(f"**Working Dir:** {os.getcwd()}")
            st.write(f"**Config:** {GITHUB_REPO} / {RELEASE_TAG}")
        
        # Initialize components
        progress_container = st.container()
        
        with progress_container:
            st.info("🚀 Initializing Flight Delay Predictor...")
            
            # Load model
            model, encoders, feature_columns, created_date, version = load_model_safely()
            
            # Load data
            df = load_data_progressively()
            
            # Force garbage collection
            gc.collect()
        
        # Status dashboard
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if model:
                st.success("✅ Model Ready")
            else:
                st.error("❌ Model Failed")
        
        with col2:
            if not df.empty:
                st.success(f"✅ Data Ready")
                st.caption(f"{len(df):,} rows")
            else:
                st.error("❌ No Data")
        
        with col3:
            memory_mb = get_memory_usage()
            if memory_mb > 0:
                if memory_mb < MAX_MEMORY_GB * 1024 * 0.8:
                    st.success("✅ Memory OK")
                else:
                    st.warning("⚠️ High Memory")
                st.caption(f"{memory_mb:.0f} MB")
            else:
                st.info("ℹ️ Memory Status")
                st.caption("Monitoring N/A")
        
        # Main interface
        if model:
            create_prediction_interface(model, encoders, feature_columns)
        else:
            st.error("❌ Cannot create interface without model")
            st.info("Please check the model file and XGBoost compatibility")
        
        # Data info
        if not df.empty:
            with st.expander("📊 Dataset Information"):
                st.write(f"**Rows:** {len(df):,}")
                st.write(f"**Columns:** {len(df.columns)}")
                if len(df.columns) > 0:
                    st.write(f"**Sample columns:** {', '.join(df.columns[:5].tolist())}")
        
    except Exception as e:
        st.error("🚨 Critical Error")
        st.code(str(e))
        st.code(traceback.format_exc())
        
        # Emergency info
        st.info("🆘 Emergency mode activated. Check system requirements and file accessibility.")

if __name__ == "__main__":
    main()
