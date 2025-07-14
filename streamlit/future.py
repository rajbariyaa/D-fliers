import pandas as pd
import tkinter as tk
from tkinter import simpledialog, messagebox, filedialog
import tkinter.simpledialog
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import numpy as np
import datetime
import pickle
import os
import requests
import warnings
warnings.filterwarnings('ignore')

def get_weather_forecast(IATA_CODE, date_str, api_key, use_coordinates=None):
    """Get weather forecast from Visual Crossing API"""
    airport_df = pd.read_csv("airports.csv")
    # Use provided coordinates or lookup from airport code
    if use_coordinates:
        lat, lon = use_coordinates
    elif IATA_CODE in airport_df['IATA_CODE'].values:
        lat, lon = airport_df['LATITUDE'],airport_df['LONGITUDE']
    else:
        print(f" Coordinates not found for {IATA_CODE}")
        return None, "unknown_airport"
    
    # Parse the prediction date and determine data type
    try:
        prediction_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_ahead = (prediction_date - today).days
    except ValueError:
        print(f" Invalid date format: {date_str}")
        return None, "invalid_date"
    
    if days_ahead > 15:
        data_type = "extended_forecast"
        print(f" Requesting extended forecast ({days_ahead} days ahead)")
    elif days_ahead > 0:
        data_type = "forecast"
        print(f" Requesting forecast ({days_ahead} days ahead)")
    elif days_ahead == 0:
        data_type = "current"
        print(f" Requesting current weather")
    else:
        data_type = "historical"
        print(f" Requesting historical weather ({abs(days_ahead)} days ago)")
    
    url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}"
    
    params = {
        'key': api_key,
        'unitGroup': 'metric',  # Using metric for consistency
        'include': 'days',
        'elements': 'temp,humidity,pressure,windspeed,winddir,cloudcover,visibility,conditions,precip,snow'
    }
    
    try:
        print(f" Fetching weather for {IATA_CODE} ({lat:.3f}, {lon:.3f})")
        response = requests.get(url, params=params, timeout=15)

        response.raise_for_status()
        
        data = response.json()
        
        if 'days' not in data or len(data['days']) == 0:
            print(f" No weather data available for {date_str}")
            return None, "no_data"
        
        day_data = data['days'][0]
        
        weather_features = {
            'temperature': day_data.get('temp', 20.0),           # Temperature in Celsius
            'humidity': day_data.get('humidity', 50.0),          # Humidity percentage  
            'pressure': day_data.get('pressure', 1013.25),      # Pressure in hPa
            'wind_speed': day_data.get('windspeed', 10.0),      # Wind speed in km/h
            'wind_direction': day_data.get('winddir', 180.0),   # Wind direction in degrees
            'cloudiness': day_data.get('cloudcover', 25.0),     # Cloud cover percentage
            'visibility': day_data.get('visibility', 10.0),     # Visibility in km
            'weather_desc': day_data.get('conditions', 'Clear'), # Weather description
            'precipitation': day_data.get('precip', 0.0),       # Precipitation in mm
            'snow': day_data.get('snow', 0.0)                   # Snow in cm
        }
        
        print(f" {data_type.title()} weather retrieved for {IATA_CODE}")
        print(f"    Temp: {weather_features['temperature']:.1f}°C")
        print(f"    Humidity: {weather_features['humidity']:.0f}%")
        print(f"    Wind: {weather_features['wind_speed']:.1f} km/h")
        print(f"    Precip: {weather_features['precipitation']:.2f} mm")
        
        return weather_features, data_type
        
    except requests.exceptions.RequestException as e:
        print(f" API request failed: {e}")
        return None, "api_error"
    except (KeyError, IndexError, ValueError) as e:
        print(f" Error parsing weather data: {e}")
        return None, "parse_error"

def get_api_key():
    api_key = 'KZG5KUC6LL62Z5LHDDZ3TTGVC'
    
    return api_key

def save_model(model, encoders, feature_columns, filename=None):
    """Save the trained model and preprocessing components"""
    if filename is None:
        root = tk.Tk()
        root.withdraw()
        try:
            filename = tk.filedialog.asksaveasfilename(
                parent=root,
                title="Save Model As",
                defaultextension=".pkl",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
                initialfile="flight_delay_model.pkl"  
            )
        finally:
            root.destroy()
    
    if not filename:
        return False
    
    try:
        model_package = {
            'model': model,
            'encoders': encoders,
            'feature_columns': feature_columns,
            'model_type': 'XGBRegressor',
            'created_date': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'version': '2.0'  # Updated for weather forecast capability
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_package, f)
        
        print(f" Model saved successfully to: {filename}")
        messagebox.showinfo("Success", f"Model saved successfully!\n\nLocation: {filename}")
        return True
        
    except Exception as e:
        print(f" Error saving model: {e}")
        messagebox.showerror("Error", f"Failed to save model:\n{e}")
        return False

def load_model(filename=None):
    """Load a previously saved model"""
    if filename is None:
        root = tk.Tk()
        root.withdraw()
        try:
            filename = tk.filedialog.askopenfilename(
                parent=root,
                title="Load Model",
                filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
            )
        finally:
            root.destroy()
    
    if not filename or not os.path.exists(filename):
        return None, None, None
    
    try:
        with open(filename, 'rb') as f:
            model_package = pickle.load(f)
        
        model = model_package['model']
        encoders = model_package['encoders']
        feature_columns = model_package['feature_columns']
        
        created_date = model_package.get('created_date', 'Unknown')
        model_type = model_package.get('model_type', 'Unknown')
        version = model_package.get('version', 'Unknown')
        
        print(f" Model loaded successfully from: {filename}")
        print(f" Created: {created_date}")
        print(f" Type: {model_type}, Version: {version}")
        print(f" Features: {len(feature_columns)}")
        
        messagebox.showinfo("Success", 
            f"Model loaded successfully!\n\n" +
            f"Created: {created_date}\n" +
            f"Type: {model_type}\n" +
            f"Features: {len(feature_columns)}\n" +
            f"Version: {version}")
        
        return model, encoders, feature_columns
        
    except Exception as e:
        print(f" Error loading model: {e}")
        messagebox.showerror("Error", f"Failed to load model:\n{e}")
        return None, None, None

def load_and_prepare_data():
    """Load and prepare the dataset"""
    try:
        df = pd.read_csv("merged_flights_weather.csv")
        print(f" Data loaded successfully. Shape: {df.shape}")
        
        initial_rows = len(df)
        df = df.dropna(subset=['ARRIVAL_DELAY'])
        print(f" Removed {initial_rows - len(df)} rows with missing ARRIVAL_DELAY")
        
        if 'CANCELLED' in df.columns and 'DIVERTED' in df.columns:
            before_filter = len(df)
            df = df[(df['CANCELLED'] == 0) & (df['DIVERTED'] == 0)]
            print(f" Removed {before_filter - len(df)} cancelled/diverted flights")
        
        return df
    except FileNotFoundError:
        print(" Error: merged_flights_weather.csv not found!")
        return None
    except Exception as e:
        print(f" Error loading data: {e}")
        return None

def preprocess_features(df):
    """Preprocess features for training with explicit airline inclusion"""
    time_cols = ['SCHEDULED_DEPARTURE', 'SCHEDULED_ARRIVAL', 'DEPARTURE_TIME', 'ARRIVAL_TIME']
    for col in time_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    y = df['ARRIVAL_DELAY'].fillna(0)
    
    exclude = [
        'DEPARTURE_DELAY', 'ARRIVAL_DELAY', 'ARRIVAL_TIME', 'DEPARTURE_TIME',
        'DIVERTED', 'CANCELLED', 'CANCELLATION_REASON',
        'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY', 
        'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY'  
    ]
    
    features = [col for col in df.columns if col not in exclude]
    
    if 'AIRLINE' in df.columns and 'AIRLINE' not in features:
        features.append('AIRLINE')
    
    print(f" Features to be used in training:")
    for i, feature in enumerate(features, 1):
        print(f"   {i:2d}. {feature}")
    
    if 'AIRLINE' in features:
        print(f" AIRLINE feature is included in training")
        airline_counts = df['AIRLINE'].value_counts()
        print(f"   Airlines in dataset: {len(airline_counts)} unique airlines")
        print(f"   Top 5 airlines: {airline_counts.head().to_dict()}")
    else:
        print("  AIRLINE feature not found in dataset")
    
    X = df[features].copy()
    
    X = X.fillna(0)
    
    categorical_cols = X.select_dtypes(include=['object']).columns
    encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le
        print(f" Encoded {col}: {len(le.classes_)} unique values")
        
        if col == 'AIRLINE':
            print(f"   Airlines encoded: {list(le.classes_)}")
    
    return X, y, encoders, features

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(" Training XGBoost model with airline feature")
    model = XGBRegressor(
        objective='reg:squarederror',
        n_estimators=300,  
        max_depth=8,       
        learning_rate=0.08, 
        subsample=0.9,     
        colsample_bytree=0.9,
        random_state=42,
        verbosity=0 
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n Evaluation Metrics:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred):.2f} minutes")
    print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
    
    within_15_min = np.mean(np.abs(y_test - y_pred) <= 15) * 100
    within_30_min = np.mean(np.abs(y_test - y_pred) <= 30) * 100
    
    print(f"Predictions within 15 minutes: {within_15_min:.1f}%")
    print(f"Predictions within 30 minutes: {within_30_min:.1f}%")
    print(f"Median prediction error: {np.median(np.abs(y_test - y_pred)):.2f} minutes")
    
    feature_importance = model.feature_importances_
    feature_names = X_train.columns
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    print(f"\n Top 10 Most Important Features:")
    for i, (_, row) in enumerate(importance_df.head(10).iterrows(), 1):
        print(f"   {i:2d}. {row['feature']}: {row['importance']:.3f}")
    
    if 'AIRLINE' in feature_names:
        airline_importance = importance_df[importance_df['feature'] == 'AIRLINE']['importance'].iloc[0]
        airline_rank = importance_df[importance_df['feature'] == 'AIRLINE'].index[0] + 1
        print(f"\n  AIRLINE feature importance: {airline_importance:.3f} (rank #{airline_rank})")
    
    return model, X_train.columns

def get_user_input():
    """Get prediction inputs from user via GUI including airline"""
    inputs = {}
    
    try:
        root = tk.Tk()
        root.withdraw()
        
        inputs['origin'] = tk.simpledialog.askstring(
            "Flight Info", 
            "Enter ORIGIN airport code (e.g., JFK, LAX, ORD):",
            parent=root
        )
        if not inputs['origin']:
            root.destroy()
            return None
        inputs['origin'] = inputs['origin'].upper().strip()
        
        inputs['dest'] = tk.simpledialog.askstring(
            "Flight Info", 
            "Enter DESTINATION airport code (e.g., JFK, LAX, ORD):",
            parent=root
        )
        if not inputs['dest']:
            root.destroy()
            return None
        inputs['dest'] = inputs['dest'].upper().strip()
        
        inputs['airline'] = tk.simpledialog.askstring(
            "Flight Info",
            "Enter AIRLINE code (e.g., AA, UA, DL, WN):",
            parent=root
        )
        if not inputs['airline']:
            root.destroy()
            return None
        inputs['airline'] = inputs['airline'].upper().strip()
        
        tomorrow = (datetime.date.today() + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        inputs['date_str'] = tk.simpledialog.askstring(
            "Flight Info", 
            f"Enter flight DATE (YYYY-MM-DD):\n\n" +
            f"For weather forecast predictions, enter future dates\n" +
            f"Tomorrow: {tomorrow}",
            parent=root
        )
        if not inputs['date_str']:
            root.destroy()
            return None
        
        dep_time = tk.simpledialog.askstring(
            "Flight Info (Optional)", 
            "Enter SCHEDULED DEPARTURE time (HHMM, e.g., 1430)\n" +
            "Press Cancel for default (14:00):",
            parent=root
        )
        inputs['scheduled_departure'] = int(dep_time) if dep_time and dep_time.isdigit() else 1400
        
        arr_time = tk.simpledialog.askstring(
            "Flight Info (Optional)", 
            "Enter SCHEDULED ARRIVAL time (HHMM, e.g., 1630)\n" +
            "Press Cancel for default (16:00):",
            parent=root
        )
        inputs['scheduled_arrival'] = int(arr_time) if arr_time and arr_time.isdigit() else 1600
        
        root.destroy()
        return inputs
        
    except Exception as e:
        print(f"Error in user input: {e}")
        if 'root' in locals():
            root.destroy()
        return None

def create_prediction_input(inputs, df, encoders, feature_columns, api_key=None):
    """Create input dataframe for prediction with weather forecast integration and airline"""
    try:
        date_obj = datetime.datetime.strptime(inputs['date_str'], "%Y-%m-%d")
    except ValueError:
        messagebox.showerror("Error", "Invalid date format. Please use YYYY-MM-DD")
        return None
    
    if len(inputs['origin']) != 3 or len(inputs['dest']) != 3:
        messagebox.showerror("Error", "Airport codes must be 3 letters (e.g., JFK, LAX)")
        return None
    
    today = datetime.date.today()
    prediction_date = date_obj.date()
    use_forecast = prediction_date >= today and api_key is not None
    
    origin_coords = None
    dest_coords = None
    
    if 'ORIGIN_LATITUDE' in df.columns and 'ORIGIN_LONGITUDE' in df.columns:
        origin_sample = df[df['ORIGIN_AIRPORT'].astype(str).str.upper() == inputs['origin']]
        if not origin_sample.empty:
            origin_coords = (origin_sample['ORIGIN_LATITUDE'].iloc[0], origin_sample['ORIGIN_LONGITUDE'].iloc[0])
    
    if 'DEST_LATITUDE' in df.columns and 'DEST_LONGITUDE' in df.columns:
        dest_sample = df[df['DESTINATION_AIRPORT'].astype(str).str.upper() == inputs['dest']]
        if not dest_sample.empty:
            dest_coords = (dest_sample['DEST_LATITUDE'].iloc[0], dest_sample['DEST_LONGITUDE'].iloc[0])
    
    route_data = df[
        (df['ORIGIN_AIRPORT'].astype(str).str.upper() == inputs['origin']) &
        (df['DESTINATION_AIRPORT'].astype(str).str.upper() == inputs['dest']) &
        (df['MONTH'] == date_obj.month)
    ]
    
    if 'AIRLINE' in df.columns:
        airline_route_data = route_data[
            route_data['AIRLINE'].astype(str).str.upper() == inputs['airline']
        ]
        if not airline_route_data.empty:
            route_data = airline_route_data
            print(f"Found {len(route_data)} historical flights for {inputs['airline']} on this route/month")
        else:
            print(f"  No historical data for {inputs['airline']} on this route, using all airlines")
    
    if route_data.empty:
        route_data = df[df['MONTH'] == date_obj.month]
        if route_data.empty:
            messagebox.showerror("Error", "No historical data available for prediction.")
            return None
        print(f" Using general historical data (no specific route data found)")
    
    input_dict = {
        'YEAR': date_obj.year,
        'MONTH': date_obj.month,
        'DAY': date_obj.day,
        'SCHEDULED_DEPARTURE': inputs['scheduled_departure'],
        'SCHEDULED_ARRIVAL': inputs['scheduled_arrival'],
        'ORIGIN_AIRPORT': inputs['origin'],
        'DESTINATION_AIRPORT': inputs['dest'],
        'AIRLINE': inputs['airline']  
    }
    
    print(f"✈️  Creating prediction for {inputs['airline']} flight {inputs['origin']} → {inputs['dest']}")
    
    if origin_coords:
        input_dict['ORIGIN_LATITUDE'] = origin_coords[0]
        input_dict['ORIGIN_LONGITUDE'] = origin_coords[1]
    if dest_coords:
        input_dict['DEST_LATITUDE'] = dest_coords[0]
        input_dict['DEST_LONGITUDE'] = dest_coords[1]
    
    if use_forecast:
        print(f"  Fetching weather forecast for {inputs['origin']} and {inputs['dest']}")
        
        origin_weather, origin_type = get_weather_forecast(
            inputs['origin'], inputs['date_str'], api_key, origin_coords
        )
        
        dest_weather, dest_type = get_weather_forecast(
            inputs['dest'], inputs['date_str'], api_key, dest_coords
        )
        
        if origin_weather:
            for key, value in origin_weather.items():
                input_dict[f'origin_{key}'] = value
        else:
            print(f"  Using default weather for {inputs['origin']}")
            
        if dest_weather:
            for key, value in dest_weather.items():
                input_dict[f'dest_{key}'] = value
        else:
            print(f" Using default weather for {inputs['dest']}")
            
    else:
        if prediction_date >= today:
            print(f" No API key provided - using historical weather averages for future prediction")
        else:
            print(f" Using historical weather data for past date")
    
    numeric_cols = route_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col in feature_columns and col not in input_dict:
            avg_val = route_data[col].mean()
            input_dict[col] = avg_val if not pd.isna(avg_val) else 0
    
    input_df = pd.DataFrame([input_dict])
    
    for col, encoder in encoders.items():
        if col in input_df.columns:
            value = str(input_df[col].iloc[0])
            if value in encoder.classes_:
                input_df[col] = encoder.transform([value])[0]
                if col == 'AIRLINE':
                    print(f" Airline '{value}' found in trained model")
            else:
                input_df[col] = encoder.transform([encoder.classes_[0]])[0]
                if col == 'AIRLINE':
                    print(f"  Unknown airline '{value}', using default: {encoder.classes_[0]}")
                else:
                    print(f"  Unknown {col} '{value}', using default")
    
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    input_df._forecast_used = use_forecast
    
    return input_df

def predict_delay(model, df, encoders, feature_columns, api_key=None):
    """Main prediction function with GUI and weather forecast integration"""
    inputs = get_user_input()
    if not inputs:
        return
    
    input_df = create_prediction_input(inputs, df, encoders, feature_columns, api_key)
    if input_df is None:
        return
    
    try:
        prediction = model.predict(input_df)[0]
        forecast_used = getattr(input_df, '_forecast_used', False)
        
        prediction_date = datetime.datetime.strptime(inputs['date_str'], "%Y-%m-%d").date()
        today = datetime.date.today()
        is_future = prediction_date >= today
        
        if prediction > 0:
            delay_msg = f" DELAY PREDICTION\n\n"
            delay_msg += f"  Flight: {inputs['airline']} {inputs['origin']} → {inputs['dest']}\n"
            delay_msg += f" Date: {inputs['date_str']}\n"
            delay_msg += f" Departure: {inputs['scheduled_departure']//100:02d}:{inputs['scheduled_departure']%100:02d}\n\n"
            delay_msg += f" Predicted arrival delay: {prediction:.1f} minutes\n\n"
            
            if prediction > 30:
                delay_msg += " Major delay expected"
            elif prediction > 15:
                delay_msg += "  Significant delay expected"
            else:
                delay_msg += " Minor delay expected"
        else:
            delay_msg = f" ON-TIME PREDICTION\n\n"
            delay_msg += f"  Flight: {inputs['airline']} {inputs['origin']} → {inputs['dest']}\n"
            delay_msg += f" Date: {inputs['date_str']}\n"
            delay_msg += f" Departure: {inputs['scheduled_departure']//100:02d}:{inputs['scheduled_departure']%100:02d}\n\n"
            delay_msg += f" Predicted to arrive {abs(prediction):.1f} minutes early!\n\n"
            delay_msg += " Flight expected on-time or early"
        
        if is_future and forecast_used:
            delay_msg += "\n\n  Prediction uses real-time weather forecast"
        elif is_future and not forecast_used:
            delay_msg += "\n\n Prediction uses historical weather averages"
        else:
            delay_msg += "\n\n Prediction uses historical data"
        
        messagebox.showinfo("Flight Delay Prediction", delay_msg)
        
    except Exception as e:
        messagebox.showerror("Prediction Error", f"Error making prediction: {e}")

def ask_model_option():
    """Ask user whether to train new model or load existing one"""
    result = messagebox.askyesnocancel(
        "Model Selection",
        "Do you want to:\n\n" +
        "YES - Train a new model\n" +
        "NO - Load an existing model\n" +
        "CANCEL - Exit program"
    )
    return result

def main():
    """Main function with weather forecast integration"""
    print("  Flight Delay Prediction System with Weather Forecast & Airline Feature")
    print("=" * 75)
    
    root = tk.Tk()
    root.withdraw()
    
    model_choice = ask_model_option()
    
    if model_choice is None:  # Cancel
        print(" Goodbye!")
        return
    
    # use_forecast = messagebox.askyesno(
    #     "Weather Forecast",
    #     "Do you want to use real-time weather forecasts?\n\n" +
    #     "YES - Get Visual Crossing API key for accurate forecasts\n" +
    #     "NO - Use historical weather averages only\n\n" +
    #     "Note: Forecasts significantly improve accuracy for future flights!"
    # )
    
    # api_key = None
    # if use_forecast:
    api_key = get_api_key()
    if not api_key:
        messagebox.showwarning(
            "Warning", 
            "No API key provided. Will use historical weather averages."
                                        )
    
    if model_choice:  
        df = load_and_prepare_data()
        if df is None:
            return
        
        X, y, encoders, features = preprocess_features(df)
        print(f" Features prepared: {len(features)} features, {len(X)} samples")
        
        model, feature_columns = train_model(X, y)
        print(" Model training completed!")
        
        save_choice = messagebox.askyesno(
            "Save Model",
            "Would you like to save the trained model?\n\n" +
            "This allows you to reuse it later without retraining."
        )
        
        if save_choice:
            save_model(model, encoders, feature_columns)
    
    else:  
        model, encoders, feature_columns = load_model()
        if model is None:
            print(" No model loaded. Exiting")
            return
        
        df = load_and_prepare_data()
        if df is None:
            print(" Warning: Dataset not found. Using fallback data.")
            df = pd.DataFrame()
    
    forecast_status = "with real-time weather forecasts" if api_key else "with historical weather data"
    messagebox.showinfo(
        "Flight Delay Predictor", 
        f"  Welcome to Flight Delay Predictor {forecast_status}!\n\n" +
        f" Features include airline information for better accuracy\n" +
        f" Use future dates for forecast predictions\n" +
        f"Historical dates use past weather data\n\n" +
        "Click OK to start making predictions."
    )
    
    while True:
        try:
            predict_delay(model, df, encoders, feature_columns, api_key)
            
            another = messagebox.askyesno(
                "Continue?", 
                "Would you like to make another prediction?"
            )
            if not another:
                break
                
        except Exception as e:
            messagebox.showerror(
                "Error", 
                f"An error occurred: {e}\n\nClick OK to exit."
            )
            break
    
    messagebox.showinfo("Goodbye", "Thank you for using Flight Delay Predictor! ")
    root.destroy()

if __name__ == "__main__":
    main()