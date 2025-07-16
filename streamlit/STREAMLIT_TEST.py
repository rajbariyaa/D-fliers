import streamlit as st
import pandas as pd
import numpy as np
import pickle
import datetime
import requests
import warnings
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor with Weather Analysis",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simplified CSS for better compatibility
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        text-align: center;
        color: white;
    }
    
    .weather-risk-high {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .weather-risk-medium {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .weather-risk-low {
        background-color: #e8f5e8;
        border-left: 5px solid #4caf50;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .weather-factor {
        background-color: #f5f5f5;
        padding: 10px;
        margin: 5px 0;
        border-radius: 5px;
        border-left: 3px solid #2196f3;
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
    
    .success-box {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
        padding: 15px;
        margin: 10px 0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


def load_model_safe():
    """Safely load the saved model and components"""
    try:
        with open('models/test02.pkl', 'rb') as f:
            model_package = pickle.load(f)

        return (
            model_package['model'],
            model_package['encoders'],
            model_package['feature_columns'],
            model_package.get('created_date', 'Unknown'),
            model_package.get('version', 'Unknown')
        )
    except FileNotFoundError:
        st.error(" Model file 'test02.pkl' not found. Please ensure the model file is in the correct directory.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f" Error loading model: {str(e)}")
        return None, None, None, None, None


def load_airports_data_safe():
    """Safely load airports data"""
    try:
        airports_df = pd.read_csv("airports.csv")
        return airports_df
    except FileNotFoundError:
        st.warning(" Airports data file 'airports.csv' not found. Weather forecasts may be limited.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f" Error loading airports data: {str(e)}")
        return pd.DataFrame()


def get_weather_forecast(iata_code, date_str, api_key, airports_df=None):
    """Get weather forecast from Visual Crossing API"""
    if not api_key:
        return None, "no_api_key"

    try:
        # Get coordinates for the airport
        if airports_df is not None and not airports_df.empty:
            if iata_code in airports_df['IATA_CODE'].values:
                airport_row = airports_df[airports_df['IATA_CODE'] == iata_code].iloc[0]
                lat, lon = airport_row['LATITUDE'], airport_row['LONGITUDE']
            else:
                return None, "unknown_airport"
        else:
            # Fallback coordinates for common airports
            fallback_coords = {
                'JFK': (40.6413, -73.7781),
                'LAX': (33.9425, -118.4081),
                'ORD': (41.9742, -87.9073),
                'DFW': (32.8975, -97.0380),
                'DEN': (39.8561, -104.6737)
            }
            if iata_code in fallback_coords:
                lat, lon = fallback_coords[iata_code]
            else:
                return None, "unknown_airport"

        # Visual Crossing API call
        url = f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{lat},{lon}/{date_str}"

        params = {
            'key': api_key,
            'unitGroup': 'metric',
            'include': 'days',
            'elements': 'temp,humidity,pressure,windspeed,winddir,cloudcover,visibility,conditions,precip,snow'
        }

        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()

        data = response.json()

        if 'days' not in data or len(data['days']) == 0:
            return None, "no_data"

        day_data = data['days'][0]

        weather_features = {
            'temperature': day_data.get('temp', 20.0),
            'humidity': day_data.get('humidity', 50.0),
            'pressure': day_data.get('pressure', 1013.25),
            'wind_speed': day_data.get('windspeed', 10.0),
            'wind_direction': day_data.get('winddir', 180.0),
            'cloudiness': day_data.get('cloudcover', 25.0),
            'visibility': day_data.get('visibility', 10.0),
            'weather_desc': day_data.get('conditions', 'Clear'),
            'precipitation': day_data.get('precip', 0.0),
            'snow': day_data.get('snow', 0.0)
        }

        return weather_features, "success"

    except Exception as e:
        return None, f"error: {str(e)}"


def analyze_weather_impact(weather_data, location_name):
    """Analyze weather conditions and identify delay-causing factors"""
    if not weather_data:
        return [], "LOW"
    
    delay_factors = []
    risk_level = "LOW"
    
    # Define thresholds for weather conditions that typically cause delays
    thresholds = {
        'high_wind': 40,        # km/h
        'low_visibility': 5,    # km
        'heavy_rain': 10,       # mm
        'snow_present': 0.5,    # cm
        'extreme_cold': -10,    # °C
        'extreme_heat': 35,     # °C
    }
    
    # Check each weather factor
    if weather_data['wind_speed'] > thresholds['high_wind']:
        severity = "HIGH" if weather_data['wind_speed'] > 60 else "MEDIUM"
        delay_factors.append({
            'factor': 'High Wind Speed',
            'value': f"{weather_data['wind_speed']:.1f} km/h",
            'threshold': f">{thresholds['high_wind']} km/h",
            'severity': severity,
            'impact': 'Can cause landing/takeoff delays and turbulence',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['visibility'] < thresholds['low_visibility']:
        severity = "HIGH" if weather_data['visibility'] < 2 else "MEDIUM"
        delay_factors.append({
            'factor': 'Poor Visibility',
            'value': f"{weather_data['visibility']:.1f} km",
            'threshold': f"<{thresholds['low_visibility']} km",
            'severity': severity,
            'impact': 'Requires instrument approaches, reduces airport capacity',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['precipitation'] > thresholds['heavy_rain']:
        severity = "HIGH" if weather_data['precipitation'] > 25 else "MEDIUM"
        delay_factors.append({
            'factor': 'Heavy Precipitation',
            'value': f"{weather_data['precipitation']:.1f} mm",
            'threshold': f">{thresholds['heavy_rain']} mm",
            'severity': severity,
            'impact': 'Reduces runway capacity, increases stopping distances',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['snow'] > thresholds['snow_present']:
        severity = "HIGH" if weather_data['snow'] > 5 else "MEDIUM"
        delay_factors.append({
            'factor': 'Snow Conditions',
            'value': f"{weather_data['snow']:.1f} cm",
            'threshold': f">{thresholds['snow_present']} cm",
            'severity': severity,
            'impact': 'Requires de-icing, runway clearing, major delays possible',
            'location': location_name
        })
        risk_level = "HIGH"  # Snow always high risk
    
    if weather_data['temperature'] < thresholds['extreme_cold']:
        severity = "HIGH" if weather_data['temperature'] < -20 else "MEDIUM"
        delay_factors.append({
            'factor': 'Extreme Cold',
            'value': f"{weather_data['temperature']:.1f}°C",
            'threshold': f"<{thresholds['extreme_cold']}°C",
            'severity': severity,
            'impact': 'Requires extensive de-icing, equipment issues possible',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    if weather_data['temperature'] > thresholds['extreme_heat']:
        severity = "MEDIUM"
        delay_factors.append({
            'factor': 'Extreme Heat',
            'value': f"{weather_data['temperature']:.1f}°C",
            'threshold': f">{thresholds['extreme_heat']}°C",
            'severity': severity,
            'impact': 'Reduces aircraft performance, weight restrictions possible',
            'location': location_name
        })
        if risk_level == "LOW":
            risk_level = "MEDIUM"
    
    # Check for severe weather conditions
    severe_conditions = ['thunderstorm', 'storm', 'fog', 'mist', 'freezing', 'blizzard', 'hail']
    if any(condition in weather_data['weather_desc'].lower() for condition in severe_conditions):
        severity = "HIGH" if any(condition in weather_data['weather_desc'].lower() 
                               for condition in ['thunderstorm', 'blizzard', 'freezing']) else "MEDIUM"
        delay_factors.append({
            'factor': 'Severe Weather Conditions',
            'value': weather_data['weather_desc'],
            'threshold': 'Clear/Partly Cloudy preferred',
            'severity': severity,
            'impact': 'Various operational restrictions and safety concerns',
            'location': location_name
        })
        if severity == "HIGH":
            risk_level = "HIGH"
        elif risk_level == "LOW":
            risk_level = "MEDIUM"
    
    return delay_factors, risk_level


def display_weather_analysis(origin_weather, dest_weather, origin_code, dest_code, prediction):
    """Display comprehensive weather analysis"""
    st.markdown("###  Weather Delay Analysis")
    
    # Analyze weather impact for both locations
    origin_factors, origin_risk = analyze_weather_impact(origin_weather, f"Origin ({origin_code})")
    dest_factors, dest_risk = analyze_weather_impact(dest_weather, f"Destination ({dest_code})")
    
    # Determine overall risk
    all_factors = origin_factors + dest_factors
    overall_risk = "LOW"
    if origin_risk == "HIGH" or dest_risk == "HIGH":
        overall_risk = "HIGH"
    elif origin_risk == "MEDIUM" or dest_risk == "MEDIUM":
        overall_risk = "MEDIUM"
    
    # Risk level display
    risk_class = f"weather-risk-{overall_risk.lower()}"
    risk_icon = {"LOW": "🟢", "MEDIUM": "🟡", "HIGH": "🔴"}[overall_risk]
    
    st.markdown(f"""
    <div class="{risk_class}">
        <h4>{risk_icon} Overall Weather Risk: {overall_risk}</h4>
        <p>Based on analysis of weather conditions at both origin and destination</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Weather factors analysis
    if all_factors:
        st.markdown("#### Weather Factors Contributing to Delays")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**🛫 {origin_code} Weather Issues**")
            if origin_factors:
                for factor in origin_factors:
                    severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[factor['severity']]
                    st.markdown(f"""
                    <div class="weather-factor">
                        <strong>{severity_icon} {factor['factor']}</strong><br>
                        Current: {factor['value']} | Threshold: {factor['threshold']}<br>
                        <em>{factor['impact']}</em>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(" No significant weather issues detected")
        
        with col2:
            st.markdown(f"**🛬 {dest_code} Weather Issues**")
            if dest_factors:
                for factor in dest_factors:
                    severity_icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}[factor['severity']]
                    st.markdown(f"""
                    <div class="weather-factor">
                        <strong>{severity_icon} {factor['factor']}</strong><br>
                        Current: {factor['value']} | Threshold: {factor['threshold']}<br>
                        <em>{factor['impact']}</em>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown(" No significant weather issues detected")
    else:
        st.markdown("""
        <div class="success-box">
            <h4> Favorable Weather Conditions</h4>
            <p>No significant weather factors identified that would cause delays. Weather conditions appear favorable for on-time operations.</p>
        </div>
        """, unsafe_allow_html=True)


def display_weather_forecast(origin_weather, dest_weather, origin_code, dest_code, flight_date):
    """Display weather forecast information"""
    st.markdown("###  Weather Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**🛫 Origin: {origin_code}**")
        if origin_weather:
            st.write(f"**Date:** {flight_date}")
            st.write(f"**Conditions:** {origin_weather['weather_desc']}")
            st.write(f"**Temperature:** {origin_weather['temperature']:.1f}°C")
            st.write(f"**Humidity:** {origin_weather['humidity']:.1f}%")
            st.write(f"**Wind Speed:** {origin_weather['wind_speed']:.1f} km/h")
            st.write(f"**Visibility:** {origin_weather['visibility']:.1f} km")
            st.write(f"**Precipitation:** {origin_weather['precipitation']:.1f} mm")
            if origin_weather['snow'] > 0:
                st.write(f"**Snow:** {origin_weather['snow']:.1f} cm")
        else:
            st.write("Weather forecast not available")
    
    with col2:
        st.markdown(f"**🛬 Destination: {dest_code}**")
        if dest_weather:
            st.write(f"**Date:** {flight_date}")
            st.write(f"**Conditions:** {dest_weather['weather_desc']}")
            st.write(f"**Temperature:** {dest_weather['temperature']:.1f}°C")
            st.write(f"**Humidity:** {dest_weather['humidity']:.1f}%")
            st.write(f"**Wind Speed:** {dest_weather['wind_speed']:.1f} km/h")
            st.write(f"**Visibility:** {dest_weather['visibility']:.1f} km")
            st.write(f"**Precipitation:** {dest_weather['precipitation']:.1f} mm")
            if dest_weather['snow'] > 0:
                st.write(f"**Snow:** {dest_weather['snow']:.1f} cm")
        else:
            st.write("Weather forecast not available")


def create_prediction_input(inputs, encoders, feature_columns, api_key, airports_df=None):
    """Create input dataframe for prediction and return weather data"""
    try:
        date_obj = datetime.datetime.strptime(inputs['date_str'], "%Y-%m-%d")
    except ValueError:
        return None, "Invalid date format", None, None

    # Create base input
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

    # Add weather data if available
    today = datetime.date.today()
    prediction_date = date_obj.date()
    use_forecast = prediction_date >= today and api_key is not None

    origin_weather = None
    dest_weather = None

    if use_forecast:
        # Get weather forecasts
        origin_weather, _ = get_weather_forecast(inputs['origin'], inputs['date_str'], api_key, airports_df)
        dest_weather, _ = get_weather_forecast(inputs['dest'], inputs['date_str'], api_key, airports_df)

        if origin_weather:
            for key, value in origin_weather.items():
                if key != 'weather_desc':  # Skip non-numeric
                    input_dict[f'origin_{key}'] = value

        if dest_weather:
            for key, value in dest_weather.items():
                if key != 'weather_desc':  # Skip non-numeric
                    input_dict[f'dest_{key}'] = value

    # Fill missing features with defaults
    for col in feature_columns:
        if col not in input_dict:
            input_dict[col] = 0

    # Create DataFrame
    input_df = pd.DataFrame([input_dict])

    # Encode categorical features
    for col, encoder in encoders.items():
        if col in input_df.columns:
            value = str(input_df[col].iloc[0])
            if value in encoder.classes_:
                input_df[col] = encoder.transform([value])[0]
            else:
                input_df[col] = encoder.transform([encoder.classes_[0]])[0]

    # Ensure all required features are present
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    return input_df, "success", origin_weather, dest_weather


def create_delay_visualization(prediction, airline, route):
    """Create a visualization for the delay prediction"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=prediction,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{airline} {route}<br>Delay Prediction (minutes)"},
        delta={'reference': 0},
        gauge={
            'axis': {'range': [None, 120]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 15], 'color': "lightgreen"},
                {'range': [15, 30], 'color': "yellow"},
                {'range': [30, 60], 'color': "orange"},
                {'range': [60, 120], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 30
            }
        }
    ))

    fig.update_layout(
        height=400,
        font={'color': "darkblue", 'family': "Arial"}
    )

    return fig


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1> Flight Delay Predictor with Weather Analysis</h1>
    </div>
    """, unsafe_allow_html=True)
    
    model, encoders, feature_columns, created_date, version = load_model_safe()
    if model is None:
        st.stop()

    airports_df = load_airports_data_safe()

    # Sidebar
    with st.sidebar:
        st.markdown("###  Model Information")
        st.write(f"**Created:** {created_date}")
        st.write(f"**Version:** {version}")
        st.write(f"**Features:** {len(feature_columns) if feature_columns is not None else 'Unknown'}")

        st.markdown("###  Weather Settings")
        use_weather = st.checkbox("Use Real-time Weather Forecasts", value=True)
        show_analysis = st.checkbox("Show Weather Delay Analysis", value=True)

    # Main content
    st.markdown("###  Flight Details")

    # Flight input form
    with st.form("flight_form"):
        col1, col2 = st.columns(2)

        with col1:
            origin = st.text_input("Origin Airport Code", value="JFK").upper()
            airline = st.selectbox("Airline", options=["AA", "UA", "DL", "WN", "AS", "B6"])
            departure_time = st.time_input("Scheduled Departure", value=datetime.time(14, 0))

        with col2:
            dest = st.text_input("Destination Airport Code", value="LAX").upper()
            flight_date = st.date_input("Flight Date", value=datetime.date.today() + datetime.timedelta(days=1))
            arrival_time = st.time_input("Scheduled Arrival", value=datetime.time(17, 0))

        submitted = st.form_submit_button(" Predict Delay")

    # Prediction results
    if submitted:
        # Validate inputs
        if len(origin) != 3 or len(dest) != 3:
            st.error("Airport codes must be exactly 3 letters!")
            st.stop()

        if origin == dest:
            st.error("Origin and destination cannot be the same!")
            st.stop()

        # Prepare input data
        inputs = {
            'origin': origin,
            'dest': dest,
            'airline': airline,
            'date_str': flight_date.strftime("%Y-%m-%d"),
            'scheduled_departure': departure_time.hour * 100 + departure_time.minute,
            'scheduled_arrival': arrival_time.hour * 100 + arrival_time.minute
        }

        # Show progress
        with st.spinner("Processing prediction and analyzing weather..."):
            api_key = 'KZG5KUC6LL62Z5LHDDZ3TTGVC' if use_weather else None
            
            input_df, status, origin_weather, dest_weather = create_prediction_input(
                inputs, encoders, feature_columns, api_key, airports_df
            )

            if input_df is None:
                st.error(f"Error creating prediction input: {status}")
                st.stop()

            # Make prediction
            try:
                prediction = model.predict(input_df)[0]

                # Display results
                st.markdown("###  Prediction Results")

                # Create visualization
                fig = create_delay_visualization(prediction, airline, f"{origin} → {dest}")
                st.plotly_chart(fig, use_container_width=True)

                # Results summary
                if prediction > 0:
                    if prediction > 30:
                        result_class = "warning-box"
                        severity = " Major Delay"
                    elif prediction > 15:
                        result_class = "warning-box"
                        severity = " Significant Delay"
                    else:
                        result_class = "info-box"
                        severity = " Minor Delay"

                    st.markdown(f"""
                    <div class="{result_class}">
                        <h4>{severity} Expected</h4>
                        <p><strong>Flight {airline} {origin} → {dest}</strong> is predicted to arrive <strong>{prediction:.1f} minutes late</strong> on {flight_date.strftime('%B %d, %Y')}.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="success-box">
                        <h4> On-Time or Early Arrival</h4>
                        <p><strong>Flight {airline} {origin} → {dest}</strong> is predicted to arrive <strong>{abs(prediction):.1f} minutes early</strong> on {flight_date.strftime('%B %d, %Y')}.</p>
                    </div>
                    """, unsafe_allow_html=True)

                # Display weather analysis
                if show_analysis and use_weather and (origin_weather or dest_weather):
                    display_weather_analysis(origin_weather, dest_weather, origin, dest, prediction)
                
                # Display weather forecast
                if use_weather and (origin_weather or dest_weather):
                    display_weather_forecast(origin_weather, dest_weather, origin, dest, flight_date.strftime('%B %d, %Y'))
                elif use_weather:
                    st.markdown("""
                    <div class="info-box">
                        <h4> Weather Information</h4>
                        <p>Weather forecasts are not available for the selected airports or date. The prediction uses historical weather patterns.</p>
                    </div>
                    """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.write("Please check that all required files are present and try again.")

    # Footer
    st.markdown("---")
    st.markdown(" **Flight Delay Predictor with Weather Analysis** | Built with Streamlit & Machine Learning")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.write("Please check your files and dependencies.")
