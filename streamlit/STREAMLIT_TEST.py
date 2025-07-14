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
import xgboost

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

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

    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }

    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }

    .weather-card {
        background: linear-gradient(135deg, #74b9ff 0%, #0984e3 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }

    .weather-location {
        background: rgba(255,255,255,0.1);
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .info-box {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #17a2b8;
        margin: 1rem 0;
    }

    .warning-box {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }

    .success-box {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }

    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-weight: bold;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102,126,234,0.4);
    }

    .sidebar .stSelectbox > div > div {
        background-color: #f8f9fa;
    }

    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_model():
    """Load the saved model and components"""
    try:
        with open('test02.pkl', 'rb') as f:
            model_package = pickle.load(f)

        return (
            model_package['model'],
            model_package['encoders'],
            model_package['feature_columns'],
            model_package.get('created_date', 'Unknown'),
            model_package.get('version', 'Unknown')
        )
    except FileNotFoundError:
        st.error("Model file not found.")
        return None, None, None, None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None, None, None, None


@st.cache_data
def load_historical_data():
    """Load historical data for reference"""
    try:
        df = pd.read_csv("merged_flights_weather.csv")
        return df
    except FileNotFoundError:
        st.warning("Historical data file not found. Using fallback data.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading historical data: {str(e)}")
        return pd.DataFrame()


@st.cache_data
def load_airports_data():
    """Load airports data"""
    try:
        airports_df = pd.read_csv("airports.csv")
        return airports_df
    except FileNotFoundError:
        st.warning("Airports data file not found. Using fallback coordinates.")
        return pd.DataFrame()
    except Exception as e:
        st.warning(f"Error loading airports data: {str(e)}")
        return pd.DataFrame()


def get_weather_forecast(iata_code, date_str, api_key, airports_df=None):
    """Get weather forecast from Visual Crossing API"""
    if not api_key:
        return None, "no_api_key"

    try:
        # Parse the prediction date
        prediction_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        today = datetime.date.today()
        days_ahead = (prediction_date - today).days

        # Get coordinates for the airport
        if airports_df is not None and not airports_df.empty:
            if iata_code in airports_df['IATA_CODE'].values:
                airport_row = airports_df[airports_df['IATA_CODE'] == iata_code].iloc[0]
                lat, lon = airport_row['LATITUDE'], airport_row['LONGITUDE']
            else:
                return None, "unknown_airport"
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


def display_weather_forecast(origin_weather, dest_weather, origin_code, dest_code, flight_date):
    """Display weather forecast information"""
    st.markdown("### Weather Forecast")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if origin_weather:
            st.markdown(f"""
            <div class="weather-card">
                <h4>🛫 Origin: {origin_code}</h4>
                <div class="weather-location">
                    <p><strong>Date:</strong> {flight_date}</p>
                    <p><strong>Conditions:</strong> {origin_weather['weather_desc']}</p>
                    <p><strong>Temperature:</strong> {origin_weather['temperature']:.1f}°C</p>
                    <p><strong>Humidity:</strong> {origin_weather['humidity']:.1f}%</p>
                    <p><strong>Wind Speed:</strong> {origin_weather['wind_speed']:.1f} km/h</p>
                    <p><strong>Visibility:</strong> {origin_weather['visibility']:.1f} km</p>
                    <p><strong>Cloud Cover:</strong> {origin_weather['cloudiness']:.1f}%</p>
                    <p><strong>Precipitation:</strong> {origin_weather['precipitation']:.1f} mm</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="weather-card">
                <h4>🛫 Origin: {origin_code}</h4>
                <div class="weather-location">
                    <p>Weather forecast not available</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        if dest_weather:
            st.markdown(f"""
            <div class="weather-card">
                <h4>🛬 Destination: {dest_code}</h4>
                <div class="weather-location">
                    <p><strong>Date:</strong> {flight_date}</p>
                    <p><strong>Conditions:</strong> {dest_weather['weather_desc']}</p>
                    <p><strong>Temperature:</strong> {dest_weather['temperature']:.1f}°C</p>
                    <p><strong>Humidity:</strong> {dest_weather['humidity']:.1f}%</p>
                    <p><strong>Wind Speed:</strong> {dest_weather['wind_speed']:.1f} km/h</p>
                    <p><strong>Visibility:</strong> {dest_weather['visibility']:.1f} km</p>
                    <p><strong>Cloud Cover:</strong> {dest_weather['cloudiness']:.1f}%</p>
                    <p><strong>Precipitation:</strong> {dest_weather['precipitation']:.1f} mm</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="weather-card">
                <h4>🛬 Destination: {dest_code}</h4>
                <div class="weather-location">
                    <p>Weather forecast not available</p>
                </div>
            </div>
            """, unsafe_allow_html=True)


def create_prediction_input(inputs, df, encoders, feature_columns, api_key='KZG5KUC6LL62Z5LHDDZ3TTGVC', airports_df=None):
    """Create input dataframe for prediction and return weather data"""
    try:
        date_obj = datetime.datetime.strptime(inputs['date_str'], "%Y-%m-%d")
    except ValueError:
        return None, "Invalid date format", None, None

    # Get historical data for this route
    if not df.empty:
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

        if route_data.empty:
            route_data = df[df['MONTH'] == date_obj.month]
    else:
        route_data = pd.DataFrame()

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
                input_dict[f'origin_{key}'] = value

        if dest_weather:
            for key, value in dest_weather.items():
                input_dict[f'dest_{key}'] = value

    # Fill missing features with historical averages or defaults
    if not route_data.empty:
        numeric_cols = route_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in feature_columns and col not in input_dict:
                avg_val = route_data[col].mean()
                input_dict[col] = avg_val if not pd.isna(avg_val) else 0

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
    # Create gauge chart for delay
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
        font={'color': "darkblue", 'family': "Arial"},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return fig


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>Flight Delay Predictor</h1>
    </div>
    """, unsafe_allow_html=True)

    # Load model and data
    model, encoders, feature_columns, created_date, version = load_model()

    if model is None:
        st.stop()

    # Load historical data and airports data
    df = load_historical_data()
    airports_df = load_airports_data()

    # Sidebar with model info
    with st.sidebar:
        st.markdown("### Model Information")
        st.markdown(f"**Created:** {created_date}")
        st.markdown(f"**Version:** {version}")
        st.markdown(f"**Features:** {len(feature_columns) if feature_columns is not None else 'Unknown'}")

        st.markdown("### Weather Settings")
        use_weather = st.checkbox("Use Real-time Weather Forecasts", value=True)

        api_key = None
        if use_weather:
            api_key = 'KZG5KUC6LL62Z5LHDDZ3TTGVC'  # Default API key
            # Uncomment below to allow custom API key input
            # api_key = st.text_input(
            #     "Visual Crossing API Key",
            #     type="password",
            #     value='KZG5KUC6LL62Z5LHDDZ3TTGVC',
            #     placeholder="Enter your API key...",
            #     help="Get your free API key from Visual Crossing Weather"
            # )

        st.markdown("### Popular Routes")
        if not df.empty and 'ORIGIN_AIRPORT' in df.columns:
            try:
                popular_routes = df.groupby(['ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']).size().nlargest(5)
                for (origin, dest), count in popular_routes.items():
                    st.markdown(f"**{origin} → {dest}** ({count:,} flights)")
            except Exception as e:
                st.markdown("No popular routes data available")

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("### Flight Details")

        # Flight input form
        with st.form("flight_form"):
            col_a, col_b = st.columns(2)

            with col_a:
                origin = st.text_input(
                    "Origin Airport Code",
                    value="JFK",
                    placeholder="e.g., JFK, LAX, ORD",
                    help="Enter 3-letter airport code"
                ).upper()

                airline = st.selectbox(
                    "Airline",
                    options=["AA", "UA", "DL", "WN", "AS", "B6", "NK", "F9", "G4", "HA"],
                    help="Select airline code"
                )

                departure_time = st.time_input(
                    "Scheduled Departure",
                    value=datetime.time(14, 0),
                    help="Select departure time"
                )

            with col_b:
                dest = st.text_input(
                    "Destination Airport Code",
                    value="LAX",
                    placeholder="e.g., JFK, LAX, ORD",
                    help="Enter 3-letter airport code"
                ).upper()

                flight_date = st.date_input(
                    "Flight Date",
                    value=datetime.date.today() + datetime.timedelta(days=1),
                    help="Select flight date"
                )

                arrival_time = st.time_input(
                    "Scheduled Arrival",
                    value=datetime.time(17, 0),
                    help="Select arrival time"
                )

            submitted = st.form_submit_button("Predict Delay", use_container_width=True)

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
            with st.spinner("Processing prediction..."):
                input_df, status, origin_weather, dest_weather = create_prediction_input(
                    inputs, df, encoders, feature_columns, api_key if use_weather else None, airports_df
                )

                if input_df is None:
                    st.error(f"Error creating prediction input: {status}")
                    st.stop()

                # Make prediction
                try:
                    prediction = model.predict(input_df)[0]

                    # Display results
                    st.markdown("### Prediction Results")

                    # Create visualization
                    fig = create_delay_visualization(prediction, airline, f"{origin} → {dest}")
                    st.plotly_chart(fig, use_container_width=True)

                    # Results summary
                    if prediction > 0:
                        if prediction > 30:
                            result_class = "warning-box"
                            severity = "Major Delay"
                        elif prediction > 15:
                            result_class = "warning-box"
                            severity = "Significant Delay"
                        else:
                            result_class = "info-box"
                            severity = "Minor Delay"

                        st.markdown(f"""
                        <div class="{result_class}">
                            <h4>{severity} Expected</h4>
                            <p><strong>Flight {airline} {origin} → {dest}</strong> is predicted to arrive <strong>{prediction:.1f} minutes late</strong> on {flight_date.strftime('%B %d, %Y')}.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="success-box">
                            <h4>On-Time or Early Arrival</h4>
                            <p><strong>Flight {airline} {origin} → {dest}</strong> is predicted to arrive <strong>{abs(prediction):.1f} minutes early</strong> on {flight_date.strftime('%B %d, %Y')}.</p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Display weather forecast if available
                    if use_weather and (origin_weather or dest_weather):
                        display_weather_forecast(origin_weather, dest_weather, origin, dest, flight_date.strftime('%B %d, %Y'))
                    elif use_weather:
                        st.markdown("""
                        <div class="info-box">
                            <h4>Weather Information</h4>
                            <p>Weather forecasts are not available for the selected airports or date. The prediction uses historical weather patterns.</p>
                        </div>
                        """, unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

    with col2:
        st.markdown("### Tips & Information")

        st.markdown("""
        <div class="metric-card">
            <h4>How to Use</h4>
            <ul>
                <li>Enter valid 3-letter airport codes (e.g., JFK, LAX)</li>
                <li>Select your airline and flight times</li>
                <li>Choose a future date for forecasting</li>
                <li>Enable weather forecasts for better accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h4>Weather Impact</h4>
            <ul>
                <li><strong>High winds:</strong> Can cause departure delays</li>
                <li><strong>Low visibility:</strong> Affects landing procedures</li>
                <li><strong>Precipitation:</strong> Increases taxi and boarding time</li>
                <li><strong>Temperature:</strong> Affects aircraft performance</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="metric-card">
            <h4>Delay Categories</h4>
            <ul>
                <li><span style="color: green;">On-time</span>: Early or &lt;5 min delay</li>
                <li><span style="color: blue;">Minor</span>: 5-15 min delay</li>
                <li><span style="color: orange;">Significant</span>: 15-30 min delay</li>
                <li><span style="color: red;">Major</span>: &gt;30 min delay</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #666;">
        <p>Flight Delay Predictor | Built with Streamlit & Machine Learning</p>
        <p>Powered by XGBoost & Real-time Weather Data</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
    
