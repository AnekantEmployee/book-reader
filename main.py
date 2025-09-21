import os
import time
import json
import threading
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
from datetime import datetime

# Streamlit Page Config
st.set_page_config(page_title="Indore Weather", page_icon="🌤️", layout="wide")

class WeatherApp:
    def __init__(self):
        self.weather_data = []
        self.is_running = False
        self.thread = None

    def get_weather_description(self, weather_code):
        """Convert weather code to description"""
        weather_codes = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Fog",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail"
        }
        return weather_codes.get(weather_code, "Unknown")

    def fetch_weather(self):
        try:
            # Enhanced Open-Meteo API call with more parameters
            url = "https://api.open-meteo.com/v1/forecast"
            params = {
                "latitude": 22.7196,  # Indore
                "longitude": 75.8577,
                "current": [
                    "temperature_2m",
                    "relative_humidity_2m", 
                    "precipitation",
                    "weather_code",
                    "wind_speed_10m",
                    "wind_direction_10m",
                    "is_day"
                ],
                "timezone": "Asia/Kolkata"
            }
            response = requests.get(url, params=params)
            data = response.json()

            current = data.get("current", {})

            entry = {
                "timestamp": datetime.now(),
                "temperature": current.get("temperature_2m"),
                "humidity": current.get("relative_humidity_2m"),
                "precipitation": current.get("precipitation"),
                "windspeed": current.get("wind_speed_10m"),
                "winddirection": current.get("wind_direction_10m"),
                "weathercode": current.get("weather_code"),
                "weather_description": self.get_weather_description(current.get("weather_code")),
                "is_day": current.get("is_day"),
            }
            return entry
        except Exception as e:
            return {
                "timestamp": datetime.now(),
                "temperature": None,
                "humidity": None,
                "precipitation": None,
                "windspeed": None,
                "winddirection": None,
                "weathercode": None,
                "weather_description": "Error",
                "is_day": None,
                "error": str(e)
            }

    def run_loop(self, interval):
        while self.is_running:
            entry = self.fetch_weather()
            self.weather_data.append(entry)
            if len(self.weather_data) > 100:  # keep last 100 records
                self.weather_data = self.weather_data[-100:]
            time.sleep(interval * 60)

    def start(self, interval):
        if not self.is_running:
            self.is_running = True
            self.thread = threading.Thread(target=self.run_loop, args=(interval,), daemon=True)
            self.thread.start()

    def stop(self):
        self.is_running = False


# Init in session
if "weather_app" not in st.session_state:
    st.session_state.weather_app = WeatherApp()

app = st.session_state.weather_app

# Sidebar controls
st.sidebar.button("🔄 Refresh Page")
st.sidebar.markdown("---")
st.sidebar.markdown("### Controls")

st.title("🌤️ Indore Weather Monitor")
st.markdown("*Real-time weather data from Open-Meteo API*")

interval = st.slider("Update Interval (minutes)", 1, 60, 5)

col1, col2 = st.columns(2)

if not app.is_running:
    with col1:
        if st.button("🚀 Start Monitoring", type="primary"):
            app.start(interval)
            st.success("Monitoring started!")
            st.rerun()
else:
    with col2:
        if st.button("⏹️ Stop Monitoring", type="secondary"):
            app.stop()
            st.warning("Monitoring stopped")
            st.rerun()

status = "🟢 Running" if app.is_running else "🔴 Stopped"
st.write(f"**Status:** {status}")

# Display weather data
if app.weather_data:
    df = pd.DataFrame(app.weather_data)
    latest = df.iloc[-1]
    
    # Current Weather Overview
    st.markdown("## 📍 Current Weather in Indore")
    
    # Main metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        temp_val = f"{latest['temperature']:.1f}°C" if latest['temperature'] else "N/A"
        st.metric("🌡️ Temperature", temp_val)
    
    with col2:
        humidity_val = f"{latest['humidity']:.0f}%" if latest['humidity'] else "N/A"
        st.metric("💧 Humidity", humidity_val)
    
    with col3:
        precip_val = f"{latest['precipitation']:.1f} mm" if latest['precipitation'] else "0 mm"
        st.metric("🌧️ Precipitation", precip_val)
    
    with col4:
        wind_val = f"{latest['windspeed']:.1f} km/h" if latest['windspeed'] else "N/A"
        st.metric("💨 Wind Speed", wind_val)

    # Weather description and time
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(f"☁️ {latest['weather_description']}")
    with col2:
        current_time = datetime.now().strftime("%A, %I:%M %p")
        st.write(f"**{current_time}**")

    # Charts
    st.markdown("## 📊 Weather Trends")
    
    if len(df) > 1:
        # Create subplots for different metrics
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Temperature (°C)', 'Humidity (%)', 'Wind Speed (km/h)', 'Precipitation (mm)'),
            vertical_spacing=0.25,  # Increased vertical spacing between top and bottom rows
            horizontal_spacing=0.1  # Added horizontal spacing between left and right columns
        )

        # Temperature
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['temperature'], 
                      name='Temperature', line=dict(color='red', width=2)),
            row=1, col=1
        )

        # Humidity
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['humidity'], 
                      name='Humidity', line=dict(color='blue', width=2)),
            row=1, col=2
        )

        # Wind Speed
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['windspeed'], 
                      name='Wind Speed', line=dict(color='green', width=2)),
            row=2, col=1
        )

        # Precipitation
        fig.add_trace(
            go.Scatter(x=df['timestamp'], y=df['precipitation'], 
                      name='Precipitation', line=dict(color='purple', width=2),
                      fill='tonexty'),
            row=2, col=2
        )

        fig.update_layout(height=700, showlegend=False, title_text="Weather Data Over Time")
        fig.update_xaxes(title_text="Time")
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        # Single data point - show simple metrics
        st.info("Collecting data... Charts will appear after multiple readings.")

    # Data Table (without raw JSON)
    st.markdown("## 📋 Recent Weather Records")
    
    # Prepare display dataframe
    df_display = df.copy()
    df_display["Time"] = df_display["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df_display["Temp (°C)"] = df_display["temperature"].round(1)
    df_display["Humidity (%)"] = df_display["humidity"].round(0)
    df_display["Precipitation (mm)"] = df_display["precipitation"].round(1)
    df_display["Wind (km/h)"] = df_display["windspeed"].round(1)
    df_display["Wind Dir (°)"] = df_display["winddirection"].round(0)
    df_display["Condition"] = df_display["weather_description"]
    
    # Select columns for display
    display_cols = ["Time", "Temp (°C)", "Humidity (%)", "Precipitation (mm)", 
                   "Wind (km/h)", "Wind Dir (°)", "Condition"]
    
    st.dataframe(
        df_display[display_cols].tail(10), 
        use_container_width=True,
        hide_index=True
    )
    
    # Summary statistics
    if len(df) > 1:
        st.markdown("## 📈 Session Summary")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_temp = df['temperature'].mean()
            st.metric("Avg Temperature", f"{avg_temp:.1f}°C" if not pd.isna(avg_temp) else "N/A")
        
        with col2:
            avg_humidity = df['humidity'].mean()
            st.metric("Avg Humidity", f"{avg_humidity:.0f}%" if not pd.isna(avg_humidity) else "N/A")
        
        with col3:
            max_wind = df['windspeed'].max()
            st.metric("Max Wind Speed", f"{max_wind:.1f} km/h" if not pd.isna(max_wind) else "N/A")
        
        with col4:
            total_precip = df['precipitation'].sum()
            st.metric("Total Precipitation", f"{total_precip:.1f} mm" if not pd.isna(total_precip) else "N/A")

else:
    st.info("🎯 No weather data yet. Click **Start Monitoring** to begin collecting real-time weather information for Indore.")
    
    # Show sample layout
    st.markdown("### What you'll see:")
    st.markdown("""
    - 🌡️ **Temperature** in Celsius
    - 💧 **Humidity** percentage  
    - 🌧️ **Precipitation** in millimeters
    - 💨 **Wind Speed** and direction
    - ☁️ **Weather conditions** (Clear, Cloudy, Rain, etc.)
    - 📊 **Interactive charts** showing trends over time
    - 📋 **Data table** with recent readings
    """)