import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# --- CONFIG & STYLING ---
st.set_page_config(page_title="AeroGrow AI", layout="wide")

# Force a cache clear to ensure we pick up the latest CSV structure
st.cache_data.clear()

def load_data():
    try:
        # Read the CSV
        df = pd.read_csv("irrigation_dataset.csv")
        # Fix the date format warning by explicitly setting errors='coerce'
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        return df.dropna(subset=['timestamp']) 
    except Exception as e:
        st.error(f"Error loading CSV: {e}")
        return None

df = load_data()

if df is not None and not df.empty:
    # Target the very last row for current status
    latest = df.iloc[-1]
    
    # --- HEADER ---
    st.title("🌾 Smart Irrigation Command Center")
    
    # --- TOP ROW METRICS ---
    # Using .get() ensures that if a column is missing, the app shows 0 instead of crashing
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("NDVI (Health)", f"{latest.get('NDVI', 0):.2f}")
    m2.metric("Soil Moisture", f"{latest.get('soil_avg', 0)*100:.1f}%")
    m3.metric("Temperature", f"{latest.get('temp_c', 0)}°C")
    m4.metric("3-Day Rain", f"{latest.get('rain_3d', 0)} mm")

    st.markdown("---")

    # --- AI DECISION LOGIC (SAFE CHECK) ---
    # We check the column names dynamically to avoid KeyError
    is_irrigating = False
    if 'irrigation_prediction' in df.columns:
        is_irrigating = (latest['irrigation_prediction'] == 1)
    
    # Get water amount safely
    water_amt = latest.get('lstm_water_mm', 0.0)

    left_col, right_col = st.columns([2, 1])

    with left_col:
        # Change color based on the decision
        bg_color = "#e74c3c" if is_irrigating else "#2ecc71"
        status_text = "IRRIGATION REQUIRED" if is_irrigating else "CONDITION OPTIMAL"
        
        st.markdown(f"""
            <div style="background-color:{bg_color}; padding:30px; border-radius:15px; text-align:center; color:white;">
                <h1 style="margin:0;">{status_text}</h1>
                <h2 style="margin:0;">Recommended Water: {water_amt:.2f} mm</h2>
            </div>
            """, unsafe_allow_html=True)
            
        if 'irrigation_prediction' not in df.columns:
            st.info("💡 **Note:** AI prediction columns were not found in all rows. Ensure 'run_system.py' has updated the dataset.")

    with right_col:
        # Moisture Gauge
        val = latest.get('soil_avg', 0) * 100
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = val,
            gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#3498db"}},
            title = {'text': "Soil Saturation %"}
        ))
        fig.update_layout(height=250, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

    # --- HISTORICAL CHARTS ---
    st.markdown("### 📊 Field Trends")
    tab1, tab2 = st.tabs(["💧 Moisture History", "🌱 Health Index"])
    
    with tab1:
        st.plotly_chart(px.line(df, x='timestamp', y='soil_avg', title="Soil Moisture Over Time"), use_container_width=True)
    with tab2:
        # Only plot indices if they exist in the columns
        available_indices = [c for c in ['NDVI', 'NDWI'] if c in df.columns]
        if available_indices:
            st.plotly_chart(px.line(df, x='timestamp', y=available_indices, title="Vegetation Trends"), use_container_width=True)

else:
    st.error("The dataset is empty or could not be read. Please run 'run_system.py' first.")