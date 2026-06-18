import streamlit as st
from spacetrack import SpaceTrackClient
import spacetrack.operators as op
from skyfield.api import EarthSatellite, load, wgs84
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone
import math
import bisect
import pandas as pd

# --- Math & Spatial Helpers ---
def euclidean_km(p1, p2) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def parse_tle_string(tle_string, ts):
    """Parses a block of TLE strings into chronologically sorted EarthSatellite objects."""
    lines = tle_string.strip().split('\n')
    entries = []
    for i in range(0, len(lines), 2):
        if i+1 >= len(lines): break
        l1, l2 = lines[i].strip(), lines[i+1].strip()
        try:
            sat = EarthSatellite(l1, l2, "sat", ts)
            epoch_dt = sat.epoch.utc_datetime()
            entries.append((epoch_dt, sat))
        except Exception:
            continue
    entries.sort(key=lambda x: x[0])
    return entries

def best_tle(entries: list, t_dt: datetime, max_age_days=7):
    """Finds the closest TLE in time for accurate relative propagation."""
    if not entries:
        return None
    epochs = [e[0] for e in entries]
    idx = bisect.bisect_left(epochs, t_dt)
    if idx == 0:
        cand_epoch, cand_sat = entries[0]
    elif idx >= len(entries):
        cand_epoch, cand_sat = entries[-1]
    else:
        be, bs = entries[idx - 1]
        ae, as_ = entries[idx]
        if abs((ae - t_dt).total_seconds()) < abs((be - t_dt).total_seconds()):
            cand_epoch, cand_sat = ae, as_
        else:
            cand_epoch, cand_sat = be, bs
    age_days = abs((t_dt - cand_epoch).total_seconds()) / 86400.0
    return None if age_days > max_age_days else cand_sat

# --- Website Page Config ---
st.set_page_config(page_title="Orbital Tracker", layout="wide")

st.title("🛰️ Satellite TLE Data Explorer")
st.markdown("Extract historical orbital elements and calculate 3D proximity directly from Space-Track.org")

# --- Initialize Session State ---
if 'data_ready' not in st.session_state:
    st.session_state['data_ready'] = False

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
st.sidebar.info("Enter your Space-Track.org credentials below.")
ST_USER = st.sidebar.text_input("Username (Email)")
ST_PASS = st.sidebar.text_input("Password", type="password")

st.sidebar.markdown("---")
st.sidebar.header("Tracking Parameters")

orbit_regime = st.sidebar.radio("Orbit Regime", ["GEO (Geosynchronous)", "LEO (Low Earth Orbit)"])

if "GEO" in orbit_regime:
    default_cands = "41838, 43874, 50321"
    default_ref = "37606"
else:
    default_cands = "43013, 48274, 40069"
    default_ref = "25544"

sat_input = st.sidebar.text_input("Candidate NORAD IDs (comma separated)", value=default_cands)
ref_sat_input = st.sidebar.text_input("Reference Satellite NORAD ID", value=default_ref)

min_allowed_date = datetime(2003, 1, 1).date()
max_allowed_date = datetime.now().date()

start_date = st.sidebar.date_input("Start Date", value=max_allowed_date - timedelta(days=7), min_value=min_allowed_date, max_value=max_allowed_date)
end_date = st.sidebar.date_input("End Date", value=max_allowed_date, min_value=min_allowed_date, max_value=max_allowed_date)

run_button = st.sidebar.button("Fetch & Generate Graphs")

# --- Backend Logic (Data Fetching) ---
if run_button:
    if not ST_USER or not ST_PASS:
        st.error("Please enter your Space-Track credentials in the sidebar.")
    else:
        try:
            with st.spinner("Fetching data from Space-Track..."):
                st_client = SpaceTrackClient(identity=ST_USER, password=ST_PASS)
                sat_list = [s.strip() for s in sat_input.split(",") if s.strip()]
                ref_id = ref_sat_input.strip()
                drange = op.inclusive_range(start_date, end_date)
                
                tle_data = st_client.gp_history(norad_cat_id=sat_list, epoch=drange, format='tle')
                ref_tle_data = st_client.gp_history(norad_cat_id=ref_id, epoch=drange, format='tle')

            if not tle_data or not ref_tle_data:
                st.warning("Insufficient data found for these satellites in the selected range.")
            else:
                ts = load.timescale()
                ref_tles = parse_tle_string(ref_tle_data, ts)

                plot_data = {sat: {'epoch': [], 'inc': [], 'raan': [], 'ecc': [], 'arg_pe': [], 'mean_anom': [], 'mean_mo': [], 'lon': [], 'dist': []} for sat in sat_list}

                lines = tle_data.strip().split('\n')
                for i in range(0, len(lines), 2):
                    if i+1 >= len(lines): break
                    l1, l2 = lines[i].strip(), lines[i+1].strip()
                    nid = str(int(l1[2:7]))
                    if nid not in plot_data: continue

                    sat_obj = EarthSatellite(l1, l2, nid, ts)
                    t = sat_obj.epoch
                    t_dt = t.utc_datetime()

                    plot_data[nid]['epoch'].append(t_dt)
                    plot_data[nid]['inc'].append(math.degrees(sat_obj.model.inclo))
                    plot_data[nid]['raan'].append(math.degrees(sat_obj.model.nodeo))
                    plot_data[nid]['ecc'].append(sat_obj.model.ecco)
                    plot_data[nid]['arg_pe'].append(math.degrees(sat_obj.model.argpo))
                    plot_data[nid]['mean_anom'].append(math.degrees(sat_obj.model.mo))
                    plot_data[nid]['mean_mo'].append(sat_obj.model.no_kozai * 1440 / (2 * math.pi))
                    plot_data[nid]['lon'].append(sat_obj.at(t).subpoint().longitude.degrees)

                    ref_sat_best = best_tle(ref_tles, t_dt)
                    if ref_sat_best:
                        plot_data[nid]['dist'].append(euclidean_km(tuple(sat_obj.at(t).position.km), tuple(ref_sat_best.at(t).position.km)))
                    else:
                        plot_data[nid]['dist'].append(None)

                # Compile CSV Data
                all_sats_data = []
                for sat in sat_list:
                    if plot_data[sat]['epoch']:
                        df_sat = pd.DataFrame(plot_data[sat])
                        df_sat['NORAD_ID'] = sat
                        all_sats_data.append(df_sat)

                csv_data = None
                if all_sats_data:
                    combined_df = pd.concat(all_sats_data, ignore_index=True)
                    cols = ['NORAD_ID', 'epoch', 'dist', 'lon', 'inc', 'raan', 'ecc', 'arg_pe', 'mean_anom', 'mean_mo']
                    csv_data = combined_df[cols].to_csv(index=False).encode('utf-8')

                # Save everything to Session State
                st.session_state['plot_data'] = plot_data
                st.session_state['sat_list'] = sat_list
                st.session_state['ref_id'] = ref_id
                st.session_state['csv_data'] = csv_data
                st.session_state['slider_min'] = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
                st.session_state['slider_max'] = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc)
                st.session_state['data_ready'] = True

        except Exception as e:
            st.error(f"Error: {e}")

# --- UI Rendering (Displays only when data is in state) ---
if st.session_state.get('data_ready'):
    st.markdown("### 🔎 Interactive Zoom Control")
    
    # 1. Native Streamlit Slider
    selected_range = st.slider(
        "Drag the handles to zoom in on a specific timeframe:",
        min_value=st.session_state['slider_min'],
        max_value=st.session_state['slider_max'],
        value=(st.session_state['slider_min'], st.session_state['slider_max']),
        format="YYYY-MM-DD",
        label_visibility="collapsed"
    )

    plot_data = st.session_state['plot_data']
    sat_list = st.session_state['sat_list']
    ref_id = st.session_state['ref_id']
    csv_data = st.session_state['csv_data']

    # 2. Build Plotly Visuals (Reduced to 8 Rows)
    fig = make_subplots(
        rows=8, cols=1, shared_xaxes=True, vertical_spacing=0.03,
        subplot_titles=(
            "Inclination (°)", "RAAN (°)", "Eccentricity", "Arg of Perigee (°)", 
            "Mean Anomaly (°)", "Mean Motion", "Longitude (°)", 
            f"3D Distance to {ref_id} (km)"
        )
    )

    sat_colors = ['#D62728', '#1F77B4', '#2CA02C', '#FF7F0E', '#9467BD', '#17BECF', '#E377C2', '#BCBD22', '#8C564B', '#FF9896']

    for idx, sat in enumerate(sat_list):
        if not plot_data[sat]['epoch']: continue
        current_color = sat_colors[idx % len(sat_colors)]
        params = [('inc', 1), ('raan', 2), ('ecc', 3), ('arg_pe', 4), ('mean_anom', 5), ('mean_mo', 6), ('lon', 7), ('dist', 8)]
        
        for p_key, row in params:
            if p_key == 'dist' and all(v is None for v in plot_data[sat][p_key]): continue
            fig.add_trace(go.Scatter(
                x=plot_data[sat]['epoch'], y=plot_data[sat][p_key],
                name=f"Sat {sat}", legendgroup=f"group_{sat}", showlegend=(True if row == 1 else False),
                mode='lines+markers', line=dict(color=current_color), marker=dict(color=current_color)    
            ), row=row, col=1)

    fig.update_layout(height=1800, hovermode="x unified", template="plotly_dark", margin=dict(t=80, b=50, l=50, r=50))
    
    # 3. Apply the Streamlit slider boundaries to the Plotly graph dynamically
    fig.update_xaxes(
        range=[selected_range[0], selected_range[1]], 
        showline=True, linewidth=1, linecolor='gray', mirror=True
    )
    
    # Rangeslider removed completely.
    fig.update_yaxes(showline=True, linewidth=1, linecolor='gray', mirror=True)
    fig.update_annotations(yshift=15) 
    
    st.plotly_chart(fig, use_container_width=True)

    # 4. CSV Download Button
    if csv_data:
        st.download_button(
            label="📥 Download Orbital Data (CSV)",
            data=csv_data,
            file_name=f"orbital_data_export.csv",
            mime="text/csv",
        )
