import streamlit as st

def apply_custom_css():
    st.markdown("""
    <style>
        .stApp { background-color: #0b0f19; color: #f1f5f9; }
        .stButton>button {
            background: linear-gradient(135deg, #FF4B4B 0%, #FF0000 100%);
            color: white; border: none; border-radius: 6px; font-weight: bold; width: 100%;
        }
        .sidebar .sidebar-content { background-color: #0d1527; }
        .crab-response { background-color: #1e293b; padding: 15px; border-radius: 8px; border-left: 4px solid #FF4B4B; }
    </style>
    """, unsafe_allow_html=True)

def render_sidebar(status, params, loss):
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/crab.png", width=70)
        st.title("⚙️ Engine Telemetry")
        
        if "ONLINE" in status:
            st.success(status)
        else:
            st.error(status)
            
        st.markdown(f"**Architecture:** ~{params}M Params")
        st.markdown(f"**Validation Loss:** {loss}")
        st.markdown("**Creator:** Arshvir (Jaiho Labs)")
        
        st.divider()
        st.subheader("Inference Settings")
        temp = st.slider("Temperature (Creativity)", 0.1, 1.5, 0.6, 0.1)
        max_t = st.slider("Max Tokens", 10, 150, 60, 10)
        return temp, max_t