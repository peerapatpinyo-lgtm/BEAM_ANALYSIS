import streamlit as st
import numpy as np
import plotly.graph_objects as go

# --- IMPORT CUSTOM ENGINE ---
# ต้องแน่ใจว่าไฟล์ beam_engine.py อยู่โฟลเดอร์เดียวกัน
try:
    from beam_engine import SimpleBeamSolver
except ImportError:
    st.error("ไม่พบไฟล์ 'beam_engine.py' กรุณาวางไฟล์ engine ไว้ในโฟลเดอร์เดียวกัน")
    st.stop()

# ==========================================
# 🎨 UI & PLOTTING FUNCTIONS
# ==========================================
st.set_page_config(page_title="Beam Pro V7", layout="wide")

def add_peak_labels(fig, x_data, y_data, inverted=False):
    """Helper to add Max/Min labels on Plotly graph"""
    # Find Max Positive
    max_idx = np.argmax(y_data)
    max_val = y_data[max_idx]
    max_x = x_data[max_idx]
    
    # Find Max Negative
    min_idx = np.argmin(y_data)
    min_val = y_data[min_idx]
    min_x = x_data[min_idx]

    # Style configuration
    font_style = dict(color="black", size=11, family="Arial Black")
    bg_color = "rgba(255,255,255,0.7)"

    # Add Label for Max
    fig.add_annotation(
        x=max_x, y=max_val, text=f"{max_val:.2f}",
        showarrow=True, arrowhead=1, yshift=10 if not inverted else -10,
        font=font_style, bgcolor=bg_color
    )
    # Add Label for Min
    fig.add_annotation(
        x=min_x, y=min_val, text=f"{min_val:.2f}",
        showarrow=True, arrowhead=1, yshift=-10 if not inverted else 10,
        font=font_style, bgcolor=bg_color
    )

# ==========================================
# 🖥️ MAIN APPLICATION
# ==========================================
st.title("🏗️ Continuous Beam Analysis (Separated Engine)")
st.caption("Architecture: Frontend (app.py) + Backend (beam_engine.py)")

# --- INPUT SECTION ---
col1, col2 = st.columns([1, 2])

with col1:
    n_span = st.number_input("Number of Spans", 1, 6, 2)
    spans = []
    st.subheader("Geometry (m)")
    for i in range(n_span):
        spans.append(st.number_input(f"Span {i+1}", 1.0, 20.0, 4.0, key=f"s_{i}"))

with col2:
    st.subheader("Supports")
    cols = st.columns(n_span + 1)
    opts = ['Pin', 'Roller', 'Fix']
    supports = [cols[i].selectbox(f"S{i+1}", opts, index=0 if i==0 else 1, key=f"sup_{i}") for i in range(n_span+1)]
        
    st.subheader("Loads (Factored: 1.4DL + 1.7LL)")
    loads_input = []
    
    for i in range(n_span):
        with st.expander(f"📍 Loads on Span {i+1}", expanded=True):
            c_u, c_p = st.columns([1, 1.5])
            
            # 1. Uniform Load
            with c_u:
                st.markdown("**Uniform Load**")
                u_dl = st.number_input("DL (kN/m)", 0.0, key=f"udl_{i}")
                u_ll = st.number_input("LL (kN/m)", 0.0, key=f"ull_{i}")
                if (u_dl + u_ll) > 0:
                    loads_input.append({
                        'span_idx': i, 'type': 'Uniform', 
                        'total_w': (1.4*u_dl + 1.7*u_ll)*1000 # Convert to N
                    })

            # 2. Multiple Point Loads
            with c_p:
                st.markdown("**Concentrated Loads**")
                num_pt = st.number_input(f"Qty Point Loads", 0, 10, 0, key=f"num_pt_{i}")
                
                if num_pt > 0:
                    for j in range(num_pt):
                        st.caption(f"--- Point Load #{j+1} ---")
                        cc1, cc2, cc3 = st.columns(3)
                        p_dl = cc1.number_input(f"DL (kN)", 0.0, key=f"pdl_{i}_{j}")
                        p_ll = cc2.number_input(f"LL (kN)", 0.0, key=f"pll_{i}_{j}")
                        p_pos = cc3.number_input(f"Pos (m)", 0.0, spans[i], spans[i]/2, key=f"ppos_{i}_{j}")
                        
                        if (p_dl + p_ll) > 0:
                            loads_input.append({
                                'span_idx': i, 'type': 'Point',
                                'total_w': (1.4*p_dl + 1.7*p_ll)*1000, # Convert to N
                                'pos': p_pos
                            })

if st.button("🚀 Calculate", type="primary"):
    # เรียกใช้ Engine จากไฟล์ภายนอก
    solver = SimpleBeamSolver(spans, supports, loads_input)
    u, err = solver.solve()
    
    if err:
        st.error(f"Analysis Error: {err}")
    else:
        # Get Results DataFrame
        df = solver.get_internal_forces(num_points=100)
        
        m_max = df['moment'].max()
        m_min = df['moment'].min()
        v_max = df['shear'].abs().max()
        
        # Display Key Metrics
        c1, c2, c3 = st.columns(3)
        c1.metric("Max Moment (+)", f"{m_max:.2f} kN-m")
        c2.metric("Max Moment (-)", f"{m_min:.2f} kN-m")
        c3.metric("Max Shear", f"{v_max:.2f} kN")
        
        # --- Plot SFD ---
        fig_v = go.Figure()
        fig_v.add_trace(go.Scatter(x=df['x'], y=df['shear'], fill='tozeroy', line=dict(color='#D32F2F'), name='Shear'))
        add_peak_labels(fig_v, df['x'].values, df['shear'].values)
        fig_v.update_layout(title="Shear Force Diagram (SFD)", yaxis_title="Shear (kN)", hovermode="x")
        st.plotly_chart(fig_v, use_container_width=True)
        
        # --- Plot BMD (Tension Side) ---
        fig_m = go.Figure()
        fig_m.add_trace(go.Scatter(x=df['x'], y=df['moment'], fill='tozeroy', line=dict(color='#1976D2'), name='Moment'))
        add_peak_labels(fig_m, df['x'].values, df['moment'].values, inverted=True)
        fig_m.update_layout(
            title="Bending Moment Diagram (BMD) - Tension Side", 
            yaxis_title="Moment (kN-m)",
            yaxis=dict(autorange="reversed"),
            hovermode="x"
        )
        st.plotly_chart(fig_m, use_container_width=True)

        # --- Quick Design Check ---
        st.divider()
        st.subheader("📝 Design Info")
        des_mu = max(abs(m_max), abs(m_min))
        fc, fy = 24, 400
        b, h, cov = 25, 50, 4
        d = (h-cov)/100
        
        # Simplified RC Calc
        try:
            Rn = (des_mu * 1000 / 0.9) / ((b/100)*d**2)
            m_rat = fy/(0.85*fc)
            term = 1 - (2*m_rat*Rn)/(fy*1e6)
            if term < 0:
                st.error("❌ Section too small (Compression Failure)")
            else:
                rho = (1/m_rat)*(1 - np.sqrt(term))
                as_req = rho*b*d*100
                st.success(f"✅ Section OK | Design Mu: {des_mu:.2f} kN-m | Required As ≈ {as_req:.2f} cm²")
        except Exception as e:
            st.warning(f"Design calculation error: {e}")
