import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import streamlit as st

# ==========================================
# 1. HELPER: LOAD TABLE & DETAILED CALCULATION
# ==========================================
def render_load_table(params, raw_loads_df=None):
    """
    แสดงรายการคำนวณ Load Analysis แบบละเอียด (Fixed Unit Calculation)
    """
    st.markdown("### 📑 Detailed Load Analysis Report")
    
    # 1. Geometry Check
    st.markdown("#### 1. Geometry & Parameters Check")
    
    # รับค่าและแปลงหน่วย (Input Handler ส่งมาเป็น mm)
    raw_b = params.get('b', params.get('width', 300))
    raw_h = params.get('h', params.get('depth', 500))
    
    # แปลง mm -> m
    b_m = float(raw_b) / 1000.0
    h_m = float(raw_h) / 1000.0
    
    conc_density = 2400  # kg/m3
    g = 9.81             # m/s2
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Width (b)", f"{raw_b:.0f} mm", f"{b_m:.2f} m")
    c2.metric("Depth (h)", f"{raw_h:.0f} mm", f"{h_m:.2f} m")
    c3.metric("Conc. Density", "2400 kg/m³")
    c4.metric("Gravity (g)", "9.81 m/s²")
    
    st.divider()

    # 2. Self-Weight Calculation
    st.markdown("#### 2. Self-Weight Calculation ($w_{sw}$)")
    inc_sw = params.get('include_sw', True)
    
    st.markdown("**1️⃣ Formula:**")
    st.latex(r"w_{sw} = b \times h \times \rho_{conc} \times g")
    
    st.markdown("**2️⃣ Substitution:**")
    st.markdown(f"$$ w_{{sw}} = {b_m:.2f} \\times {h_m:.2f} \\times 2400 \\times 9.81 $$")
    
    # คำนวณ N/m แล้วแปลงเป็น kN/m
    val_N_m = b_m * h_m * conc_density * g
    val_kN_m = val_N_m / 1000.0
    
    st.markdown("**3️⃣ Result:**")
    st.markdown(f"$$ = {val_N_m:.2f} \\text{{ N/m}} \\Rightarrow \\mathbf{{{val_kN_m:.3f} \\text{{ kN/m}}}} $$")
    
    if not inc_sw:
        st.error("❌ Self-Weight is DISABLED (Not added to DL)")
        val_kN_m = 0.0

    st.divider()

    # 3. Ultimate Load Equation
    st.markdown("#### 3. Ultimate Design Load ($U$)")
    dl_f = params.get('dl_factor', 1.4)
    ll_f = params.get('ll_factor', 1.7)
    
    st.latex(r"U = " + f"{dl_f}" + r" \times (DL_{user} + " + f"{val_kN_m:.3f}" + r") + " + f"{ll_f}" + r" \times LL_{user}")

    if raw_loads_df is not None and not raw_loads_df.empty:
        with st.expander("Show User Input Loads"):
            st.dataframe(raw_loads_df, use_container_width=True)
            
    st.divider()

# ==========================================
# 2. BOQ CALCULATION
# ==========================================
def calculate_boq_summary(design_res, spans):
    # (คงเดิม - ไม่มีการเปลี่ยนแปลงส่วนนี้)
    total_concrete_vol = 0.0
    total_formwork_area = 0.0
    total_steel_weight = 0.0
    
    for i, res in enumerate(design_res):
        L = spans[i]
        b_m = (res.get('b') or 300) / 1000.0
        h_m = (res.get('h') or 500) / 1000.0
        
        total_concrete_vol += b_m * h_m * L
        total_formwork_area += (2 * h_m + b_m) * L
        
        w_span = 0.0
        def calc_w(n, db, length): return n * (db**2 / 162) * length if n > 0 else 0

        if 'top' in res and 'all_layers' in res['top']:
             for l in res['top']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)
        if 'bot' in res and 'all_layers' in res['bot']:
             for l in res['bot']['all_layers']: w_span += calc_w(l['n'], l['db'], L * 1.1)

        stir_db = res.get('shear', {}).get('db', 6)
        stir_s = (res.get('shear', {}).get('s', 200)) / 1000.0
        if stir_s > 0:
            n_stir = int(L / stir_s) + 1
            len_stir = 2 * (b_m + h_m)
            w_span += n_stir * (stir_db**2 / 162) * len_stir
            
        total_steel_weight += w_span

    return pd.DataFrame([
        {"Item": "Concrete (240 ksc)", "Quantity": float(f"{total_concrete_vol:.2f}"), "Unit": "m³"},
        {"Item": "Formwork", "Quantity": float(f"{total_formwork_area:.2f}"), "Unit": "m²"},
        {"Item": "Rebar (DB+RB)", "Quantity": float(f"{total_steel_weight:.2f}"), "Unit": "kg"}
    ])

# ==========================================
# 3. PLOTLY ANALYSIS GRAPH (FIXED UNITS)
# ==========================================
def plot_analysis_results(res_df, spans, supports, loads, reactions):
    """
    Standard Engineering FBD with Smart Unit Detection
    """
    # แปลง Loads เป็น List
    if isinstance(loads, pd.DataFrame):
        load_list = loads.to_dict('records')
    elif isinstance(loads, list):
        load_list = loads
    else:
        load_list = []

    # --- Smart Unit Detection Logic ---
    # ถ้าค่า mag > 500 ให้เดาว่าเป็น N -> หาร 1000 เพื่อโชว์ kN
    # ถ้าค่า mag <= 500 ให้เดาว่าเป็น kN -> โชว์ค่าเดิม
    processed_loads = []
    max_val_for_scale = 1.0
    
    for l in load_list:
        raw_mag = float(l['mag'])
        # Logic แก้ปัญหาค่าเกิน 1000:
        # ถ้าค่าเกิน 1000 (เช่น 5000) -> หาร 1000 = 5 kN
        # ถ้าค่าน้อย (เช่น 10) -> ใช้ 10 kN เลย
        if abs(raw_mag) > 1000: 
            display_mag = raw_mag / 1000.0
        else:
            display_mag = raw_mag
            
        l_copy = l.copy()
        l_copy['display_mag'] = display_mag
        processed_loads.append(l_copy)
        
        if abs(display_mag) > max_val_for_scale:
            max_val_for_scale = abs(display_mag)

    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=("<b>1. Free Body Diagram (FBD)</b>", "<b>2. Shear Force Diagram (SFD)</b>", "<b>3. Bending Moment Diagram (BMD)</b>", "<b>4. Deflection Diagram</b>"),
        row_heights=[0.30, 0.24, 0.24, 0.22]
    )

    total_L = sum(spans)
    cum_dist = [0] + list(np.cumsum(spans))
    
    # 1. Beam Line
    fig.add_trace(go.Scatter(x=[0, total_L], y=[0, 0], mode='lines', line=dict(color='black', width=5), hoverinfo='skip'), row=1, col=1)
    
    # 2. Supports
    for idx, row in supports.iterrows():
        sym = "square" if row['type'] == 'Fixed' else ("circle" if row['type'] == 'Roller' else "triangle-up")
        fig.add_trace(go.Scatter(
            x=[row['x']], y=[-0.04], mode='markers+text',
            marker=dict(symbol=sym, size=10, color='white', line=dict(width=1.5, color='black')),
            text=[row['type'][0]], textposition="bottom center", hoverinfo='name', name="Support"
        ), row=1, col=1)

    # 3. Loads Drawing
    UDL_MIN_H, UDL_MAX_H = 0.25, 0.55
    P_MIN_H, P_MAX_H = 0.70, 1.30
    ARROW_TIP_OFFSET = 0.08
    
    for l in processed_loads:
        d_mag = l['display_mag'] # ใช้ค่าที่ปรับหน่วยแล้ว
        
        # --- Draw UDL ---
        if l['type'] == 'U':
            span_idx = int(l['span_index'])
            start_x = cum_dist[span_idx] + float(l.get('d_start', 0))
            end_x = start_x + float(l['dist'])
            
            ratio = abs(d_mag) / max_val_for_scale if max_val_for_scale > 0 else 0.5
            h_visual = UDL_MIN_H + (ratio * (UDL_MAX_H - UDL_MIN_H))
            color = '#e74c3c' if l.get('case') == 'LL' else '#2980b9'
            
            # Area
            fig.add_trace(go.Scatter(x=[start_x, end_x, end_x, start_x], y=[0, 0, h_visual, h_visual], fill='toself', fillcolor=color, opacity=0.12, line=dict(width=0), hoverinfo='skip', showlegend=False), row=1, col=1)
            # Top Line
            fig.add_trace(go.Scatter(x=[start_x, end_x], y=[h_visual, h_visual], mode='lines', line=dict(color=color, width=1, dash='dot'), hoverinfo='skip'), row=1, col=1)
            
            # Label (ใช้ d_mag ที่ปรับหน่วยแล้ว)
            label_txt = f"w={d_mag:.2f} kN/m"
            fig.add_annotation(x=(start_x+end_x)/2, y=h_visual, text=label_txt, showarrow=False, yshift=8, font=dict(color=color, size=9), row=1, col=1)
            
            # Arrows
            n_arrows = max(3, int(float(l['dist']) * 1.8))
            for ax_x in np.linspace(start_x, end_x, n_arrows + 2)[1:-1]:
                 fig.add_annotation(x=ax_x, y=ARROW_TIP_OFFSET, ax=ax_x, ay=h_visual, axref='x', ayref='y', xref='x', yref='y', showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1, arrowcolor=color, row=1, col=1)

        # --- Draw Point Load ---
        elif l['type'] == 'P':
            span_idx = int(l['span_index'])
            x_loc = cum_dist[span_idx] + float(l['d_start'])
            
            ratio = abs(d_mag) / max_val_for_scale if max_val_for_scale > 0 else 0.5
            h_arrow = P_MIN_H + (ratio * (P_MAX_H - P_MIN_H))
            color = '#c0392b' if l.get('case') == 'LL' else '#2980b9'
            
            fig.add_annotation(
                x=x_loc, y=ARROW_TIP_OFFSET, ax=x_loc, ay=h_arrow, xref='x', yref='y', axref='x', ayref='y',
                showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2.0, arrowcolor=color,
                text=f"<b>P={d_mag:.2f} kN</b>", xanchor='center', yanchor='bottom', yshift=5, font=dict(color=color, size=11, family="Arial"), row=1, col=1
            )

    # 4. Results (SFD / BMD / Deflection)
    # Solver Output (res_df) is ALWAYS in N and Nm -> Must divide by 1000
    
    # SFD
    fig.add_hline(y=0, line_color="black", line_width=1, row=2, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['shear']/1000.0, mode='lines', line=dict(color='#e74c3c', width=2), fill='tozeroy', fillcolor='rgba(231, 76, 60, 0.1)'), row=2, col=1)
    
    # Annotate SFD
    v_vals = res_df['shear']/1000.0
    for val in [v_vals.max(), v_vals.min()]:
        if abs(val) > 0.01:
            idx = (v_vals - val).abs().idxmin()
            fig.add_annotation(x=res_df['x'].iloc[idx], y=val, text=f"<b>{val:.2f}</b>", showarrow=False, yshift=10 if val>0 else -10, font=dict(color='#e74c3c', size=11), row=2, col=1)

    # BMD
    fig.add_hline(y=0, line_color="black", line_width=1, row=3, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['moment']/1000.0, mode='lines', line=dict(color='#27ae60', width=2), fill='tozeroy', fillcolor='rgba(39, 174, 96, 0.1)'), row=3, col=1)
    
    # Annotate BMD
    m_vals = res_df['moment']/1000.0
    for val in [m_vals.max(), m_vals.min()]:
        if abs(val) > 0.01:
            idx = (m_vals - val).abs().idxmin()
            fig.add_annotation(x=res_df['x'].iloc[idx], y=val, text=f"<b>{val:.2f}</b>", showarrow=True, arrowhead=1, ay=20 if val>0 else -20, font=dict(color='#27ae60', size=11), row=3, col=1)

    # Deflection
    fig.add_hline(y=0, line_color="black", line_width=1, row=4, col=1)
    fig.add_trace(go.Scatter(x=res_df['x'], y=res_df['deflection'], mode='lines', line=dict(color='#8e44ad', width=2)), row=4, col=1)
    if not res_df['deflection'].empty:
        idx_max = res_df['deflection'].abs().idxmax()
        val_max = res_df['deflection'].iloc[idx_max]
        if abs(val_max) > 0.001:
             fig.add_annotation(x=res_df['x'].iloc[idx_max], y=val_max, text=f"<b>Max: {val_max:.2f} mm</b>", showarrow=True, arrowhead=1, ay=30 if val_max < 0 else -30, font=dict(color='#8e44ad', size=11), row=4, col=1)

    # Grid & Layout
    for x_pos in cum_dist:
        fig.add_vline(x=x_pos, line_width=1, line_dash="dash", line_color="gray", opacity=0.3)

    fig.update_layout(height=1100, showlegend=False, template="plotly_white", hovermode="x unified", margin=dict(t=50, b=40, l=60, r=20))
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_yaxes(title_text="Shear (kN)", showgrid=True, row=2, col=1)
    fig.update_yaxes(title_text="Moment (kNm)", autorange="reversed", showgrid=True, row=3, col=1)
    fig.update_yaxes(title_text="Def. (mm)", showgrid=True, row=4, col=1)
    fig.update_xaxes(title_text="Length (m)", row=4, col=1)

    return fig
