import streamlit as st
import pandas as pd
import numpy as np

try:
    import input_handler
    import solver
    import design_view
    import section_plotter # Now this exists!
except ImportError as e:
    st.error(f"Missing Module: {e}")
    st.stop()

st.set_page_config(page_title="Pro Beam Design", layout="wide")

# CSS
st.markdown("""
<style>
    .report-frame { border: 1px solid #ddd; padding: 20px; border-radius: 8px; background: white; }
    .status-ok { color: green; font-weight: bold; }
    .status-fail { color: red; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

if 'analyzed' not in st.session_state: st.session_state.analyzed = False

# --- SIDEBAR INPUTS ---
with st.sidebar:
    st.header("⚙️ Project Setup")
    design_std = st.radio("Standard", ["ACI 318", "EIT 1008"])
    
    # Factors
    if "EIT" in design_std:
        factors = {'DL': 1.4, 'LL': 1.7, 'phi_m': 0.90, 'phi_v': 0.85}
        avg_f = 1.7
    else:
        factors = {'DL': 1.2, 'LL': 1.6, 'phi_m': 0.90, 'phi_v': 0.75}
        avg_f = 1.6

    # Render Inputs
    params, n_spans, spans, sup_df, loads_df, stable = input_handler.render_all_sidebar_inputs()
    
    st.divider()
    run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

# --- EXECUTION ---
if run_btn:
    if not stable:
        st.error("Structure Unstable!")
    else:
        # 1. Prepare Data
        sw = params['b'] * params['h'] * 24000 # N/m
        loads_list = loads_df.to_dict('records') if not loads_df.empty else []
        
        # Add SW
        for i in range(n_spans):
            loads_list.append({"id":f"sw{i}", "type":"U", "span_index":i, "x":0.0, "mag":sw, "dist":spans[i]})
            
        # 2. Solve
        s = solver.BeamSolver(spans, sup_df.to_dict('records'), loads_list, params['E'], params['b'], params['h'], params['I'])
        res, reac, status = s.solve()
        
        if "error" in status:
            st.error(status['error'])
        else:
            # 3. SAVE EVERYTHING TO SESSION
            st.session_state.res = res
            st.session_state.reac = [{'node_id':i, 'fy':v} for i,v in enumerate(reac)] if isinstance(reac, list) else [{'node_id':int(k), 'fy':v} for k,v in reac.items()]
            st.session_state.loads = loads_list
            
            # Critical: Save inputs explicitly
            params['spans_data'] = spans
            params['sup_data'] = sup_df.to_dict('records')
            st.session_state.params = params
            st.session_state.factors = factors
            st.session_state.avg_f = avg_f
            
            st.session_state.analyzed = True
            st.rerun()

# ... (โค้ดส่วนบนเหมือนเดิม) ...

# --- REPORT VIEW ---
# เปลี่ยนบรรทัดนี้: เช็คว่ามี 'res' และ 'avg_f' อยู่จริงไหม
if st.session_state.analyzed and 'res' in st.session_state:
    
    # --- FIX: ใช้ .get(key, default) เพื่อป้องกัน Error ---
    res = st.session_state.get('res')
    reac = st.session_state.get('reac', [])
    loads = st.session_state.get('loads', [])
    p = st.session_state.get('params', {})
    f = st.session_state.get('factors', {})
    saf_f = st.session_state.get('avg_f', 1.6)  # <--- จุดที่ Error แก้เป็นแบบนี้ครับ
    
    # Retrieve Lists safely
    spans_data = p.get('spans_data', [5.0])
    sup_data = p.get('sup_data', [])

    tab1, tab2 = st.tabs(["📊 Analysis Results", "📝 Detail Design"])
    
    with tab1:
        # Technical Note: แจ้ง User ว่าเราใช้วิธีอะไร
        st.info("ℹ️ Calculation Method: **Euler-Bernoulli Beam Theory** (Ignored shear deformation).")
        
        # Plot
        st.plotly_chart(design_view.plot_analysis_results(res, spans_data, sup_data, loads, reac), use_container_width=True)
        
        # Equilibrium
        total_load = sum([l['mag']*l['dist'] if l['type']=='U' else l['mag'] for l in loads])
        total_reac = sum([r['fy'] for r in reac])
        st.write(f"**Equilibrium Check:** Load {total_load/1000:.2f} kN vs Reaction {total_reac/1000:.2f} kN")

    with tab2:
        # ... (ส่วน Design เหมือนเดิม) ...
        # (Copy Code ส่วน Design เดิมมาวางต่อตรงนี้)
        pass 

elif st.session_state.analyzed:
    st.warning("⚠️ Session data missing. Please click 'RUN ANALYSIS' again.")
