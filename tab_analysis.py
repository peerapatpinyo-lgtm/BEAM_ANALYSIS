# tab_analysis.py
import streamlit as st
import pandas as pd
import design_view # สมมติว่าไฟล์เดิมที่ใช้วาดกราฟชื่อนี้

def render(x_plot, M_plot, V_plot, D_plot, R_plot, spans, sup_df, display_loads, tag):
    st.subheader(f"📈 Force Diagrams ({tag} Load)")
    
    # เตรียม DataFrame สำหรับ Plot
    df_for_plot = pd.DataFrame({
        'x': x_plot, 
        'moment': M_plot, 
        'shear': V_plot, 
        'deflection': D_plot * 1000
    })
    
    # เรียกฟังก์ชันวาดกราฟ (จากโมดูลเดิมของคุณ)
    if not df_for_plot.empty:
        st.plotly_chart(
            design_view.plot_analysis_results(df_for_plot, spans, sup_df, display_loads, R_plot), 
            use_container_width=True
        )
    
    # แสดงค่าสรุป (Metrics)
    master_df = pd.DataFrame({
        'M_kNm': M_plot / 1000.0,
        'V_kN': V_plot / 1000.0,
        'D_mm': D_plot * 1000.0
    })
    
    v_max = master_df['V_kN'].abs().max()
    m_max = master_df['M_kNm'].max()
    m_min = master_df['M_kNm'].min()
    d_max = master_df['D_mm'].abs().max()
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Max Shear", f"{v_max:.2f} kN")
    c2.metric("Max Moment (+)", f"{max(0, m_max):.2f} kNm")
    c3.metric("Max Moment (-)", f"{abs(min(0, m_min)):.2f} kNm")
    c4.metric("Max Deflection", f"{d_max:.2f} mm")
