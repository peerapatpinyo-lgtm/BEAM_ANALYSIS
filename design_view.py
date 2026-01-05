import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def render_result_tables(df_res, reactions, spans, unit_force="kg", unit_len="m"):
    st.markdown("---")
    st.subheader("📋 Analysis Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"**📍 Support Reactions ({unit_force})**")
        if reactions is not None and len(reactions) > 0:
            react_data = [{"Support": f"Support {i+1}", "Reaction": f"{r:,.2f}"} for i, r in enumerate(reactions)]
            st.table(pd.DataFrame(react_data))
        else:
            st.warning("No reaction data available.")

    with col2:
        st.markdown(f"**📊 Critical Design Values**")
        if df_res is not None and not df_res.empty:
            st.dataframe(df_res.style.format({"Value": "{:,.2f}"}))
        else:
            st.info("No result data to display.")

def plot_professional_diagrams(L_total, loads, reactions_locs, shear_x, shear_y, moment_x, moment_y):
    # ตั้งค่า Style กราฟให้ดูคลีน (Textbook Style)
    plt.style.use('default')
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, gridspec_kw={'height_ratios': [1, 1, 1]})
    
    # === 1. Free Body Diagram ===
    beam_h = max(0.4, L_total * 0.05)
    # คาน (สี่เหลี่ยมมีขอบ)
    ax1.add_patch(patches.Rectangle((0, -beam_h/2), L_total, beam_h, lw=2, ec='#333', fc='#f9f9f9', zorder=2))
    
    # Support (สามเหลี่ยมใต้คาน)
    supp_sz = beam_h * 0.8
    for loc in reactions_locs:
        ax1.add_patch(patches.Polygon([[loc, -beam_h/2], [loc-supp_sz/2, -beam_h/2-supp_sz], [loc+supp_sz/2, -beam_h/2-supp_sz]], 
                                      closed=True, ec='#333', fc='#fff', lw=1.5, zorder=1))
        # พื้นดิน (Ground)
        ax1.plot([loc-supp_sz, loc+supp_sz], [-beam_h/2-supp_sz]*2, 'k-', lw=1.5)

    # Loads
    load_h = beam_h * 2.0
    for l_type, val, p1, p2 in loads:
        y_top = beam_h/2 + load_h
        if l_type == 'udl': # Distributed Load (Comb Style)
            ax1.plot([p1, p2], [y_top]*2, color='#005b96', lw=1.5)
            ax1.plot([p1, p1], [beam_h/2, y_top], color='#005b96', lw=1.5)
            ax1.plot([p2, p2], [beam_h/2, y_top], color='#005b96', lw=1.5)
            # ลูกศรถี่ๆ
            for x in np.linspace(p1, p2, max(3, int((p2-p1)*3))):
                ax1.arrow(x, y_top, 0, -load_h*0.85, head_width=L_total*0.015, head_length=load_h*0.15, fc='#005b96', ec='#005b96')
            ax1.text((p1+p2)/2, y_top + beam_h*0.2, f"w = {val:,.0f}", ha='center', va='bottom', color='#005b96', fontweight='bold')
        elif l_type == 'point': # Point Load
            ax1.arrow(p1, y_top, 0, -load_h*0.85, head_width=L_total*0.02, head_length=load_h*0.2, fc='#d9534f', ec='#d9534f', width=L_total*0.003)
            ax1.text(p1, y_top + beam_h*0.2, f"P = {val:,.0f}", ha='center', va='bottom', color='#d9534f', fontweight='bold')

    ax1.set_title("Free Body Diagram", fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')
    ax1.set_ylim(-beam_h*4, beam_h*5)

    # === 2. Shear Diagram ===
    ax2.plot(shear_x, shear_y, color='#ff9f43', lw=2)
    ax2.fill_between(shear_x, shear_y, 0, facecolor='#ff9f43', alpha=0.15)
    ax2.axhline(0, color='black', lw=0.8)
    ax2.set_ylabel("Shear Force", fontweight='bold')
    ax2.grid(True, ls=':', alpha=0.6)
    # Annotate Max
    if len(shear_y) > 0:
        v_max, v_min = np.max(shear_y), np.min(shear_y)
        ax2.text(shear_x[np.argmax(shear_y)], v_max, f"{v_max:,.0f}", ha='center', va='bottom', fontsize=9, bbox=dict(fc='white', ec='#ff9f43', boxstyle='round,pad=0.2'))

    # === 3. Moment Diagram ===
    ax3.plot(moment_x, moment_y, color='#54a0ff', lw=2)
    ax3.fill_between(moment_x, moment_y, 0, facecolor='#54a0ff', alpha=0.15)
    ax3.axhline(0, color='black', lw=0.8)
    ax3.set_ylabel("Moment", fontweight='bold')
    ax3.set_xlabel("Beam Length (m)")
    ax3.grid(True, ls=':', alpha=0.6)
    # Annotate Max
    if len(moment_y) > 0:
        m_max, m_min = np.max(moment_y), np.min(moment_y)
        if abs(m_max) > abs(m_min): val_show = m_max; idx = np.argmax(moment_y)
        else: val_show = m_min; idx = np.argmin(moment_y)
        ax3.text(moment_x[idx], val_show, f"{val_show:,.0f}", ha='center', va='bottom' if val_show>0 else 'top', fontsize=9, bbox=dict(fc='white', ec='#54a0ff', boxstyle='round,pad=0.2'))

    plt.tight_layout()
    return fig
