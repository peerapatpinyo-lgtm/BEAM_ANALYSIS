import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import PIL 

# --- FIX: ปลดล็อคขีดจำกัดขนาดรูปภาพ ---
PIL.Image.MAX_IMAGE_PIXELS = None 

from solver import BeamSolver

st.set_page_config(page_title="Beam Analysis", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .stButton>button { width: 100%; border-radius: 5px; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f0f2f6; border-radius: 5px 5px 0px 0px; gap: 1px; padding-top: 10px; padding-bottom: 10px; }
    .stTabs [aria-selected="true"] { background-color: #ffffff; border-bottom: 2px solid #4e8cff; }
</style>
""", unsafe_allow_html=True)

st.title("🏗️ Beam Analysis Pro")

# --- SIDEBAR: GLOBAL SETTINGS ---
with st.sidebar:
    st.header("⚙️ คุณสมบัติวัสดุ (Properties)")
    E = st.number_input("Elastic Modulus (E) [Pa]", value=2.0e11, format="%.2e")
    I = st.number_input("Moment of Inertia (I) [m^4]", value=1.0e-4, format="%.2e")
    
    st.markdown("---")
    st.markdown("**Advanced Settings**")
    beam_type = st.selectbox("ทฤษฎีการคำนวณ", ["Euler", "Timoshenko"])
    if beam_type == "Timoshenko":
        A = st.number_input("Cross-sectional Area (A) [m^2]", value=0.01, format="%.4f")
    else:
        A = 0.01

# --- INPUT SECTION: 3 TABS ---
tab1, tab2, tab3 = st.tabs(["📏 1. Geometry (คาน)", "base 2. Supports (จุดรองรับ)", "⬇️ 3. Loads (แรงกระทำ)"])

# --- TAB 1: GEOMETRY ---
with tab1:
    st.subheader("กำหนดความยาวช่วงคาน (Span Lengths)")
    col1, col2 = st.columns([2, 1])
    with col1:
        spans_input = st.text_input("ระบุความยาวแต่ละช่วง (เมตร) คั่นด้วยคอมม่า", "5, 5")
        try:
            spans = [float(x.strip()) for x in spans_input.split(',')]
            node_locs = np.concatenate(([0], np.cumsum(spans)))
            st.success(f"✅ ความยาวรวม: {sum(spans)} เมตร | จำนวนจุดต่อ (Nodes): {len(node_locs)}")
            st.write(f"📍 ตำแหน่ง Nodes: {list(node_locs)}")
        except:
            st.error("❌ รูปแบบตัวเลขไม่ถูกต้อง")
            spans = []
            node_locs = []

# --- TAB 2: SUPPORTS ---
with tab2:
    st.subheader("กำหนดจุดรองรับ (Supports)")
    
    # Init Session State
    if 'supports_list' not in st.session_state:
        st.session_state.supports_list = [
            {'id': 0, 'type': 'Pin', 'settlement': 0.0, 'k_spring': 0.0},
            {'id': len(spans) if spans else 1, 'type': 'Roller', 'settlement': 0.0, 'k_spring': 0.0}
        ]
    
    col_s1, col_s2 = st.columns([1, 2])
    
    with col_s1:
        st.markdown("##### เพิ่ม/แก้ไข จุดรองรับ")
        if len(node_locs) > 0:
            s_node = st.selectbox("เลือกตำแหน่ง Node", range(len(node_locs)))
            s_type = st.selectbox("ประเภท Support", ["Pin", "Roller", "Fixed"])
            s_settlement = st.number_input("การทรุดตัว (เมตร)", value=0.0, step=0.001, format="%.4f", help="ค่าบวกคือทรุดลง")
            
            if st.button("➕ เพิ่ม Support"):
                # ลบอันเก่าที่ node นั้นออกก่อน
                st.session_state.supports_list = [s for s in st.session_state.supports_list if s['id'] != s_node]
                st.session_state.supports_list.append({
                    'id': s_node, 'type': s_type, 
                    'settlement': s_settlement, 'k_spring': 0.0
                })
                st.rerun()
                
    with col_s2:
        st.markdown("##### รายการจุดรองรับปัจจุบัน")
        if st.session_state.supports_list:
            df_sup = pd.DataFrame(st.session_state.supports_list)
            if 'settlement' not in df_sup.columns: df_sup['settlement'] = 0.0
            # Sort by ID
            df_sup = df_sup.sort_values(by='id')
            st.dataframe(df_sup[['id', 'type', 'settlement']], hide_index=True, use_container_width=True)
            if st.button("ล้างค่า Supports ทั้งหมด"):
                st.session_state.supports_list = []
                st.rerun()

# --- TAB 3: LOADS ---
with tab3:
    st.subheader("กำหนดแรงกระทำ (Loads)")
    
    if 'loads_list' not in st.session_state:
        st.session_state.loads_list = [{'span_idx': 0, 'type': 'U', 'mag': 10000, 'x': 0, 'dist': 5}]

    col_l1, col_l2 = st.columns([1, 2])
    
    with col_l1:
        st.markdown("##### เพิ่มแรงกระทำ")
        if spans:
            l_span = st.selectbox("ช่วงคานที่ (Span Index)", range(len(spans)))
            l_type = st.selectbox("ประเภทแรง", ["Point Load (P)", "Uniform Load (U)", "Moment (M)"])
            
            l_mag = st.number_input("ขนาดแรง (Magnitude)", value=1000.0, help="แรงลงเป็นบวก (+)")
            l_x = st.number_input("ระยะจากซ้ายของช่วง (x)", value=2.5)
            
            l_dist = 0.0
            if "Uniform" in l_type:
                l_dist = st.number_input("ความยาวแรงแผ่ (dist)", value=1.0)
            
            if st.button("➕ เพิ่ม Load"):
                l_code = 'P' if "Point" in l_type else ('U' if "Uniform" in l_type else 'M')
                new_load = {'span_idx': l_span, 'type': l_code, 'mag': l_mag, 'x': l_x}
                if l_code == 'U': new_load['dist'] = l_dist
                st.session_state.loads_list.append(new_load)
                st.rerun()
                
    with col_l2:
        st.markdown("##### รายการแรงกระทำปัจจุบัน")
        if st.session_state.loads_list:
            df_load = pd.DataFrame(st.session_state.loads_list)
            st.dataframe(df_load, hide_index=True, use_container_width=True)
            if st.button("ล้างค่า Loads ทั้งหมด"):
                st.session_state.loads_list = []
                st.rerun()

st.markdown("---")

# --- PLOTTING FUNCTIONS (ENGINEERING STYLE) ---
def plot_beam_diagram(ax, spans, supports, loads):
    total_len = sum(spans)
    node_x = np.concatenate(([0], np.cumsum(spans)))
    
    # Beam Line
    ax.plot([0, total_len], [0, 0], 'k-', linewidth=4, solid_capstyle='round')
    # Nodes
    ax.scatter(node_x, np.zeros_like(node_x), color='white', edgecolor='black', zorder=10, s=40)
    
    # Supports
    for s in supports:
        if int(s['id']) < len(node_x):
            x = node_x[int(s['id'])]
            if s['type'] == 'Pin':
                ax.plot(x, -0.25, marker='^', color='#444', markersize=14)
            elif s['type'] == 'Roller':
                ax.plot(x, -0.25, marker='o', color='#444', markersize=12)
            elif s['type'] == 'Fixed':
                ax.add_patch(patches.Rectangle((x-0.1, -0.6), 0.2, 1.2, color='#444'))
    
    # Loads
    max_load = 1
    if loads: max_load = max([abs(l['mag']) for l in loads])
    if max_load == 0: max_load = 1
    
    for l in loads:
        if int(l['span_idx']) < len(spans):
            x = node_x[int(l['span_idx'])] + l['x']
            mag = l['mag']
            if l['type'] == 'P':
                dy = -0.8 if mag > 0 else 0.8 # Positive load = Downward arrow
                ax.arrow(x, -dy, 0, dy*0.8, head_width=0.15, head_length=0.2, fc='red', ec='red', linewidth=2)
                ax.text(x, -dy*1.3, f"P={mag}", ha='center', color='red', fontweight='bold')
            elif l['type'] == 'U':
                dist = l.get('dist', 1.0)
                ax.add_patch(patches.Rectangle((x, 0.1), dist, 0.4, facecolor='orange', alpha=0.5))
                ax.text(x + dist/2, 0.8, f"w={mag}", ha='center', color='orange', fontweight='bold')
            elif l['type'] == 'M':
                ax.text(x, 0.5, f"M={mag}", ha='center', color='purple', fontweight='bold')
                
    ax.set_ylim(-2, 2)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.axis('off')
    ax.set_title("System Diagram (แผนภาพโครงสร้าง)", loc='left', fontsize=12, fontweight='bold')

def annotate_peaks(ax, x, y, color, invert=False):
    if len(y) == 0: return
    valid_idx = ~np.isnan(y)
    x = np.array(x)[valid_idx]
    y = np.array(y)[valid_idx]
    if len(y) == 0: return
    
    ymax = np.max(y); ymin = np.min(y)
    xmax = x[np.argmax(y)]; xmin = x[np.argmin(y)]
    
    if abs(ymax) < 1e-9 and abs(ymin) < 1e-9: return

    # Annotate Max
    ax.annotate(f"{ymax:.2f}", xy=(xmax, ymax), xytext=(0, 10 if not invert else -15),
                textcoords="offset points", ha='center', color=color, fontweight='bold', fontsize=9)
    # Annotate Min
    ax.annotate(f"{ymin:.2f}", xy=(xmin, ymin), xytext=(0, -15 if not invert else 10),
                textcoords="offset points", ha='center', color=color, fontweight='bold', fontsize=9)

# --- CALCULATION BUTTON ---
st.markdown("###")
if st.button("🚀 คำนวณ (Analyze)", type="primary"):
    if not spans:
        st.error("กรุณาระบุความยาวช่วงคานใน Tab 1")
    else:
        try:
            beam_props = {'E': E, 'I': I, 'A': A, 'type': beam_type}
            
            # SOLVER CALL
            solver = BeamSolver(spans, st.session_state.supports_list, st.session_state.loads_list, beam_props)
            results, R = solver.solve()
            
            st.success("✅ คำนวณเสร็จสิ้น")
            
            # --- PLOTTING ---
            fig, ax = plt.subplots(4, 1, figsize=(10, 12), gridspec_kw={'height_ratios': [1, 2, 2, 2], 'hspace': 0.5})
            
            # 1. System
            plot_beam_diagram(ax[0], spans, st.session_state.supports_list, st.session_state.loads_list)
            
            # 2. Shear (SFD)
            x_vals = results['x']; v_vals = results['shear']
            ax[1].plot(x_vals, v_vals, color='#1f77b4', linewidth=1.5)
            ax[1].fill_between(x_vals, v_vals, 0, color='#1f77b4', alpha=0.2)
            ax[1].set_ylabel("Shear (N)", fontweight='bold')
            ax[1].set_title("Shear Force Diagram (SFD)", loc='left', fontsize=10)
            ax[1].grid(True, linestyle='--', alpha=0.4)
            ax[1].axhline(0, color='black', linewidth=0.8)
            annotate_peaks(ax[1], x_vals, v_vals, '#1f77b4')
            
            # 3. Moment (BMD) - Inverted
            m_vals = results['moment']
            ax[2].plot(x_vals, m_vals, color='#d62728', linewidth=1.5)
            ax[2].fill_between(x_vals, m_vals, 0, color='#d62728', alpha=0.2)
            ax[2].set_ylabel("Moment (N·m)", fontweight='bold')
            ax[2].set_title("Bending Moment Diagram (BMD)", loc='left', fontsize=10)
            ax[2].grid(True, linestyle='--', alpha=0.4)
            ax[2].axhline(0, color='black', linewidth=0.8)
            ax[2].invert_yaxis() # Invert for Civil style
            annotate_peaks(ax[2], x_vals, m_vals, '#d62728', invert=True)
            
            # 4. Deflection
            d_vals = results['deflection']
            ax[3].plot(x_vals, d_vals, color='#2ca02c', linewidth=1.5)
            ax[3].fill_between(x_vals, d_vals, 0, color='#2ca02c', alpha=0.1)
            ax[3].set_ylabel("Deflection (m)", fontweight='bold')
            ax[3].set_xlabel("Position (m)")
            ax[3].grid(True, linestyle='--', alpha=0.4)
            ax[3].axhline(0, color='black', linewidth=0.8)
            
            if len(d_vals) > 0:
                max_idx = np.argmax(np.abs(d_vals))
                max_val = d_vals[max_idx]
                ax[3].plot(x_vals[max_idx], max_val, 'ko', markersize=4)
                ax[3].text(x_vals[max_idx], max_val, f" Max: {max_val:.4e} m", ha='left', fontweight='bold')

            st.pyplot(fig, dpi=100)
            
        except Exception as e:
            st.error(f"เกิดข้อผิดพลาด: {e}")
