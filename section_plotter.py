import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 Professional Configuration (Safe & Sharp) ---
# ใช้ DPI สูงเพื่อความคมชัดของข้อความ
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.weight'] = 'black'

C_BEAM = '#000000' # ดำ
C_TOP  = '#E31A1C' # แดง
C_BOT  = '#33A02C' # เขียว
C_DIM  = '#2980B9' # ฟ้า

def _draw_pro_support(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support พร้อมระบุประเภทแบบ High-Contrast"""
    t = str(sup_type).upper()
    # สร้าง Polygon/Rectangle โดยปิด antialiasing (aa=False) เพื่อความคมกริบ
    if t == 'FIXED':
        rect = patches.Rectangle((x-100, y_bottom-400), 200, 400, fc='#BDC3C7', ec='black', lw=2, hatch='////', aa=False)
        ax.add_patch(rect)
    elif t == 'ROLLER':
        pts = np.array([[x, y_bottom], [x-100, y_bottom-180], [x+100, y_bottom-180]])
        poly = patches.Polygon(pts, fc='white', ec='black', lw=2, aa=False, zorder=10)
        ax.add_patch(poly)
        ax.add_patch(patches.Circle((x, y_bottom-220), 30, fc='black', aa=False, zorder=11))
    else: # PIN
        pts = np.array([[x, y_bottom], [x-100, y_bottom-200], [x+100, y_bottom-200]])
        poly = patches.Polygon(pts, fc='#2C3E50', ec='black', lw=2, aa=False, zorder=10)
        ax.add_patch(poly)

    # Label Support: ใช้พิกัดที่คงที่
    ax.annotate(f"S{sup_id}: {t}", xy=(x, y_bottom-500), ha='center', va='top', 
                fontsize=11, fontweight='black', color='blue',
                bbox=dict(facecolor='white', edgecolor='none', pad=0.5))

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """วาดรูปตัดตามยาว: แก้ไข Error และเน้นความคมชัดสูงสุด"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 450 # ความสูงคานเพรียวบาง
    
    # คำนวณความกว้างรูปตามความยาวจริง
    fig_w = max(18, total_L / 300)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    # 1. วาด Concrete Body (ปิด antialiasing เพื่อขอบคม)
    beam_rect = patches.Rectangle((0, 0), total_L, v_h, lw=2.5, ec=C_BEAM, fc='white', zorder=2, aa=False)
    ax.add_patch(beam_rect)
    
    # 2. Grid & Dimension Chain
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        # Grid Circle
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=2), 
                    fontsize=15, fontweight='black')
        # Grid Line
        ax.plot([curr_x, curr_x], [-700, v_h + 400], color='#7F8C8D', ls='--', lw=1, zorder=1)
        
        if i < len(spans_mm):
            # Dimension Line
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.5))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", 
                    ha='center', color=C_DIM, fontsize=12, fontweight='black')
            curr_x += s_mm

    # 3. Supports
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_pro_support(ax, row['x']*1000, 0, row.get('type', 'PIN'), row.get('id', ''))

    # 4. Reinforcement (คมกริบด้วย aa=False)
    y_t, y_b = v_h * 0.85, v_h * 0.15
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # วาดเส้นเหล็ก (Main Reinforcement)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color=C_TOP, lw=4, zorder=10, aa=False)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color=C_BOT, lw=4, zorder=10, aa=False)
        
        # Labels (คมชัดสูง)
        txt_top = f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)"
        ax.annotate(txt_top, xy=(mid, v_h + 80), ha='center', va='bottom', 
                    fontsize=12, color=C_TOP, fontweight='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        txt_bot = f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)"
        ax.annotate(txt_bot, xy=(mid, y_b - 50), ha='center', va='top', 
                    fontsize=12, color=C_BOT, fontweight='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        x_curr += span_L

    # 5. Final Setup
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1200, total_L + 1200)
    ax.set_ylim(-1000, v_h + 700)
    
    return fig
