import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 Ultra-High Definition Settings ---
plt.rcParams['figure.dpi'] = 300           # บังคับความละเอียดพื้นฐาน
plt.rcParams['font.family'] = 'sans-serif' # ฟอนต์ที่อ่านง่ายที่สุด
plt.rcParams['font.weight'] = 'black'      # บังคับตัวหนาพิเศษเพื่อความชัดเจน
plt.rcParams['axes.antialiased'] = False   # ปิดการเกลี่ยขอบเพื่อให้เส้นคมแบบ CAD

C_CONC = '#000000' # Black
C_TOP  = '#FF0000' # Pure Red
C_BOT  = '#008000' # Pure Green
C_DIM  = '#2980B9' # Blue

def _draw_pro_support(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support พร้อมระบุประเภทแบบคมชัด"""
    t = sup_type.upper()
    if t == 'FIXED':
        ax.add_patch(patches.Rectangle((x-100, y_bottom-400), 200, 400, fc='#BDC3C7', ec='black', lw=2, hatch='////'))
    elif t == 'ROLLER':
        pts = np.array([[x, y_bottom], [x-100, y_bottom-200], [x+100, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=2))
        ax.add_patch(patches.Circle((x, y_bottom-230), 30, fc='black'))
    else: # PIN
        pts = np.array([[x, y_bottom], [x-100, y_bottom-200], [x+100, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='#2C3E50', ec='black', lw=2))

    # Label: ID + Type (คมชัดสูง)
    ax.annotate(f"S{sup_id}: {t}", xy=(x, y_bottom-500), ha='center', va='top', 
                fontsize=12, fontweight='black', bbox=dict(facecolor='white', edgecolor='none', pad=1))

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาวฉบับสมบูรณ์: ไม่แตก คมชัด บอกระยะครบถ้วน"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 450 # คานเพรียวบางพิเศษ
    
    fig_w = max(18, total_L / 300)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    # 1. วาด Concrete Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2.5, ec=C_CONC, fc='white', zorder=2))
    
    # 2. Grid Lines & Dimension Lines (คมกริบ)
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        # Grid Circle (A, B, C...)
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=2), fontsize=15)
        ax.plot([curr_x, curr_x], [-700, v_h + 400], color='#7F8C8D', ls='--', lw=1, zorder=1)
        
        # Dimension Line (Span Length)
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color=C_DIM, lw=1.5))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", ha='center', color=C_DIM, fontsize=12)
            curr_x += s_mm

    # 3. Supports (ระบุตำแหน่งเสา)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_pro_support(ax, row['x']*1000, 0, row.get('type', 'PIN'), row.get('id', ''))

    # 4. Reinforcement (เหล็กเส้นคมกริบ)
    y_t, y_b = v_h * 0.85, v_h * 0.15
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กเมน (ไม่มีการ Anti-alias เพื่อความคม)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color=C_TOP, lw=4, zorder=10)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color=C_BOT, lw=4, zorder=10)
        
        # รายละเอียดเหล็ก (High-Contrast Labels)
        ax.annotate(f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", xy=(mid, v_h + 80), 
                    ha='center', va='bottom', fontsize=12, color=C_TOP, fontweight='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))
        
        ax.annotate(f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", xy=(mid, y_b - 50), 
                    ha='center', va='top', fontsize=12, color=C_BOT, fontweight='black',
                    bbox=dict(facecolor='white', alpha=0.9, edgecolor='none'))

        x_curr += span_L

    # 5. Final Display Adjustment
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1200, total_L + 1200)
    ax.set_ylim(-1000, v_h + 700)
    
    # คำแนะนำ: หากต้องการความคมชัดสูงสุด ให้เซฟไฟล์ด้วยคำสั่ง:
    # fig.savefig('beam_detail.svg') หรือ fig.savefig('beam_detail.pdf')
    return fig
