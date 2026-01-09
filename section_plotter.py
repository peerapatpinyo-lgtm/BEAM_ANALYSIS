import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 Precision Settings (แก้ภาพฟุ้ง) ---
plt.rcParams['path.simplify'] = True
plt.rcParams['path.simplify_threshold'] = 1.0
plt.rcParams['text.antialiased'] = True 

C_CONC = '#000000' # ดำสนิท
C_TOP  = '#FF0000' # แดงสด
C_BOT  = '#008000' # เขียวเข้ม
C_DIM  = '#2980b9' # สีฟ้าสำหรับเส้นบอกระยะ

def _draw_pro_support(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support พร้อมระบุชื่อ Type"""
    t = sup_type.upper()
    if t == 'FIXED':
        ax.add_patch(patches.Rectangle((x-120, y_bottom-400), 240, 400, fc='#dfe6e9', ec='black', lw=2, hatch='///'))
    elif t == 'ROLLER':
        pts = np.array([[x, y_bottom], [x-100, y_bottom-200], [x+100, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=2))
        ax.add_patch(patches.Circle((x, y_bottom-230), 30, fc='black'))
    else: # PIN
        pts = np.array([[x, y_bottom], [x-100, y_bottom-200], [x+100, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='#2c3e50', ec='black', lw=2))
        ax.plot([x-150, x+150], [-200, -200], color='black', lw=2)

    # ระบุทั้ง ID และ TYPE
    ax.text(x, y_bottom - 480, f"S{sup_id}: {t}", ha='center', fontweight='black', fontsize=11)

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาวฉบับวิศวกรมืออาชีพ: คมชัด บอกระยะ และระบุประเภท Support"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 500 # ความสูงคานในรูป (เพรียวบาง)
    
    fig_w = max(20, total_L / 300)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=300) # DPI 300 เพื่อความคมชัดสูงสุด
    
    # 1. วาดคอนกรีตและเส้นบอกระยะหุ้ม (Dimension Line แนวตั้ง)
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2.5, ec=C_CONC, fc='white', zorder=2))
    # เส้นบอกระยะหุ้ม (Covering) ที่หัวคาน
    ax.annotate('', xy=(50, 0), xytext=(50, cover_mm), arrowprops=dict(arrowstyle='<->', color=C_DIM))
    ax.text(60, cover_mm/2, f'cov.{cover_mm}', color=C_DIM, fontsize=8, va='center')

    # 2. เส้นบอกระยะ Span (Dimension Lines แนวนอนด้านบน)
    curr_x = 0
    for s_mm in spans_mm:
        ax.annotate('', xy=(curr_x, v_h + 300), xytext=(curr_x + s_mm, v_h + 300),
                    arrowprops=dict(arrowstyle='<->', lw=1.5, color=C_DIM))
        ax.text(curr_x + s_mm/2, v_h + 350, f"{s_mm/1000:.2f} m", ha='center', color=C_DIM, fontweight='bold')
        curr_x += s_mm

    # 3. Support และ Grid Lines
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            _draw_pro_support(ax, sx, 0, row.get('type', 'PIN'), row.get('id', ''))
            # Grid line
            ax.plot([sx, sx], [-600, v_h + 400], color='#bdc3c7', ls='--', lw=1, zorder=1)

    # 4. เหล็กเสริม (Reinforcement)
    y_t, y_b = v_h * 0.85, v_h * 0.15
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # วาดเส้นเหล็ก (หนาและคม)
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color=C_TOP, lw=4, zorder=10, solid_capstyle='butt')
        ax.plot([x_curr + 30, x_curr + span_L - 30], [y_b, y_b], color=C_BOT, lw=4, zorder=10, solid_capstyle='butt')
        
        # ตัวหนังสือแบบ High-Contrast (ใช้ Bbox สีขาวตัดขอบ)
        props = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9)
        ax.text(mid, v_h + 100, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", 
                color=C_TOP, ha='center', fontweight='black', fontsize=12, bbox=props)
        ax.text(mid, y_b - 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", 
                color=C_BOT, ha='center', va='top', fontweight='black', fontsize=11, bbox=props)
        
        x_curr += span_L

    # 5. Final Layout
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-1000, v_h + 800)
    plt.tight_layout()
    return fig
