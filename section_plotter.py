import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 Engineering Visual Standards ---
plt.rcParams['figure.dpi'] = 300           # บังคับความชัดระดับพิมพ์เขียว
plt.rcParams['font.size'] = 10             # ขนาดฟอนต์มาตรฐาน
plt.rcParams['font.weight'] = 'bold'       # ตัวหนาเพื่อความชัดเจน
plt.rcParams['axes.linewidth'] = 1.5       # เส้นขอบภาพหนาขึ้น

C_BEAM = '#000000'
C_TOP  = '#E31A1C' # แดงเข้ม (Top Rebar)
C_BOT  = '#33A02C' # เขียวเข้ม (Bot Rebar)
C_STIR = '#7F8C8D' # เทาเข้ม (Stirrup)

def _draw_advanced_support(ax, x, y_bottom, sup_type, sup_id):
    """วาด Support แบบสัญลักษณ์ทางวิศวกรรมสากล (Sharp & Solid)"""
    if sup_type.lower() == 'fixed':
        # สัญลักษณ์การยึดแน่น (Hatch wall)
        ax.add_patch(patches.Rectangle((x-120, y_bottom-400), 240, 400, fc='#BDC3C7', ec='black', lw=2, hatch='///'))
    elif sup_type.lower() == 'roller':
        # Roller: สามเหลี่ยม + ฐานวงกลม
        pts = np.array([[x, y_bottom], [x-100, y_bottom-180], [x+100, y_bottom-180]])
        ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=2, zorder=10))
        ax.add_patch(patches.Circle((x, y_bottom-220), 30, fc='black', zorder=11))
    else: # Pin / Hinge
        pts = np.array([[x, y_bottom], [x-100, y_bottom-200], [x+100, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='#2C3E50', ec='black', lw=2, zorder=10))
    
    ax.annotate(f"S{sup_id}", xy=(x, y_bottom-450), ha='center', va='top', fontsize=12, fontweight='black', color='blue')

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดตามยาวระดับ Advanced Structural Detailing 
    คานผอมเพรียว ตัวหนังสือคมชัดสูงสุด ไม่มีการทับซ้อน
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 500  # บังคับความสูงคานให้เพรียวบาง (Thin-Beam Visualization)
    
    # 1. ปรับขนาดรูปภาพตามความยาวคาน (Dynamic Scaling)
    fig_w = max(18, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4))
    
    # 2. วาดตัวคาน (Beam Body) - แยกเลเยอร์ชัดเจน
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2.5, ec=C_BEAM, fc='white', zorder=2))
    
    # 3. วาด Grid Lines (Center-to-Center Line)
    curr_x = 0
    for i, span_L in enumerate(spans_mm + [0]):
        ax.plot([curr_x, curr_x], [-600, v_h + 400], color='#BDC3C7', ls='-.', lw=1, zorder=1)
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center', 
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), fontsize=14)
        if i < len(spans_mm): curr_x += spans_mm[i]

    # 4. วาด Support
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_advanced_support(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 5. วาดเหล็กเสริม (Reinforcement Layers)
    y_t = v_h * 0.8  # Top Zone
    y_b = v_h * 0.2  # Bottom Zone
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (Top Bar) พร้อมการวาด Hook ปลายคาน
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color=C_TOP, lw=4, zorder=20, solid_capstyle='round')
        
        # เหล็กล่าง (Bottom Bar)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color=C_BOT, lw=4, zorder=20)
        
        # --- การใส่ตัวหนังสือแบบ Annotation (ไม่แตกเมื่อซูม) ---
        # เหล็กบน
        ax.annotate(f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", xy=(mid, v_h + 80), 
                    ha='center', va='bottom', fontsize=12, color=C_TOP, fontweight='black')
        
        # เหล็กล่าง (ระบุจำนวนเหล็กเสริมล่างตามที่คุณต้องการ)
        ax.annotate(f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", xy=(mid, y_b + 40), 
                    ha='center', va='bottom', fontsize=11, color=C_BOT, fontweight='black')
        
        # เหล็กปลอก
        ax.annotate(f"RB{int(res['stir_db'])} @ {int(res['shear']['s'])} mm", xy=(mid, -150), 
                    ha='center', va='top', fontsize=10, color='#34495E', style='italic')

        x_curr += span_L

    # 6. ตั้งค่ามุมมองและสเกล
    ax.set_aspect('auto') # บังคับให้ยาวเพรียวตามแนวราบ
    ax.axis('off')
    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-900, v_h + 700)
    
    plt.tight_layout()
    return fig

def plot_section(b_m, h_m, cover_mm, n_top, n_bot, db_top, db_bot, title="SECTION"):
    """หน้าตัดขวางความละเอียดสูง"""
    b, h = b_m * 1000, h_m * 1000
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=3, ec='black', fc='none', zorder=1))
    
    gap = cover_mm + 20
    def _draw_bars(n, y, db, color):
        if n <= 0: return
        xs = np.linspace(gap, b - gap, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=1, zorder=10))

    _draw_bars(n_top, h - gap, db_top, C_TOP)
    _draw_bars(n_bot, gap, db_bot, C_BOT)
    
    ax.set_title(title, fontweight='black', fontsize=14, pad=15)
    ax.axis('equal')
    ax.axis('off')
    return fig
