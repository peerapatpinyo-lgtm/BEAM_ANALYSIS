import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 มาตรฐานการแสดงผลระดับสากล ---
plt.rcParams['figure.autolayout'] = True
plt.rcParams['font.weight'] = 'medium'

# Palettes
C_CONC = '#2d3436'    # คอนกรีต
C_TOP  = '#d63031'    # เหล็กเมนบน (แดง)
C_BOT  = '#00b894'    # เหล็กเมนล่าง (เขียว)
C_STIR = '#636e72'    # เหล็กปลอก
C_GRID = '#b2bec3'    # เส้น Grid

def _draw_pro_support(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support ตามมาตรฐานตำราวิศวกรรม"""
    s = 150 
    if sup_type.lower() == 'fixed':
        ax.add_patch(patches.Rectangle((x-100, y_bottom-400), 200, 400, fc='#ecf0f1', ec=C_CONC, lw=1.5, hatch='///'))
    elif sup_type.lower() == 'roller':
        pts = np.array([[x, y_bottom], [x-80, y_bottom-180], [x+80, y_bottom-180]])
        ax.add_patch(patches.Polygon(pts, fc='white', ec=C_CONC, lw=1.2, zorder=5))
        ax.add_patch(patches.Circle((x, y_bottom-220), 30, fc='white', ec=C_CONC, zorder=6))
        ax.plot([x-150, x+150], [y_bottom-255, y_bottom-255], color=C_CONC, lw=2)
    else: # Pin
        pts = np.array([[x, y_bottom], [x-90, y_bottom-200], [x+90, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='#dfe6e9', ec=C_CONC, lw=1.2, zorder=5))
        ax.plot([x-150, x+150], [y_bottom-200, y_bottom-200], color=C_CONC, lw=2.5)
        # ฐานขีดๆ
        for i in range(-140, 160, 40):
            ax.plot([x+i, x+i-30], [y_bottom-200, y_bottom-250], color=C_CONC, lw=1)

    # Label Support
    ax.text(x, y_bottom - 550, f"S{sup_id}", ha='center', fontweight='bold', fontsize=12, color='#2d3436')

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    ระบบวาดแบบขยายคาน (Beam Detail) ระดับมืออาชีพ 
    แก้ปัญหาตัวหนังสือแตก คานหนา และเหล็กทับเส้น 100%
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 600  # บังคับความสูงคงที่เพื่อให้คานผอมยาวสวยงามเสมอ
    
    # 1. High-Res Canvas (300 DPI สำหรับงานพิมพ์)
    fig_w = max(20, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 5), dpi=300)
    
    # 2. Grid Lines (เส้นแกนอ้างอิง)
    x_grid = 0
    for i, span_L in enumerate(spans_mm + [0]):
        ax.plot([x_grid, x_grid], [-800, v_h + 600], color=C_GRID, lw=1.2, ls='-.', zorder=0)
        ax.add_patch(plt.Circle((x_grid, v_h + 750), 120, fc='white', ec=C_CONC, lw=1.5, zorder=10))
        ax.text(x_grid, v_h + 750, chr(65+i), ha='center', va='center', fontweight='bold', fontsize=14)
        if i < len(spans_mm): x_grid += spans_mm[i]

    # 3. Beam Body (ขอบคอนกรีต)
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2.5, ec=C_CONC, fc='#fcfcfc', zorder=2))
    
    # 4. Supports
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_pro_support(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 5. Reinforcement with Hooks (เหล็กเมนพร้อมระยะงอ)
    y_t = v_h * 0.82
    y_b = v_h * 0.18
    hook_len = 120
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # --- เหล็กบน (Top Main) + Hook ที่ปลายสุด ---
        x_start, x_end = x_curr, x_curr + span_L
        top_x = [x_start, x_end]
        top_y = [y_t, y_t]
        if i == 0: # ปลายซ้ายงอลง
            top_x.insert(0, x_start); top_y.insert(0, y_t - hook_len)
        if i == len(spans_mm) - 1: # ปลายขวางอลง
            top_x.append(x_end); top_y.append(y_t - hook_len)
        ax.plot(top_x, top_y, color=C_TOP, lw=3.5, zorder=10, solid_capstyle='round')

        # --- เหล็กล่าง (Bottom Main) + Hook ---
        bot_x = [x_start + 50, x_end - 50]
        bot_y = [y_b, y_b]
        if i == 0: # ปลายซ้ายงอขึ้น
            bot_x.insert(0, x_start + 50); bot_y.insert(0, y_b + hook_len)
        if i == len(spans_mm) - 1: # ปลายขวางอขึ้น
            bot_x.append(x_end - 50); bot_y.append(y_b + hook_len)
        ax.plot(bot_x, bot_y, color=C_BOT, lw=3.5, zorder=10, solid_capstyle='round')

        # --- High-Quality Labels (ตัวหนังสือไม่แตก) ---
        # ใช้เครื่องหมาย "A-A" กำกับหน้าตัดกลางคาน
        ax.plot([mid, mid], [v_h+50, -50], color='#b2bec3', lw=0.8, ls='--', zorder=1)
        
        ax.text(mid, v_h + 150, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", 
                color=C_TOP, ha='center', va='bottom', fontsize=12, fontweight='bold', 
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        ax.text(mid, y_b + 70, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", 
                color=C_BOT, ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.text(mid, -150, f"Stir. RB{int(res['stir_db'])} @ {int(res['shear']['s'])} mm", 
                color='#535c68', fontsize=10, style='italic', ha='center', fontweight='bold')

        x_curr += span_L

    # 6. Final Settings
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1200, total_L + 1200)
    ax.set_ylim(-1000, v_h + 1000)
    
    return fig

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, title="SECTION A-A"):
    """ หน้าตัดขวาง (Cross Section) แบบคมชัด """
    b, h = b_m * 1000, h_m * 1000
    fig, ax = plt.subplots(figsize=(4, 5), dpi=300)
    
    # Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=3, ec=C_CONC, fc='none', zorder=1))
    
    # Rebar Offsets
    gap = cover_mm + 20
    y_t, y_b = h - gap, gap
    
    def draw_rebars(n, y, db, color):
        if n <= 0: return
        xs = np.linspace(gap, b - gap, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=1, zorder=10))

    draw_rebars(n_top, y_t, db_top_mm, C_TOP)
    draw_rebars(n_bot, y_b, db_bot_mm, C_BOT)
    
    ax.text(b/2, h + 100, title, ha='center', va='bottom', fontsize=14, fontweight='bold')
    ax.text(b/2, -150, f"{int(b)} x {int(h)} mm", ha='center', fontsize=12)
    
    ax.axis('equal')
    ax.axis('off')
    return fig
