import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 🏗️ Professional Constants ---
C_CONC = '#000000'
C_TOP  = '#d63031' # สีแดงเหล็กบน
C_BOT  = '#27ae60' # สีเขียวเหล็กล่าง
C_STIR = '#b2bec3'

def _draw_support_by_type(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support ตามประเภท (Standard Structural Symbols)"""
    if sup_type.lower() == 'fixed':
        # เสาคอนกรีตหนา
        ax.add_patch(patches.Rectangle((x-100, y_bottom-500), 200, 500, fc='#dfe6e9', ec='black', lw=1.5, zorder=1))
    elif sup_type.lower() == 'roller':
        # สามเหลี่ยมมีล้อ
        pts = np.array([[x, y_bottom], [x-80, y_bottom-200], [x+80, y_bottom-200]])
        ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=1.2, zorder=1))
        ax.add_patch(patches.Circle((x, y_bottom-230), 25, fc='white', ec='black', lw=1, zorder=1))
    else: # Pin / Hinge (Default)
        # รูปสามเหลี่ยมฐานติดพื้น
        pts = np.array([[x, y_bottom], [x-80, y_bottom-250], [x+80, y_bottom-250]])
        ax.add_patch(patches.Polygon(pts, fc='#f1f2f6', ec='black', lw=1.2, zorder=1))
    
    # ชื่อ Support
    ax.text(x, y_bottom - 650, f"S{sup_id}", ha='center', va='top', fontweight='bold', fontsize=9)

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาวระดับสากล: แยกประเภท Support และระบุเหล็กครบถ้วน"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 1000 # Normalized Visual Height สำหรับคานผอมยาว
    
    fig_w = max(16, total_L / 450)
    fig, ax = plt.subplots(figsize=(fig_w, 3.5), dpi=140)
    
    # 1. วาดตัวคาน (Beam Outline)
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2, ec=C_CONC, fc='white', zorder=2))
    
    # 2. วาด Support ตาม Type
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_by_type(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 3. วาดเหล็กเสริมและใส่ Label ทั้งบนและล่าง
    y_top = v_h * 0.8
    y_bot = v_h * 0.2
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # วาดเหล็กเมน (Lines)
        ax.plot([x_curr, x_curr + span_L], [y_top, y_top], color=C_TOP, lw=3, zorder=5, solid_capstyle='round')
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_bot, y_bot], color=C_BOT, lw=3, zorder=5, solid_capstyle='round')
        
        # --- 🏷️ ใส่ Label เหล็กบน (Top) ---
        ax.text(mid, v_h + 150, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", 
                color=C_TOP, ha='center', fontweight='bold', fontsize=9)
        
        # --- 🏷️ ใส่ Label เหล็กล่าง (Bottom) ---
        ax.text(mid, y_bot + 60, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", 
                color=C_BOT, ha='center', va='bottom', fontweight='bold', fontsize=8)
        
        # --- 🏷️ เหล็กปลอก (Stirrup Label) ---
        ax.text(mid, -150, f"Stir. RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
                color='#636e72', ha='center', fontsize=8, style='italic')

        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-600, total_L + 600)
    ax.set_ylim(-900, v_h + 500)
    
    plt.tight_layout()
    return fig
