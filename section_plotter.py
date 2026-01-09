import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- 📐 Professional Styling ---
C_BEAM = '#000000'
C_TOP  = '#D63031'
C_BOT  = '#27AE60'
C_STIR = '#636E72'

def _draw_pro_support(ax, x, y_bottom, sup_type, sup_id):
    """วาดสัญลักษณ์ Support มาตรฐานวิศวกรรมสากล"""
    s = 180 # Scale size
    if sup_type.lower() == 'fixed':
        # สัญลักษณ์ผนังรับแรง (Hatch pattern)
        ax.add_patch(patches.Rectangle((x-100, y_bottom-400), 200, 400, fc='#dfe6e9', ec='black', lw=1.5, hatch='///'))
    elif sup_type.lower() == 'roller':
        # สามเหลี่ยม Roller แบบมีช่องว่างด้านล่าง
        pts = np.array([[x, y_bottom], [x-s/2, y_bottom-s], [x+s/2, y_bottom-s]])
        ax.add_patch(patches.Polygon(pts, fc='white', ec='black', lw=1.2, zorder=5))
        ax.plot([x-s, x+s], [y_bottom-s-30, y_bottom-s-30], color='black', lw=1.5)
    else: # Pin/Hinge
        # สามเหลี่ยมมีจุดหมุนและฐานหยัก
        pts = np.array([[x, y_bottom], [x-s/2, y_bottom-s], [x+s/2, y_bottom-s]])
        ax.add_patch(patches.Polygon(pts, fc='#ced6e0', ec='black', lw=1.2, zorder=5))
        ax.plot([x-s, x+s], [y_bottom-s, y_bottom-s], color='black', lw=2)

    ax.text(x, y_bottom - 600, f"S{sup_id}", ha='center', fontweight='bold', fontsize=10, color='blue')

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    รูปตัดตามยาวระดับ High-Resolution (300 DPI) 
    คานบางยาว (Long-Span) เหล็กชัดเจน ไม่ทับเส้น
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 800  # ปรับความสูงคานในรูปให้บางลงอีกเพื่อความสวยงาม
    
    # 1. สร้าง Canvas ความละเอียดสูง
    fig_w = max(18, total_L / 400)
    fig, ax = plt.subplots(figsize=(fig_w, 4), dpi=300) # เพิ่ม DPI เป็น 300
    
    # 2. วาดขอบคอนกรีต (Outline)
    ax.add_patch(patches.Rectangle((0, 0), total_L, v_h, lw=2, ec=C_BEAM, fc='white', zorder=2))
    
    # 3. วาด Support (ใต้ท้องคาน)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_pro_support(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 4. วาดเหล็กเสริม (Reinforcement Layers)
    y_top = v_h * 0.85
    y_bot = v_h * 0.15
    
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_curr + span_L/2
        
        # เหล็กบน (Top Main)
        ax.plot([x_curr, x_curr + span_L], [y_top, y_top], color=C_TOP, lw=3.5, zorder=10, solid_capstyle='round')
        
        # เหล็กล่าง (Bottom Main)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_bot, y_bot], color=C_BOT, lw=3.5, zorder=10, solid_capstyle='round')
        
        # --- 🏷️ Text Annotations (ใช้ semibold เพื่อความคมชัด) ---
        ax.text(mid, v_h + 100, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", 
                color=C_TOP, ha='center', va='bottom', fontsize=11, fontweight='semibold')
        
        ax.text(mid, y_bot + 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", 
                color=C_BOT, ha='center', va='bottom', fontsize=10, fontweight='semibold')
        
        ax.text(mid, -150, f"RB{int(res['stir_db'])} @ {int(res['shear']['s'])} mm", 
                color=C_STIR, ha='center', fontsize=9, style='italic')

        x_curr += span_L

    # 5. Final Display Setup
    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-800, v_h + 600)
    
    plt.tight_layout()
    return fig
