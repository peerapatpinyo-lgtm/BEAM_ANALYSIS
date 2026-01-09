import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดยาวคาน (Longitudinal Section) - คงเดิมไว้เพื่อความครบถ้วนของไฟล์
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350  
    
    fig_w = max(16, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', zorder=2)
    ax.add_patch(beam)
    
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        ax.plot([curr_x, curr_x], [-600, v_h + 400], color='#7f8c8d', ls='-.', lw=1, zorder=1)
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), 
                    fontsize=14, fontweight='bold')
        
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#2980b9', lw=1.2))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", 
                    ha='center', color='#2980b9', fontsize=12, fontweight='bold')
            curr_x += s_mm

    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-100, -350), 200, 350, fc='#dfe6e9', ec='black', lw=1.5, hatch='////'))
            elif stype == 'ROLLER':
                ax.add_patch(patches.Polygon([[sx, 0], [sx-90, -180], [sx+90, -180]], fc='white', ec='black', lw=1.5))
                ax.add_patch(patches.Circle((sx, -215), 30, fc='black'))
            else: # PIN
                ax.add_patch(patches.Polygon([[sx, 0], [sx-90, -180], [sx+90, -180]], fc='#2c3e50', ec='black', lw=1.5))
            ax.text(sx, -500, f"S{row['id']}: {stype}", ha='center', fontweight='bold', fontsize=10)

    y_t, y_b = v_h * 0.82, v_h * 0.18
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        s_spacing = res['shear']['s']
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            stir_x = x_curr + (j * s_spacing)
            if stir_x <= x_curr + span_L:
                ax.plot([stir_x, stir_x], [y_b - 20, y_t + 20], color='#bdc3c7', lw=0.7, alpha=0.6, zorder=3)
        
        mid = x_curr + span_L/2
        ax.text(mid, -150, f"RB{int(res['stir_db'])}@{int(s_spacing)}", color='#7f8c8d', fontsize=9, ha='center', style='italic')
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color='#d30000', lw=3.5, zorder=10)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color='#008c00', lw=3.5, zorder=10)
        
        label_opt = dict(ha='center', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        ax.text(mid, v_h + 80, f"{int(res['neg']['n'])}-DB{int(res['top_db'])}", color='#d30000', **label_opt)
        ax.text(mid, y_b - 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])}", color='#008c00', va='top', **label_opt)
        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-800, v_h + 800)
    
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight')
    svg_string = f_svg.getvalue()
    
    f_png = io.BytesIO()
    fig.savefig(f_png, format="png", dpi=300, bbox_inches='tight')
    png_bytes = f_png.getvalue()
    plt.close(fig)
    return svg_string, png_bytes

def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section)
    - แก้ไข Error mutation_scale
    - เหล็กปลอกโค้งมน (Rounded) และจัดวาง Text ด้านขวา
    """
    b, h = float(res['b']), float(res['h'])
    cover = float(res['cover'])
    
    # กำหนดขนาด Figure ให้กว้างและสูงพอสมควร
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    x0, y0 = -b/2, -h/2
    
    # 1. วาดหน้าตัดคอนกรีต
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=3, zorder=1))
    
    # 2. วาดเหล็กปลอกแบบโค้ง (Rounded Stirrup)
    s_w, s_h = b - 2*cover, h - 2*cover
    # แก้ไข Syntax: mutation_scale อยู่ด้านนอก boxstyle
    stirrup = patches.FancyBboxPatch(
        (x0 + cover, y0 + cover), s_w, s_h,
        boxstyle="round,pad=0,rounding_size=10", 
        mutation_scale=1, 
        fill=False, edgecolor='#2c3e50', lw=2.5, zorder=2
    )
    ax.add_patch(stirrup)
    
    # วาด Hook (ปากเหล็กปลอก) 45 องศา สองเส้นคู่
    hook_size = 15
    ax.plot([x0+cover, x0+cover+hook_size], [y0+h-cover, y0+h-cover-hook_size], color='#2c3e50', lw=2.5, zorder=3)
    ax.plot([x0+cover+hook_size, x0+cover+hook_size], [y0+h-cover-hook_size, y0+h-cover], color='#2c3e50', lw=2.5, zorder=3)
    
    # 3. เหล็กเสริมเมน (Circles)
    n_top, db_top = int(res['top']['n']), float(res['top_db'])
    y_pos_top = (h/2) - cover - (db_top/2) - 2
    x_top = np.linspace(x0 + cover + 15, x0 + b - cover - 15, n_top) if n_top > 1 else [0]
    for x in x_top:
        ax.add_patch(patches.Circle((x, y_pos_top), db_top/2 + 1, color='#d30000', zorder=10))

    n_bot, db_bot = int(res['bot']['n']), float(res['bot_db'])
    y_pos_bot = (-h/2) + cover + (db_bot/2) + 2
    x_bot = np.linspace(x0 + cover + 15, x0 + b - cover - 15, n_bot) if n_bot > 1 else [0]
    for x in x_bot:
        ax.add_patch(patches.Circle((x, y_pos_bot), db_bot/2 + 1, color='#008c00', zorder=10))

    # 4. การจัดวางตัวหนังสือด้านขวา (Right Side Annotation)
    # เพิ่มระยะ Space ให้ห่างจากคานมากขึ้น (b * 0.45)
    text_x = b/2 + (b * 0.45) 
    
    ax.text(text_x, y_pos_top, f"{n_top}-DB{int(db_top)} (Main Top)", 
            color='#d30000', va='center', ha='left', fontsize=11, fontweight='bold')
    
    ax.text(text_x, 0, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])} (Stirrups)", 
            color='#34495e', va='center', ha='left', fontsize=10, fontweight='bold')
    
    ax.text(text_x, y_pos_bot, f"{n_bot}-DB{int(db_bot)} (Main Bot)", 
            color='#008c00', va='center', ha='left', fontsize=11, fontweight='bold')

    # 5. ชื่อ Section (เว้นระยะห่างด้านบนเพิ่มขึ้นเป็น 0.3 ของความสูง)
    ax.text(0, h/2 + (h * 0.3), f"SECTION {int(b)}x{int(h)}", 
            ha='center', va='bottom', fontweight='black', fontsize=13)
    
    # --- 6. Viewport Optimization ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # ขยับรูปขึ้นโดยบีบ ylim ล่างให้เหลือน้อยที่สุด และเปิดพื้นที่ขวาให้ตัวหนังสือ
    ax.set_ylim(-h*0.5, h*1.2)
    ax.set_xlim(-b*0.8, b*2.2)
    
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.05, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
