import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดยาวคาน (Longitudinal Section)
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350  
    
    fig_w = max(16, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', antialiased=False, zorder=2)
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
        ax.plot([x_curr, x_curr + span_L], [y_t, y_t], color='#d30000', lw=3.5, zorder=10, antialiased=False)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b, y_b], color='#008c00', lw=3.5, zorder=10, antialiased=False)
        
        label_opt = dict(ha='center', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        ax.text(mid, v_h + 80, f"{int(res['neg']['n'])}-DB{int(res['top_db'])} (TOP)", color='#d30000', **label_opt)
        ax.text(mid, y_b - 50, f"{int(res['pos']['n'])}-DB{int(res['bot_db'])} (BOT)", color='#008c00', va='top', **label_opt)
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
    วาดรูปตัดขวางคาน (Cross Section) จัดกลุ่มตัวหนังสือไว้ด้านขวาทั้งหมด
    และเว้นระยะห่างระหว่างรูปกับตัวหนังสือให้ชัดเจน
    """
    b, h = float(res['b']), float(res['h'])
    cover = float(res['cover'])
    
    # กำหนดขนาด Figure (เพิ่มความกว้างเพื่อรองรับ Space ด้านขวา)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    x0, y0 = -b/2, -h/2
    
    # 1. วาดหน้าตัดคอนกรีต
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    
    # 2. วาดเหล็กปลอก
    ax.add_patch(patches.Rectangle((x0 + cover, y0 + cover), b - 2*cover, h - 2*cover, 
                                   fill=False, edgecolor='#34495e', lw=1.2, zorder=2))
    
    # 3. เหล็กบน (Top Bars)
    n_top = int(res['top']['n'])
    db_top = float(res['top_db'])
    y_pos_top = (h/2) - cover - (db_top/2) - 1.5
    x_top = np.linspace(x0 + cover + 10, x0 + b - cover - 10, n_top) if n_top > 1 else [0]
    for x in x_top:
        ax.add_patch(patches.Circle((x, y_pos_top), db_top/2 + 1.2, color='#d30000', zorder=10))

    # 4. เหล็กเมนล่าง (Bottom Bars)
    n_bot = int(res['bot']['n'])
    db_bot = float(res['bot_db'])
    y_pos_bot = (-h/2) + cover + (db_bot/2) + 1.5
    x_bot = np.linspace(x0 + cover + 10, x0 + b - cover - 10, n_bot) if n_bot > 1 else [0]
    for x in x_bot:
        ax.add_patch(patches.Circle((x, y_pos_bot), db_bot/2 + 1.2, color='#008c00', zorder=10))

    # --- ส่วนการจัดวางตัวหนังสือด้านขวา (Right Side Annotation) ---
    # เว้น Space จากขอบคานด้านขวาออกไป (b*0.15)
    text_x_start = b/2 + (b * 0.15)
    
    # ระบุเหล็กบน
    ax.text(text_x_start, y_pos_top, f"{n_top}-DB{int(db_top)} (TOP)", 
            color='#d30000', va='center', ha='left', fontsize=10, fontweight='bold')
    
    # ระบุเหล็กปลอก (Stirrup) - วางไว้ตรงกลางความสูงคาน
    ax.text(text_x_start, 0, f"RB{int(res['stir_db'])}@{int(res['shear']['s'])} (STIR.)", 
            color='#34495e', va='center', ha='left', fontsize=9, fontweight='bold')
    
    # ระบุเหล็กล่าง
    ax.text(text_x_start, y_pos_bot, f"{n_bot}-DB{int(db_bot)} (BOT)", 
            color='#008c00', va='center', ha='left', fontsize=10, fontweight='bold')

    # 5. ชื่อ Section (วางไว้กึ่งกลางคานด้านบน)
    ax.text(0, h/2 + (h*0.05), f"SECTION {int(b)}x{int(h)}", 
            ha='center', va='bottom', fontweight='black', fontsize=11)
    
    # --- 6. Viewport Optimization ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # บีบแนว Y ให้รูปขยับขึ้น (ylim ด้านล่างน้อย ด้านบนเผื่อชื่อ Section)
    ax.set_ylim(-h/2 - (h*0.1), h/2 + (h*0.25))
    
    # ขยายแนว X ด้านขวาให้กว้างกว่าด้านซ้ายเพื่อให้เห็นตัวหนังสือครบและมี Space
    ax.set_xlim(-b/2 - (b*0.1), b/2 + (b*0.8))
    
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.01, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
