import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    วาดรูปตัดยาวคาน (Longitudinal Section) พร้อมรายละเอียดเหล็กเสริมและระยะห่าง
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
    วาดรูปตัดขวางคาน (Cross Section) แก้ไขปัญหารูปโดนตัดครึ่งและจัดตำแหน่งใหม่ให้สมดุล
    """
    b = float(res['b'])
    h = float(res['h'])
    cover = float(res['cover'])
    
    # 1. ตั้งค่า Figure ให้สมดุล
    fig, ax = plt.subplots(figsize=(5, 5))
    
    # ใช้ระบบพิกัดที่ให้ 0,0 อยู่ตรงกลางคานเพื่อให้การกระจาย Text รอบข้างทำได้ง่าย
    x0, y0 = -b/2, -h/2
    
    # 2. วาดหน้าตัดคอนกรีต
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    
    # 3. วาดเหล็กปลอก (Stirrup)
    stir_off = cover
    ax.add_patch(patches.Rectangle((x0 + stir_off, y0 + stir_off), b - 2*stir_off, h - 2*stir_off, 
                                   fill=False, edgecolor='#2c3e50', lw=1.5, zorder=2))
    
    # 4. วาดและระบุเหล็กเมนบน (Top Bars)
    n_top = int(res['top']['n'])
    db_top = float(res['top_db'])
    y_pos_top = (h/2) - stir_off - (db_top/2) - 2
    # กระจายเหล็กบน
    if n_top > 1:
        x_top = np.linspace(x0 + stir_off + 12, x0 + b - stir_off - 12, n_top)
    else:
        x_top = [0]
    
    for x in x_top:
        ax.add_patch(patches.Circle((x, y_pos_top), db_top/2 + 1, color='#d30000', zorder=10))
    
    # Label เหล็กบน (วางเหนือคาน)
    ax.text(0, h/2 + (h*0.1), f"{n_top}-DB{int(db_top)}", color='#d30000', 
            ha='center', va='bottom', fontweight='bold', fontsize=11)

    # 5. วาดและระบุเหล็กเมนล่าง (Bottom Bars)
    n_bot = int(res['bot']['n'])
    db_bot = float(res['bot_db'])
    y_pos_bot = (-h/2) + stir_off + (db_bot/2) + 2
    # กระจายเหล็กล่าง
    if n_bot > 1:
        x_bot = np.linspace(x0 + stir_off + 12, x0 + b - stir_off - 12, n_bot)
    else:
        x_bot = [0]
    
    for x in x_bot:
        ax.add_patch(patches.Circle((x, y_pos_bot), db_bot/2 + 1, color='#008c00', zorder=10))
        
    # Label เหล็กล่าง (วางใต้คาน)
    ax.text(0, -h/2 - (h*0.1), f"{n_bot}-DB{int(db_bot)}", color='#008c00', 
            ha='center', va='top', fontweight='bold', fontsize=11)

    # 6. ข้อความหัวข้อ (Section Name) และเหล็กปลอก
    ax.text(0, h/2 + (h*0.3), f"SECTION {int(b)}x{int(h)} mm", ha='center', fontweight='black', fontsize=13)
    ax.text(0, -h/2 - (h*0.3), f"Stirrup: RB{int(res['stir_db'])}@{int(res['shear']['s'])}", 
            ha='center', color='#34495e', fontsize=10, fontweight='bold')
    
    # --- 7. ปรับ Viewport ให้สมดุลและไม่โดนตัดขอบ ---
    ax.set_aspect('equal')
    ax.axis('off')
    
    # กำหนดขอบเขตการแสดงผล (Margin) ให้พอดีกับข้อความทั้งบนและล่าง
    # เพิ่มระยะแนวตั้ง (ylim) ให้มากขึ้นเพื่อไม่ให้ตัวหนังสือ Stirrup หาย
    v_margin = h * 0.5
    h_margin = b * 0.3
    ax.set_ylim(-h/2 - v_margin, h/2 + v_margin)
    ax.set_xlim(-b/2 - h_margin, b/2 + h_margin)
    
    f = io.StringIO()
    # ใช้ bbox_inches='tight' และเพิ่ม pad_inches เล็กน้อยเพื่อป้องกันขอบตัวหนังสือขาด
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.15, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
