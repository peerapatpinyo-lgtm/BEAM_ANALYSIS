# section_plotter.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_mm, cover_mm):
    """
    วาดรูปตัดยาวคาน (Longitudinal Section) - รองรับการแสดงผลจำนวนชั้นเหล็ก
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 350  
    
    fig_w = max(16, total_L / 350)
    fig, ax = plt.subplots(figsize=(fig_w, 4.5))
    
    # 1. วาดตัวคาน
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='white', antialiased=False, zorder=2)
    ax.add_patch(beam)
    
    # 2. วาด Grid Line
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

    # 3. วาดจุดรองรับ
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

    # 4. วาดเหล็กเสริม
    y_t_base, y_b_base = v_h * 0.85, v_h * 0.15
    x_curr = 0
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        s_spacing = res['shear'].get('s', 150)
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            stir_x = x_curr + (j * s_spacing)
            if stir_x <= x_curr + span_L:
                ax.plot([stir_x, stir_x], [y_b_base - 20, y_t_base + 20], color='#bdc3c7', lw=0.7, alpha=0.6, zorder=3)
        
        mid = x_curr + span_L/2
        ax.text(mid, -150, f"RB{int(res.get('stir_db', 9))}@{int(s_spacing)}", color='#7f8c8d', fontsize=9, ha='center', style='italic')
        
        ax.plot([x_curr, x_curr + span_L], [y_t_base, y_t_base], color='#d30000', lw=3.5, zorder=10)
        ax.plot([x_curr + 40, x_curr + span_L - 40], [y_b_base, y_b_base], color='#008c00', lw=3.5, zorder=10)
        
        label_opt = dict(ha='center', fontweight='bold', fontsize=11, bbox=dict(facecolor='white', edgecolor='none', alpha=0.85))
        
        t_info = f"{int(res.get('neg', {}).get('n', 0))}-DB{int(res.get('top_db', 16))} ({res.get('top', {}).get('layers', 1)} Layers)"
        b_info = f"{int(res.get('pos', {}).get('n', 0))}-DB{int(res.get('bot_db', 16))} ({res.get('bot', {}).get('layers', 1)} Layers)"
        
        ax.text(mid, v_h + 80, t_info, color='#d30000', **label_opt)
        ax.text(mid, y_b_base - 50, b_info, color='#008c00', va='top', **label_opt)
        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-1000, total_L + 1000)
    ax.set_ylim(-800, v_h + 800)
    
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight')
    svg_string = f_svg.getvalue()
    plt.close(fig)
    return svg_string, None

def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section) - รองรับเหล็กหลายชั้น และป้องกัน KeyError
    """
    b, h = float(res.get('b', 200)), float(res.get('h', 400))
    cover = float(res.get('cover', 25))
    stir_db = float(res.get('stir_db', 9))
    
    # ปรับการดึงข้อมูลเพื่อป้องกัน KeyError: 'top'
    top_layers = res.get('top_layers')
    if top_layers is None:
        # ถ้าไม่มี top_layers ให้เช็ค top แบบเก่า ถ้าไม่มีอีกให้ใช้ค่าว่าง
        old_top_n = res.get('top', {}).get('n', 0)
        top_layers = [{'n': old_top_n, 'db': res.get('top_db', 16)}]
        
    bot_layers = res.get('bot_layers')
    if bot_layers is None:
        old_bot_n = res.get('bot', {}).get('n', 0)
        bot_layers = [{'n': old_bot_n, 'db': res.get('bot_db', 16)}]
    
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x0, y0 = -b/2, -h/2
    
    # 1. วาดคอนกรีต
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    
    # 2. วาดเหล็กปลอก
    s_x, s_y = x0 + cover, y0 + cover
    s_w, s_h = b - 2*cover, h - 2*cover
    ax.add_patch(patches.Rectangle((s_x, s_y), s_w, s_h, fill=False, edgecolor='#34495e', lw=1.5, zorder=2))
    
    # 3. วาดเหล็กบน
    v_spacing = 25.0 
    curr_y_top = (h/2) - cover - stir_db
    for l_idx, layer in enumerate(top_layers):
        n = int(layer.get('n', 0))
        db = float(layer.get('db', 16))
        if n <= 0: continue
        y_pos = curr_y_top - (db/2)
        x_pos = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_pos:
            ax.add_patch(patches.Circle((x, y_pos), db/2, color='#d30000', zorder=10))
        curr_y_top -= (db + v_spacing)

    # 4. วาดเหล็กล่าง
    curr_y_bot = (-h/2) + cover + stir_db
    for l_idx, layer in enumerate(bot_layers):
        n = int(layer.get('n', 0))
        db = float(layer.get('db', 16))
        if n <= 0: continue
        y_pos = curr_y_bot + (db/2)
        x_pos = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_pos:
            ax.add_patch(patches.Circle((x, y_pos), db/2, color='#008c00', zorder=10))
        curr_y_bot += (db + v_spacing)

    # 5. Labels
    text_x_start = b/2 + (b * 0.2)
    top_label = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in top_layers if int(l.get('n',0)) > 0])
    ax.text(text_x_start, h/2 - cover, f"Top: {top_label if top_label else 'None'}", color='#d30000', va='top', fontweight='bold')
    
    s_info = res.get('shear', {})
    ax.text(text_x_start, 0, f"Stirrup: RB{int(stir_db)}@{int(s_info.get('s', 150))}", color='#34495e', va='center', fontweight='bold')
    
    bot_label = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in bot_layers if int(l.get('n',0)) > 0])
    ax.text(text_x_start, -h/2 + cover, f"Bot: {bot_label if bot_label else 'None'}", color='#008c00', va='bottom', fontweight='bold')

    ax.text(0, h/2 + (h*0.15), f"SECTION {int(b)}x{int(h)}", ha='center', fontweight='black', fontsize=12)
    
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-b*0.7, b*2.0)
    ax.set_ylim(-h*0.7, h*1.1)
    
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.1, transparent=True)
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
