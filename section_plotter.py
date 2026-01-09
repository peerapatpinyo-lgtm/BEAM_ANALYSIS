# section_plotter.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_mm, cover_mm):
    """
    วาดรูปตัดยาวคาน (Longitudinal Section) แบบละเอียด
    หากมีการเสริมเหล็กหลายชั้น จะวาดเส้นเหล็กแยกตามจำนวนชั้นจริง
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 400  # ความสูงสำหรับการวาดใน Plot
    
    fig_w = max(16, total_L / 300)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    
    # 1. วาดตัวคาน (Beam Outline)
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='#fdfdfd', antialiased=False, zorder=2)
    ax.add_patch(beam)
    
    # 2. วาด Grid Line และชื่อ Grid
    curr_x = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        ax.plot([curr_x, curr_x], [-650, v_h + 450], color='#bdc3c7', ls='--', lw=1, zorder=1)
        ax.annotate(chr(65+i), xy=(curr_x, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), 
                    fontsize=14, fontweight='bold')
        
        if i < len(spans_mm):
            ax.annotate('', xy=(curr_x, v_h + 250), xytext=(curr_x + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#34495e', lw=1.2))
            ax.text(curr_x + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", 
                    ha='center', color='#34495e', fontsize=12, fontweight='bold')
            curr_x += s_mm

    # 3. วาดจุดรองรับ (Supports)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-120, -350), 240, 350, fc='#dfe6e9', ec='black', lw=1.5, hatch='////'))
            elif stype == 'ROLLER':
                ax.add_patch(patches.Polygon([[sx, 0], [sx-100, -200], [sx+100, -200]], fc='white', ec='black', lw=1.5))
                ax.add_patch(patches.Circle((sx, -240), 40, fc='black'))
            else: # PIN
                ax.add_patch(patches.Polygon([[sx, 0], [sx-100, -200], [sx+100, -200]], fc='#2c3e50', ec='black', lw=1.5))
            ax.text(sx, -550, f"S{row['id']}", ha='center', fontweight='bold', fontsize=11)

    # 4. วาดเหล็กเสริม (Reinforcement Layers)
    x_curr = 0
    v_spacing = 30.0 # ระยะห่างระหว่างเส้นเหล็กในรูปวาด
    
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        stir_db = res.get('stir_db', 9)
        
        # --- วาดเหล็กบน (Top Reinforcement) ---
        top_layers = res.get('top', {}).get('all_layers', [])
        curr_y_top = v_h - (cover_mm + stir_db)
        for l_idx, layer in enumerate(top_layers):
            if layer['n'] > 0:
                # วาดเส้นเหล็กบน (ลากยาวตลอด Span สำหรับชั้น 1, ชั้น 2-3 อาจจะสั้นลงตาม Curtailment)
                # ในที่นี้ลากยาวให้เห็นทุกชั้นก่อน
                ax.plot([x_curr, x_curr + span_L], [curr_y_top, curr_y_top], 
                        color='#d30000', lw=2.5, zorder=10, label=f"Top L{l_idx+1}")
                curr_y_top -= v_spacing
        
        # --- วาดเหล็กล่าง (Bottom Reinforcement) ---
        bot_layers = res.get('bot', {}).get('all_layers', [])
        curr_y_bot = cover_mm + stir_db
        for l_idx, layer in enumerate(bot_layers):
            if layer['n'] > 0:
                # วาดเส้นเหล็กล่าง
                ax.plot([x_curr + 50, x_curr + span_L - 50], [curr_y_bot, curr_y_bot], 
                        color='#008c00', lw=2.5, zorder=10, label=f"Bot L{l_idx+1}")
                curr_y_bot += v_spacing
        
        # --- วาดเหล็กปลอก (Stirrups) ---
        s_spacing = res['shear'].get('s', 150)
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            stir_x = x_curr + (j * s_spacing)
            if stir_x <= x_curr + span_L:
                ax.plot([stir_x, stir_x], [cover_mm, v_h - cover_mm], 
                        color='#bdc3c7', lw=0.8, alpha=0.5, zorder=3)
        
        # ส่วน Label บอกรายละเอียด
        mid = x_curr + span_L/2
        t_label = "\n".join([f"L{idx+1}: {int(l['n'])}DB{int(l['db'])}" for idx, l in enumerate(top_layers) if l['n'] > 0])
        b_label = "\n".join([f"L{idx+1}: {int(l['n'])}DB{int(l['db'])}" for idx, l in enumerate(bot_layers) if l['n'] > 0])
        
        ax.text(mid, v_h + 50, t_label, color='#d30000', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax.text(mid, -100, b_label, color='#008c00', ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(mid, v_h/2, f"RB{int(stir_db)}@{int(s_spacing)}", color='#7f8c8d', ha='center', style='italic', fontsize=9)
        
        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-800, total_L + 800)
    ax.set_ylim(-900, v_h + 900)
    
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight', transparent=True)
    svg_string = f_svg.getvalue()
    plt.close(fig)
    return svg_string, None

def plot_cross_section(res):
    """
    (คงเดิมตามเวอร์ชันที่แก้ KeyError แล้ว)
    """
    b, h = float(res.get('b', 200)), float(res.get('h', 400))
    cover = float(res.get('cover', 25))
    stir_db = float(res.get('stir_db', 9))
    
    top_layers = res.get('top_layers') or [{'n': res.get('top', {}).get('n', 0), 'db': res.get('top_db', 16)}]
    bot_layers = res.get('bot_layers') or [{'n': res.get('bot', {}).get('n', 0), 'db': res.get('bot_db', 16)}]
    
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x0, y0 = -b/2, -h/2
    
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    s_x, s_y = x0 + cover, y0 + cover
    s_w, s_h = b - 2*cover, h - 2*cover
    ax.add_patch(patches.Rectangle((s_x, s_y), s_w, s_h, fill=False, edgecolor='#34495e', lw=1.5, zorder=2))
    
    # วาดเหล็กบนทุกลเยอร์
    v_spacing = 25.0 
    curr_y_top = (h/2) - cover - stir_db
    for l_idx, layer in enumerate(top_layers):
        n, db = int(layer.get('n', 0)), float(layer.get('db', 16))
        if n <= 0: continue
        y_pos = curr_y_top - (db/2)
        x_pos = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_pos:
            ax.add_patch(patches.Circle((x, y_pos), db/2, color='#d30000', zorder=10))
        curr_y_top -= (db + v_spacing)

    # วาดเหล็กล่างทุกลเยอร์
    curr_y_bot = (-h/2) + cover + stir_db
    for l_idx, layer in enumerate(bot_layers):
        n, db = int(layer.get('n', 0)), float(layer.get('db', 16))
        if n <= 0: continue
        y_pos = curr_y_bot + (db/2)
        x_pos = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_pos:
            ax.add_patch(patches.Circle((x, y_pos), db/2, color='#008c00', zorder=10))
        curr_y_bot += (db + v_spacing)

    # รายละเอียด Text
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
    return f.getvalue()
