# section_plotter.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_mm, cover_mm):
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 400  
    
    fig_w = max(16, total_L / 300)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    
    # 1. วาดตัวคาน
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='#fdfdfd', zorder=5)
    ax.add_patch(beam)
    
    # 2. วาด Grid Line
    x_curr = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        ax.plot([x_curr, x_curr], [-650, v_h + 450], color='#bdc3c7', ls='--', lw=1, zorder=1)
        ax.annotate(chr(65+i), xy=(x_curr, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), 
                    fontsize=14, fontweight='bold')
        if i < len(spans_mm):
            ax.annotate('', xy=(x_curr, v_h + 250), xytext=(x_curr + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#34495e', lw=1.2))
            ax.text(x_curr + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", ha='center', fontweight='bold')
            x_curr += s_mm

    # 3. วาดจุดรองรับตามประเภท (Improved Support Shapes)
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            stype = str(row.get('type', 'PIN')).upper()
            if stype == 'FIXED':
                ax.add_patch(patches.Rectangle((sx-120, -300), 240, 300, fc='#dfe6e9', ec='black', lw=1.5, hatch='///', zorder=4))
            elif stype == 'ROLLER':
                ax.add_patch(patches.Polygon([[sx, 0], [sx-100, -180], [sx+100, -180]], fc='white', ec='black', lw=1.5, zorder=4))
                ax.add_patch(patches.Circle((sx, -220), 40, fc='black', zorder=4))
            else: # PIN
                ax.add_patch(patches.Polygon([[sx, 0], [sx-100, -200], [sx+100, -200]], fc='#2c3e50', ec='black', lw=1.5, zorder=4))
            ax.text(sx, -450, f"S{row['id']}\n({stype})", ha='center', fontweight='bold', fontsize=9)

    # 4. วาดเหล็กเสริมและ Label
    x_curr = 0
    v_spacing = 35.0 
    
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        stir_db = res.get('stir_db', 9)
        
        # --- TOP REINFORCEMENT ---
        top_layers = res.get('top', {}).get('all_layers', [])
        curr_y_top = v_h - (cover_mm + stir_db)
        # วาดเส้นและเก็บข้อมูล Label (เรียง L1 บนสุดไปหาล่าง)
        t_labels = []
        for l_idx, layer in enumerate(top_layers):
            if layer['n'] > 0:
                t_labels.append(f"L{l_idx+1}: {int(layer['n'])}DB{int(layer['db'])}")
                if l_idx == 0:
                    x_s, x_e = x_curr, x_curr + span_L
                else:
                    cut_off = span_L * 0.30
                    ax.plot([x_curr, x_curr + cut_off], [curr_y_top, curr_y_top], color='#d30000', lw=2.5, zorder=10)
                    ax.plot([x_curr + span_L - cut_off, x_curr + span_L], [curr_y_top, curr_y_top], color='#d30000', lw=2.5, zorder=10)
                    x_s, x_e = None, None
                if x_s is not None:
                    ax.plot([x_s, x_e], [curr_y_top, curr_y_top], color='#d30000', lw=2.5, zorder=10)
                curr_y_top -= v_spacing
        
        # --- BOTTOM REINFORCEMENT ---
        bot_layers = res.get('bot', {}).get('all_layers', [])
        curr_y_bot = cover_mm + stir_db
        b_labels = [] # เราจะเก็บ Label เพื่อเรียงใหม่
        for l_idx, layer in enumerate(bot_layers):
            if layer['n'] > 0:
                b_labels.append(f"L{l_idx+1}: {int(layer['n'])}DB{int(layer['db'])}")
                if l_idx == 0:
                    x_s, x_e = x_curr + 50, x_curr + span_L - 50
                else:
                    offset = span_L * 0.125
                    x_s, x_e = x_curr + offset, x_curr + span_L - offset
                ax.plot([x_s, x_e], [curr_y_bot, curr_y_bot], color='#008c00', lw=2.5, zorder=10)
                curr_y_bot += v_spacing
        
        # 5. วาดเหล็กปลอก
        s_spacing = res['shear'].get('s', 150)
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            stir_x = x_curr + (j * s_spacing)
            if stir_x <= x_curr + span_L:
                ax.plot([stir_x, stir_x], [cover_mm, v_h - cover_mm], color='#bdc3c7', lw=0.7, alpha=0.5, zorder=6)

        # 6. Label เรียงลำดับ (Top: L1 บน | Bot: L1 ล่าง)
        mid = x_curr + span_L/2
        # Top Label เรียง L1 ไว้บนสุด
        ax.text(mid, v_h + 80, "\n".join(t_labels), color='#d30000', ha='center', va='bottom', fontsize=9, fontweight='bold')
        # Bot Label เรียง L1 ไว้ล่างสุด (ใช้ reversed เพื่อให้ L2, L3 อยู่บรรทัดบน L1)
        ax.text(mid, -120, "\n".join(reversed(b_labels)), color='#008c00', ha='center', va='top', fontsize=9, fontweight='bold')
        
        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-800, v_h + 800)
    
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight', transparent=True)
    svg_string = f_svg.getvalue()
    plt.close(fig)
    return svg_string, None

def plot_cross_section(res):
    """
    วาดรูปตัดขวางคาน (Cross Section) - ไม่มีการย่อหรือตัดทิ้ง
    """
    b, h = float(res.get('b', 200)), float(res.get('h', 400))
    cover = float(res.get('cover', 25))
    stir_db = float(res.get('stir_db', 9))
    
    top_layers = res.get('top_layers') or [{'n': res.get('top', {}).get('n', 0), 'db': res.get('top_db', 16)}]
    bot_layers = res.get('bot_layers') or [{'n': res.get('bot', {}).get('n', 0), 'db': res.get('bot_db', 16)}]
    
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x0, y0 = -b/2, -h/2
    
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    s_x, s_y, s_w, s_h = x0+cover, y0+cover, b-2*cover, h-2*cover
    ax.add_patch(patches.Rectangle((s_x, s_y), s_w, s_h, fill=False, edgecolor='#34495e', lw=1.5, zorder=2))
    
    # วาดเหล็กบน
    v_spacing = 25.0 
    curr_y_top = (h/2) - cover - stir_db
    for l in top_layers:
        n, db = int(l.get('n', 0)), float(l.get('db', 16))
        if n <= 0: continue
        y_p = curr_y_top - (db/2)
        x_p = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_p: ax.add_patch(patches.Circle((x, y_p), db/2, color='#d30000', zorder=10))
        curr_y_top -= (db + v_spacing)

    # วาดเหล็กล่าง (L1 อยู่ล่างสุด)
    curr_y_bot = (-h/2) + cover + stir_db
    for l in bot_layers:
        n, db = int(l.get('n', 0)), float(l.get('db', 16))
        if n <= 0: continue
        y_p = curr_y_bot + (db/2)
        x_p = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_p: ax.add_patch(patches.Circle((x, y_p), db/2, color='#008c00', zorder=10))
        curr_y_bot += (db + v_spacing)

    # Label รายละเอียด
    text_x = b/2 + (b * 0.2)
    top_t = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in top_layers if int(l.get('n',0)) > 0])
    bot_t = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in bot_layers if int(l.get('n',0)) > 0])
    
    ax.text(text_x, h/2 - cover, f"Top: {top_t}", color='#d30000', va='top', fontweight='bold')
    ax.text(text_x, 0, f"Stirrup: RB{int(stir_db)}@{int(res.get('shear', {}).get('s', 150))}", color='#34495e', va='center', fontweight='bold')
    ax.text(text_x, -h/2 + cover, f"Bot: {bot_t}", color='#008c00', va='bottom', fontweight='bold')
    ax.text(0, h/2 + (h*0.15), f"SECTION {int(b)}x{int(h)}", ha='center', fontweight='black', fontsize=12)

    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(-b*0.7, b*2.0)
    ax.set_ylim(-h*0.7, h*1.1)
    
    f = io.StringIO()
    fig.savefig(f, format="svg", bbox_inches='tight', transparent=True)
    return f.getvalue()
