# section_plotter.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_mm, cover_mm):
    """
    วาดรูปตัดยาวคาน พร้อม Logic การหยุดเหล็ก (Bar Curtailment)
    และปรับปรุงตำแหน่ง Support รวมถึงลำดับชั้นเหล็กล่าง
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 400  
    
    fig_w = max(16, total_L / 300)
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    
    # 1. วาดตัวคาน (Beam Outline)
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='#fdfdfd', zorder=5)
    ax.add_patch(beam)
    
    # 2. วาด Grid Line และระยะ Span
    x_curr = 0
    for i, s_mm in enumerate(spans_mm + [0]):
        # เส้น Center Line
        ax.plot([x_curr, x_curr], [-650, v_h + 450], color='#bdc3c7', ls='--', lw=1, zorder=1)
        # สัญลักษณ์ Grid
        ax.annotate(chr(65+i), xy=(x_curr, v_h + 500), ha='center', va='center',
                    bbox=dict(boxstyle='circle', fc='white', ec='black', lw=1.5), 
                    fontsize=14, fontweight='bold')
        if i < len(spans_mm):
            ax.annotate('', xy=(x_curr, v_h + 250), xytext=(x_curr + s_mm, v_h + 250),
                        arrowprops=dict(arrowstyle='<->', color='#34495e', lw=1.2))
            ax.text(x_curr + s_mm/2, v_h + 300, f"{s_mm/1000:.2f} m", ha='center', fontweight='bold')
            x_curr += s_mm

    # 3. วาดจุดรองรับ (Improved Supports) - วาดเป็นเสาประคองใต้คาน
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            sx = row['x'] * 1000
            # วาดเสาขนาด 200mm รองรับใต้คานพอดี
            ax.add_patch(patches.Rectangle((sx-100, -300), 200, 300, fc='#dfe6e9', ec='black', lw=1.5, zorder=4))
            ax.text(sx, -450, f"S{row['id']}", ha='center', fontweight='bold', fontsize=11)

    # 4. วาดเหล็กเสริมแยกตาม Layer พร้อม Curtailment
    x_curr = 0
    v_spacing = 30.0 
    
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        stir_db = res.get('stir_db', 9)
        
        # --- TOP REINFORCEMENT ---
        # วาด L1 อยู่บนสุด แล้วไล่ L2, L3 ลงมา
        top_layers = res.get('top', {}).get('all_layers', [])
        curr_y_top = v_h - (cover_mm + stir_db)
        for l_idx, layer in enumerate(top_layers):
            if layer['n'] > 0:
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
        
        # --- BOTTOM REINFORCEMENT (สลับลำดับ L1 อยู่ล่างสุด) ---
        bot_layers = res.get('bot', {}).get('all_layers', [])
        curr_y_bot = cover_mm + stir_db
        for l_idx, layer in enumerate(bot_layers):
            if layer['n'] > 0:
                if l_idx == 0:
                    # L1: เหล็กเมน อยู่ล่างสุด ลากเกือบเต็ม Span
                    x_s, x_e = x_curr + 50, x_curr + span_L - 50
                else:
                    # L2, L3: เหล็กเสริมพิเศษ ซ้อนขึ้นข้างบน และหยุดเหล็ก (Curtailment)
                    offset = span_L * 0.125
                    x_s, x_e = x_curr + offset, x_curr + span_L - offset
                
                ax.plot([x_s, x_e], [curr_y_bot, curr_y_bot], color='#008c00', lw=2.5, zorder=10)
                # เลื่อนระดับขึ้นสำหรับชั้นถัดไป (L2 อยู่เหนือ L1)
                curr_y_bot += v_spacing
        
        # 5. วาดเหล็กปลอก (Stirrups)
        s_spacing = res['shear'].get('s', 150)
        num_stirrups = int(span_L / s_spacing)
        for j in range(num_stirrups + 1):
            stir_x = x_curr + (j * s_spacing)
            if stir_x <= x_curr + span_L:
                ax.plot([stir_x, stir_x], [cover_mm, v_h - cover_mm], color='#bdc3c7', lw=0.8, alpha=0.5, zorder=6)

        # 6. Label รายละเอียด
        mid = x_curr + span_L/2
        t_label = "\n".join([f"L{idx+1}: {int(l['n'])}DB{int(l['db'])}" for idx, l in enumerate(top_layers) if l['n'] > 0])
        b_label = "\n".join([f"L{idx+1}: {int(l['n'])}DB{int(l['db'])}" for idx, l in enumerate(bot_layers) if l['n'] > 0])
        
        ax.text(mid, v_h + 80, t_label, color='#d30000', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(mid, -120, b_label, color='#008c00', ha='center', va='top', fontsize=9, fontweight='bold')
        
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
    วาดรูปตัดขวางคาน (Cross Section) - ห้ามตัดทิ้งเด็ดขาด
    """
    b, h = float(res.get('b', 200)), float(res.get('h', 400))
    cover = float(res.get('cover', 25))
    stir_db = float(res.get('stir_db', 9))
    
    top_layers = res.get('top_layers') or [{'n': res.get('top', {}).get('n', 0), 'db': res.get('top_db', 16)}]
    bot_layers = res.get('bot_layers') or [{'n': res.get('bot', {}).get('n', 0), 'db': res.get('bot_db', 16)}]
    
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    x0, y0 = -b/2, -h/2
    
    # 1. วาดคอนกรีต
    ax.add_patch(patches.Rectangle((x0, y0), b, h, facecolor='#ffffff', edgecolor='black', lw=2.5, zorder=1))
    
    # 2. วาดเหล็กปลอก
    s_x, s_y, s_w, s_h = x0+cover, y0+cover, b-2*cover, h-2*cover
    ax.add_patch(patches.Rectangle((s_x, s_y), s_w, s_h, fill=False, edgecolor='#34495e', lw=1.5, zorder=2))
    
    # 3. วาดเหล็กบน
    v_spacing = 25.0 
    curr_y_top = (h/2) - cover - stir_db
    for l in top_layers:
        n, db = int(l.get('n', 0)), float(l.get('db', 16))
        if n <= 0: continue
        y_p = curr_y_top - (db/2)
        x_p = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_p:
            ax.add_patch(patches.Circle((x, y_p), db/2, color='#d30000', zorder=10))
        curr_y_top -= (db + v_spacing)

    # 4. วาดเหล็กล่าง (L1 อยู่ล่างสุด)
    curr_y_bot = (-h/2) + cover + stir_db
    for l in bot_layers:
        n, db = int(l.get('n', 0)), float(l.get('db', 16))
        if n <= 0: continue
        y_p = curr_y_bot + (db/2)
        x_p = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
        for x in x_p:
            ax.add_patch(patches.Circle((x, y_p), db/2, color='#008c00', zorder=10))
        curr_y_bot += (db + v_spacing)

    # 5. ใส่รายละเอียด
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
    svg_string = f.getvalue()
    plt.close(fig)
    return svg_string
