# section_plotter.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import io
import numpy as np
import textwrap # สำหรับตัดบรรทัดตัวอักษร

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_mm, cover_mm):
    """
    วาดรูปตัดยาวคาน พร้อมระบบตัดบรรทัด Label และลดพื้นที่ว่างส่วนเกิน
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    v_h = 400  
    
    fig_w = max(16, total_L / 300)
    # ปรับ figsize ให้กระชับขึ้น
    fig, ax = plt.subplots(figsize=(fig_w, 5))
    
    # 1. วาดตัวคาน
    beam = patches.Rectangle((0, 0), total_L, v_h, lw=2, ec='black', fc='#fdfdfd', zorder=5)
    ax.add_patch(beam)
    
    # 2. วาด Grid Line และระยะช่วงคาน
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

    # 3. วาดเหล็กเสริม (Longitudinal Reinforcement)
    x_curr = 0
    v_spacing = 35.0 
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        stir_db = res.get('stir_db', 9)
        
        # --- TOP REINFORCEMENT ---
        top_layers = res.get('top', {}).get('all_layers', [])
        curr_y_top = v_h - (cover_mm + stir_db)
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
        b_labels = [] 
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
        
        # 4. Label เหล็ก (Top: L1 บนสุด | Bot: L1 ล่างสุด)
        mid = x_curr + span_L/2
        ax.text(mid, v_h + 80, "\n".join(t_labels), color='#d30000', ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(mid, -120, "\n".join(reversed(b_labels)), color='#008c00', ha='center', va='top', fontsize=9, fontweight='bold')
        x_curr += span_L

    ax.set_aspect('auto')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    # ปรับ ylim ให้ชิดขอบตัวหนังสือมากขึ้นเพื่อลดพื้นที่สีขาว
    ax.set_ylim(-700, v_h + 700) 
    
    f_svg = io.StringIO()
    fig.savefig(f_svg, format="svg", bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    return f_svg.getvalue(), None

def plot_cross_section(res):
    """
    วาดรูปตัดขวางคานแบบตัดบรรทัดอัตโนมัติ 
    ลดพื้นที่ขาวด้านบน และขยายขอบล่างเฉพาะเมื่อมี Warning
    """
    b, h = float(res.get('b', 200)), float(res.get('h', 400))
    cover = float(res.get('cover', 25))
    stir_db = float(res.get('stir_db', 9))
    
    top_layers = res.get('top_layers') or [{'n': res.get('top', {}).get('n', 0), 'db': res.get('top_db', 16)}]
    bot_layers = res.get('bot_layers') or [{'n': res.get('bot', {}).get('n', 0), 'db': res.get('bot_db', 16)}]
    
    # ปรับสัดส่วนรูปภาพให้กระชับ
    fig, ax = plt.subplots(figsize=(6, 5)) 
    x0, y0 = -b/2, -h/2
    
    # 1. วาดคอนกรีตและเหล็กปลอก
    ax.add_patch(patches.Rectangle((x0, y0), b, h, fc='white', ec='black', lw=2.5, zorder=1))
    s_x, s_y, s_w, s_h = x0+cover, y0+cover, b-2*cover, h-2*cover
    ax.add_patch(patches.Rectangle((s_x, s_y), s_w, s_h, fill=False, ec='#34495e', lw=1.5, zorder=2))
    
    warnings = []
    
    # 2. วาดเหล็กบน + เช็ค Spacing
    curr_y_top = (h/2) - cover - stir_db
    for idx, l in enumerate(top_layers):
        n, db = int(l.get('n', 0)), float(l.get('db', 16))
        if n > 0:
            if n > 1:
                h_space = (s_w - 2*stir_db - n*db) / (n - 1)
                if h_space < max(25, db):
                    warnings.append(f"Top L{idx+1}: {h_space:.1f}mm")
            y_p = curr_y_top - (db/2)
            x_p = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
            for x in x_p: 
                ax.add_patch(patches.Circle((x, y_p), db/2, color='#d30000', zorder=10))
            curr_y_top -= (db + 25.0)

    # 3. วาดเหล็กล่าง + เช็ค Spacing
    curr_y_bot = (-h/2) + cover + stir_db
    for idx, l in enumerate(bot_layers):
        n, db = int(l.get('n', 0)), float(l.get('db', 16))
        if n > 0:
            if n > 1:
                h_space = (s_w - 2*stir_db - n*db) / (n - 1)
                if h_space < max(25, db):
                    warnings.append(f"Bot L{idx+1}: {h_space:.1f}mm")
            y_p = curr_y_bot + (db/2)
            x_p = np.linspace(s_x + stir_db + db/2, s_x + s_w - stir_db - db/2, n) if n > 1 else [0]
            for x in x_p: 
                ax.add_patch(patches.Circle((x, y_p), db/2, color='#008c00', zorder=10))
            curr_y_bot += (db + 25.0)

    # 4. จัดการ Warning พร้อมตัดบรรทัด (ขยายพื้นที่ล่างอัตโนมัติ)
    if warnings:
        warn_msg = "⚠️ SPACING WARNING: " + " | ".join(warnings)
        # ตัดบรรทัดทุกๆ 45 ตัวอักษร
        wrapped_warn = "\n".join(textwrap.wrap(warn_msg, width=45))
        
        # วาง Warning ไว้กึ่งกลางล่าง (ใช้ transform เพื่อให้ตำแหน่งสัมพัทธ์กับรูปภาพ)
        ax.text(0.5, -0.05, wrapped_warn, transform=ax.transAxes,
                color='white', fontweight='bold', fontsize=9,
                ha='center', va='top', 
                bbox=dict(boxstyle="round,pad=0.4", fc='#d30000', ec='none'))

    # 5. Label รายละเอียดเหล็ก (ตัดบรรทัดถ้าชื่อเหล็กยาว)
    text_x = b/2 + 20
    top_t = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in top_layers if int(l.get('n',0)) > 0])
    bot_t = " + ".join([f"{int(l['n'])}DB{int(l['db'])}" for l in bot_layers if int(l.get('n',0)) > 0])
    
    ax.text(text_x, h/2 - 10, f"Top:\n{textwrap.fill(top_t, 18)}", color='#d30000', va='top', fontweight='bold', fontsize=9)
    ax.text(text_x, -h/2 + 10, f"Bot:\n{textwrap.fill(bot_t, 18)}", color='#008c00', va='bottom', fontweight='bold', fontsize=9)
    # ชื่อ Section วางให้ชิดขอบบนของคาน
    ax.text(0, h/2 + 20, f"SECTION {int(b)}x{int(h)}", ha='center', va='bottom', fontweight='black', fontsize=11)

    ax.set_aspect('equal')
    ax.axis('off')

    # 6. คุม Margin ให้พอดี (ลดพื้นที่ขาวด้านบน และเผื่อล่างเมื่อมี Warning)
    upper_lim = h/2 + (h * 0.25)
    lower_lim = -h/2 - (h * 0.45 if warnings else h * 0.15)
    
    ax.set_xlim(-b*0.8, b*1.8)
    ax.set_ylim(lower_lim, upper_lim) 
    
    f = io.StringIO()
    # ตัดขอบขาวรอบนอกออกด้วย pad_inches
    fig.savefig(f, format="svg", bbox_inches='tight', pad_inches=0.1, transparent=True)
    plt.close(fig)
    return f.getvalue()
