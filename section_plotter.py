import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np

# --- 🏗️ Engineering Standard Config ---
COLOR_CONCRETE = '#FFFFFF'      # พื้นหลัง
COLOR_COLUMN   = '#e0e0e0'      # สีเสา/ตอม่อ
COLOR_STIRRUP  = '#2c3e50'      # สีเหล็กปลอก (Dark Blue/Grey)
COLOR_TOP      = '#c0392b'      # สีเหล็กบน (Red)
COLOR_BOT      = '#27ae60'      # สีเหล็กล่าง (Green)
COLOR_DIM      = '#000000'      # สีเส้นบอกระยะ
FONT_MAIN      = 10
FONT_DIM       = 9

def _setup_figure(figsize):
    """Canvas Setup"""
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_break_line(ax, x, y, width, height):
    """วาดเส้น Zig-Zag (Break Line) ที่ตีนเสา"""
    zigzag_h = height * 0.1
    path_data = [
        (mpath.Path.MOVETO, (x - width/2, y)),
        (mpath.Path.LINETO, (x - width/2, y + height)), # Left edge
        (mpath.Path.LINETO, (x + width/2, y + height)), # Top edge (connect to beam)
        (mpath.Path.LINETO, (x + width/2, y)), # Right edge
        # Zig Zag Bottom
        (mpath.Path.LINETO, (x + width/4, y - zigzag_h)),
        (mpath.Path.LINETO, (x, y + zigzag_h)),
        (mpath.Path.LINETO, (x - width/4, y - zigzag_h)),
        (mpath.Path.LINETO, (x - width/2, y)),
        (mpath.Path.CLOSEPOLY, (x - width/2, y)),
    ]
    codes, verts = zip(*path_data)
    path = mpath.Path(verts, codes)
    patch = patches.PathPatch(path, facecolor=COLOR_COLUMN, edgecolor='black', lw=1, zorder=0)
    ax.add_patch(patch)

def _draw_dim_line(ax, p1, p2, text, offset=0, is_vert=False):
    """วาดเส้น Dimension แบบสถาปัตย์/วิศวะ (ตัวเลขอยู่กลางเส้น)"""
    if is_vert:
        # Vertical Dimension
        x_pos = p1[0] - offset
        mid_y = (p1[1] + p2[1]) / 2
        
        # Dimension Line
        ax.annotate("", xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        # Extension Lines
        ax.plot([p1[0], x_pos], [p1[1], p1[1]], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], x_pos], [p2[1], p2[1]], color=COLOR_DIM, lw=0.5)
        # Text
        ax.text(x_pos - 10, mid_y, text, ha='right', va='center', rotation=90, fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
    else:
        # Horizontal Dimension
        y_pos = p1[1] + offset
        mid_x = (p1[0] + p2[0]) / 2
        
        # Dimension Line
        ax.annotate("", xy=(p1[0], y_pos), xytext=(p2[0], y_pos),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        # Extension Lines
        ax.plot([p1[0], p1[0]], [p1[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], p2[0]], [p2[1], y_pos], color=COLOR_DIM, lw=0.5)
        # Text
        ax.text(mid_x, y_pos, text, ha='center', va='center', fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """
    Cross Section Generation
    """
    b = b_m * 1000.0
    h = h_m * 1000.0
    
    fig, ax = _setup_figure((7, 6))
    
    # 1. Concrete (Beam Face)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    
    # 2. Stirrup (เส้นประด้านใน)
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, ls='--', fill=False, zorder=2))
    
    # --- Rebar Logic ---
    def draw_bars(n, y, db, color):
        if n < 2: n = 2
        r = db/2
        # Spread bars evenly
        xs = np.linspace(st_off + db, b - st_off - db, int(n)) if n > 1 else [b/2]
        for x in xs:
            circle = patches.Circle((x, y), r, fc=color, ec='black', lw=0.8, zorder=10)
            ax.add_patch(circle)
        return xs[-1] # Return right-most bar X

    # Draw Bars
    y_top = h - cover_mm - 10 - (db_top_mm/2)
    y_bot = cover_mm + 10 + (db_bot_mm/2)
    
    last_x_top = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    last_x_bot = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # --- Professional Labels (หลบเส้น ทับกันให้น้อยที่สุด) ---
    
    # Label: Top Bars (ดึง Leader ขึ้นบนขวา)
    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", 
                xy=(last_x_top, y_top), xytext=(b + 80, h - 30),
                arrowprops=dict(arrowstyle='->', color=COLOR_TOP, connectionstyle="arc3,rad=0.2"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_TOP, fontweight='bold')

    # Label: Bottom Bars (ดึง Leader ลงล่างขวา)
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", 
                xy=(last_x_bot, y_bot), xytext=(b + 80, 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_BOT, connectionstyle="arc3,rad=-0.2"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_BOT, fontweight='bold')

    # Label: Stirrup (ดึง Leader ไปทางซ้าย)
    ax.annotate(f"Stirrup: {stir_text}", 
                xy=(st_off, h/2), xytext=(-80, h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP),
                ha='right', va='center', fontsize=FONT_MAIN, color=COLOR_STIRRUP)

    # --- Dimensions ---
    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-50)         # Width
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=50, is_vert=True) # Height

    # Info Box
    info = f"Cover: {cover_mm} mm\nfc': {fc} MPa\nfy: {fy} MPa"
    ax.text(b + 100, h/2, info, fontsize=9, color='#555', ha='left', va='center',
            bbox=dict(facecolor='#f0f0f0', edgecolor='none', pad=5))

    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-100, b + 250)
    ax.set_ylim(-100, h + 100)
    
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """
    Longitudinal Section - Professional Grade
    """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    fig, ax = _setup_figure((12, 5.5))
    
    # 1. Beam Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='#FFFFFF', zorder=1))
    
    # 2. Supports as COLUMNS (เสา) - ไม่ใช่สามเหลี่ยม
    sup_w = 300 # ความกว้างเสา mm
    sup_h = 400 # ความสูงตอม่อ mm
    
    for _, row in sup_df.iterrows():
        x = row['x'] * 1000
        # วาดรูปเสาพร้อม Break line ด้านล่าง
        _draw_break_line(ax, x, -sup_h, sup_w, sup_h)
        # Label ชื่อเสา
        ax.text(x, -sup_h - 40, str(row.get('id','')), ha='center', fontsize=9, fontweight='bold')

    # 3. Reinforcement & Details
    x_cursor = 0
    dim_offset_top = 250  # ระยะเส้นบอกขนาดด้านบน
    
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        end_cursor = x_cursor + span_L
        mid_span = x_cursor + span_L/2
        
        # --- A. TOP BARS (เหล็กบน) ---
        # วาดเส้นเหล็ก
        top_y = h_mm - cover_mm - 20
        # Logic: เหล็กบนเน้นที่หัวเสา (Support) แต่ใน Span Draw ให้เห็นว่ามี (Simplified Line)
        # ถ้าจะให้เหมือนจริง เหล็กบนมักจะหยุดที่ L/3 หรือ L/4
        L_anch = span_L * 0.25
        
        # วาดเส้นเหล็กบน ช่วงซ้าย (หัวเสาซ้าย)
        ax.plot([x_cursor, x_cursor + L_anch], [top_y, top_y], color=COLOR_TOP, lw=3, solid_capstyle='round')
        # วาดเส้นเหล็กบน ช่วงขวา (หัวเสาขวา)
        ax.plot([end_cursor - L_anch, end_cursor], [top_y, top_y], color=COLOR_TOP, lw=3, solid_capstyle='round')
        # วาดเส้นประเชื่อม (แสดง Hanger Bars หรือเหล็กยึดปลอก)
        ax.plot([x_cursor + L_anch, end_cursor - L_anch], [top_y, top_y], color=COLOR_TOP, lw=1, ls=':')
        
        # ** FIX: TEXT เหล็กบน ดึงขึ้นไปเหนือ Dimension Line **
        # ใช้ลูกศรชี้ลงมาที่เหล็ก ช่วงหัวเสาขวา (จุดที่รับโมเมนต์ลบจริง)
        target_x_top = end_cursor - L_anch/2
        text_y_top = h_mm + dim_offset_top + 80 # อยู่สูงกว่าเส้นบอกระยะ
        
        ax.annotate(f"{res['neg']['n']}-DB{int(res['top_db'])}", 
                    xy=(target_x_top, top_y), xytext=(target_x_top, text_y_top),
                    arrowprops=dict(arrowstyle='->', color=COLOR_TOP, lw=1),
                    ha='center', va='center', color=COLOR_TOP, fontweight='bold', fontsize=9,
                    bbox=dict(facecolor='white', edgecolor='none', pad=1))

        # --- B. BOTTOM BARS (เหล็กล่าง) ---
        bot_y = cover_mm + 20
        # วาดเส้นเหล็กยาวตลอดช่วง (Main Reinforcement)
        ax.plot([x_cursor + 50, end_cursor - 50], [bot_y, bot_y], color=COLOR_BOT, lw=3, solid_capstyle='round')
        
        # Text เหล็กล่าง (วางเหนือเส้นนิดหน่อย ในตัวคาน อ่านง่าย)
        ax.text(mid_span, bot_y + 50, f"{res['pos']['n']}-DB{int(res['bot_db'])}", 
                color=COLOR_BOT, ha='center', fontsize=9, fontweight='bold')

        # --- C. STIRRUP (เหล็กปลอก) ---
        # ย้าย Text ลงมาใต้คาน ไม่ให้ทับเหล็ก
        stir_text = f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}"
        ax.text(mid_span, -120, f"Stir: {stir_text}", color=COLOR_STIRRUP, ha='center', va='top', fontsize=9)
        # เส้นชี้บ่งบอกช่วง
        ax.plot([mid_span, mid_span], [0, -110], color=COLOR_STIRRUP, lw=0.5, ls=':')

        # --- D. Section Cuts ---
        # Line A-A (Mid Span)
        ax.vlines(mid_span, -100, h_mm+100, colors='purple', linestyles='dashdot', lw=1)
        ax.text(mid_span, h_mm+120, "A", color='purple', ha='center', fontweight='bold')
        
        # Line B-B (Support Face)
        sec_b = end_cursor - 150
        ax.vlines(sec_b, -100, h_mm+100, colors='orange', linestyles='dashdot', lw=1)
        ax.text(sec_b, h_mm+120, "B", color='orange', ha='center', fontweight='bold')

        x_cursor += span_L

    # 4. Total Dimension (อยู่เหนือคาน แบบสถาปัตย์)
    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=dim_offset_top)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-400, total_L + 400)
    ax.set_ylim(-600, h_mm + 600) # เพิ่มพื้นที่ด้านบนให้ Text เหล็กบน
    
    return fig
