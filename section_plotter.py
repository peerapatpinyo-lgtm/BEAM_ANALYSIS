import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np

# --- 🏗️ Engineering Standard Config ---
COLOR_CONCRETE = '#FFFFFF'
COLOR_DIM      = '#000000'
COLOR_STIRRUP  = '#2c3e50'
COLOR_TOP      = '#c0392b'
COLOR_BOT      = '#27ae60'
FONT_MAIN      = 10
FONT_DIM       = 9

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_dim_line(ax, p1, p2, text, offset=0, is_vert=False):
    """วาดเส้น Dimension แบบมาตรฐาน"""
    if is_vert:
        x_pos = p1[0] - offset
        mid_y = (p1[1] + p2[1]) / 2
        ax.annotate("", xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        ax.plot([p1[0], x_pos], [p1[1], p1[1]], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], x_pos], [p2[1], p2[1]], color=COLOR_DIM, lw=0.5)
        ax.text(x_pos - 10, mid_y, text, ha='right', va='center', rotation=90, fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
    else:
        y_pos = p1[1] + offset
        mid_x = (p1[0] + p2[0]) / 2
        ax.annotate("", xy=(p1[0], y_pos), xytext=(p2[0], y_pos),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        ax.plot([p1[0], p1[0]], [p1[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], p2[0]], [p2[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.text(mid_x, y_pos, text, ha='center', va='center', fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

def _draw_support_symbol(ax, x, y, sup_type, sup_id):
    """วาดสัญลักษณ์ Support ตามหลักวิศวกรรม (Pin, Roller, Fixed)"""
    size = 250 # mm base size
    
    # Text ID
    ax.text(x, y - size - 100, str(sup_id), ha='center', fontsize=9, fontweight='bold')
    
    if sup_type == 'Fixed':
        # Draw Vertical Wall
        w, h = 100, 400
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor='#bdc3c7', edgecolor='black', hatch='///')
        ax.add_patch(rect)
    
    elif sup_type == 'Pin':
        # Triangle
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black')
        ax.add_patch(tri)
        # Hinge Circle
        ax.add_patch(patches.Circle((x, y), 20, fc='white', ec='black', zorder=10))
        # Ground Line with Hatch
        ax.plot([x-size, x+size], [y-size, y-size], color='black', lw=2)
        # Hatching lines below
        for i in range(int(x-size), int(x+size), 50):
            ax.plot([i, i-30], [y-size, y-size-30], color='black', lw=0.5)

    elif sup_type == 'Roller':
        # Triangle
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black')
        ax.add_patch(tri)
        # Hinge Circle
        ax.add_patch(patches.Circle((x, y), 20, fc='white', ec='black', zorder=10))
        # Wheels (Circles)
        wheel_r = 30
        ax.add_patch(patches.Circle((x-size/3, y-size-wheel_r), wheel_r, fc='white', ec='black'))
        ax.add_patch(patches.Circle((x+size/3, y-size-wheel_r), wheel_r, fc='white', ec='black'))
        # Ground Line
        g_y = y - size - 2*wheel_r
        ax.plot([x-size, x+size], [g_y, g_y], color='black', lw=2)
        
    else: # Default column
        rect = patches.Rectangle((x-100, y-300), 200, 300, fc='#eee', ec='black')
        ax.add_patch(rect)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """ Cross Section """
    b = b_m * 1000.0
    h = h_m * 1000.0
    
    fig, ax = _setup_figure((7, 6))
    
    # Concrete & Stirrup
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, ls='--', fill=False, zorder=2))
    
    # Rebars
    def draw_bars(n, y, db, color):
        if n < 2: n = 2
        xs = np.linspace(st_off + db, b - st_off - db, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1]

    y_top = h - cover_mm - 10 - (db_top_mm/2)
    y_bot = cover_mm + 10 + (db_bot_mm/2)
    
    last_x_top = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    last_x_bot = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Labels (Smart Offset)
    # Stirrup - ขยับหนีคาน
    ax.annotate(f"Stirrup: {stir_text}", xy=(st_off, h/2), xytext=(-100, h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP),
                ha='right', va='center', fontsize=FONT_MAIN, color=COLOR_STIRRUP)

    # Top Bars
    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", xy=(last_x_top, y_top), xytext=(b+80, h-30),
                arrowprops=dict(arrowstyle='->', color=COLOR_TOP, connectionstyle="arc3,rad=0.2"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_TOP, fontweight='bold')
    
    # Bot Bars
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", xy=(last_x_bot, y_bot), xytext=(b+80, 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_BOT, connectionstyle="arc3,rad=-0.2"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_BOT, fontweight='bold')

    # Dimensions
    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-60)
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=60, is_vert=True)

    # Material Info
    ax.text(b + 100, h/2, f"Cover: {cover_mm} mm\nfc': {fc} MPa\nfy: {fy} MPa", 
            fontsize=9, color='#555', bbox=dict(facecolor='#f0f0f0', edgecolor='none', pad=5))

    ax.set_title(title, fontsize=12, fontweight='bold', pad=25)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-150, b + 250)
    ax.set_ylim(-150, h + 150)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ Longitudinal Section - Corrected """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    fig, ax = _setup_figure((12, 6)) # เพิ่มความสูง Canvas
    
    # 1. Beam Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='#FFFFFF', zorder=1))
    
    # 2. Supports (Engineering Symbols)
    for _, row in sup_df.iterrows():
        x = row['x'] * 1000
        _draw_support_symbol(ax, x, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 3. Reinforcement
    x_cursor = 0
    # **ยกเส้นบอกระยะรวมขึ้นไปสูงๆ**
    dim_offset_top = 400 
    
    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        end_cursor = x_cursor + span_L
        mid_span = x_cursor + span_L/2
        
        # --- Top Bars (Support) ---
        top_y = h_mm - cover_mm - 25
        L_neg = span_L * 0.25
        
        # Draw Lines
        ax.plot([x_cursor, x_cursor + L_neg], [top_y, top_y], color=COLOR_TOP, lw=3, solid_capstyle='round')
        ax.plot([end_cursor - L_neg, end_cursor], [top_y, top_y], color=COLOR_TOP, lw=3, solid_capstyle='round')
        ax.plot([x_cursor + L_anch, end_cursor - L_anch], [top_y, top_y], color=COLOR_TOP, lw=0.8, ls=':') # Hanger
        
        # Text Top (อยู่สูงกว่า Dimension Line ไปอีก หรืออยู่ใต้เส้น Dimension เล็กน้อย แต่เหนือคาน)
        # แก้ปัญหาทับกัน: ให้ text อยู่เหนือเส้นเหล็กขึ้นไปเยอะๆ (ใต้ Dimension หลัก)
        text_y_top = h_mm + 150 
        target_x = end_cursor - L_neg/2
        
        ax.annotate(f"{res['neg']['n']}-DB{int(res['top_db'])}", 
                    xy=(target_x, top_y), xytext=(target_x, text_y_top),
                    arrowprops=dict(arrowstyle='->', color=COLOR_TOP, lw=1),
                    ha='center', va='center', color=COLOR_TOP, fontweight='bold', fontsize=9,
                    bbox=dict(fc='white', ec='none', pad=1))

        # --- Bottom Bars (Mid) ---
        bot_y = cover_mm + 25
        ax.plot([x_cursor + 80, end_cursor - 80], [bot_y, bot_y], color=COLOR_BOT, lw=3)
        ax.text(mid_span, bot_y + 50, f"{res['pos']['n']}-DB{int(res['bot_db'])}", 
                color=COLOR_BOT, ha='center', fontweight='bold', fontsize=9)

        # --- Stirrup ---
        # ย้ายตำแหน่งไม่ให้ทับคาน
        stir_text = f"RB{int(res['stir_db'])}@{int(res['shear']['s'])}"
        ax.text(mid_span, -150, f"Stir: {stir_text}", color=COLOR_STIRRUP, ha='center', va='top', fontsize=9)
        # Leader line ชี้ไปที่กลางคาน (ไม่ทับเส้นขอบล่าง)
        ax.annotate("", xy=(mid_span, cover_mm), xytext=(mid_span, -140), 
                    arrowprops=dict(arrowstyle='-', color=COLOR_STIRRUP, lw=0.5, linestyle=':'))

        # --- Section Cuts ---
        ax.vlines(mid_span, -100, h_mm+100, colors='purple', linestyles='dashdot', lw=1)
        ax.text(mid_span, h_mm+120, "A", color='purple', ha='center', fontweight='bold')
        
        sec_b = end_cursor - 150
        ax.vlines(sec_b, -100, h_mm+100, colors='orange', linestyles='dashdot', lw=1)
        ax.text(sec_b, h_mm+120, "B", color='orange', ha='center', fontweight='bold')

        x_cursor += span_L

    # 4. Total Dimension (อยู่สูงที่สุด)
    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=dim_offset_top)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 700) # เพิ่มพื้นที่ด้านบน
    
    return fig
