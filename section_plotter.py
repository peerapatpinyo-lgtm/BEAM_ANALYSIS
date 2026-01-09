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
    ax.text(x, y - size - 150, f"S{sup_id}", ha='center', fontsize=9, fontweight='bold')
    
    if sup_type == 'Fixed':
        w, h = 120, 500
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor='#bdc3c7', edgecolor='black', hatch='///', zorder=5)
        ax.add_patch(rect)
    elif sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=5)
        ax.add_patch(tri)
        ax.plot([x-size, x+size], [y-size, y-size], color='black', lw=2, zorder=6)
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=5)
        ax.add_patch(tri)
        wheel_r = 40
        ax.add_patch(patches.Circle((x-size/4, y-size-wheel_r), wheel_r, fc='white', ec='black', zorder=6))
        ax.add_patch(patches.Circle((x+size/4, y-size-wheel_r), wheel_r, fc='white', ec='black', zorder=6))
        ax.plot([x-size, x+size], [y-size-2*wheel_r, y-size-2*wheel_r], color='black', lw=2, zorder=5)
    else:
        rect = patches.Rectangle((x-100, y-400), 200, 400, fc='#eee', ec='black', zorder=4)
        ax.add_patch(rect)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """ Cross Section Plot """
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((7, 6))
    
    # 1. Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    
    # 2. Stirrup Outline (Closed Loop)
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, fill=False, zorder=2))
    
    # 3. Rebars Placement Logic
    def draw_bars(n, y, db, color):
        if n < 1: return
        # Calculate horizontal spacing
        usable_w = b - 2*st_off - 2*(db/2) - 10 # 10mm buffer from stirrup corner
        if n > 1:
            xs = np.linspace(st_off + 10 + db/2, b - st_off - 10 - db/2, int(n))
        else:
            xs = [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1]

    y_top = h - cover_mm - 10 - (db_top_mm/2)
    y_bot = cover_mm + 10 + (db_bot_mm/2)
    
    last_x_top = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    last_x_bot = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Annotations
    ax.annotate(f"Stirrup: {stir_text}", xy=(st_off, h/2), xytext=(-120, h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP),
                ha='right', va='center', fontsize=FONT_MAIN, color=COLOR_STIRRUP)

    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", xy=(last_x_top if n_top>0 else b/2, y_top), 
                xytext=(b+80, h-30), arrowprops=dict(arrowstyle='->', color=COLOR_TOP, connectionstyle="arc3,rad=0.1"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_TOP, fontweight='bold')
    
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", xy=(last_x_bot if n_bot>0 else b/2, y_bot), 
                xytext=(b+80, 50), arrowprops=dict(arrowstyle='->', color=COLOR_BOT, connectionstyle="arc3,rad=-0.1"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_BOT, fontweight='bold')

    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-70)
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=70, is_vert=True)

    ax.text(b/2, -180, f"fc': {fc} MPa | fy: {fy} MPa", ha='center', fontsize=9, 
            bbox=dict(facecolor='#f8f9fa', edgecolor='#ccc', boxstyle='round,pad=0.5'))

    ax.set_title(title, fontsize=12, fontweight='bold', pad=30)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-200, b + 300)
    ax.set_ylim(-250, h + 200)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ Detailed Longitudinal Section with Real Stirrups """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    fig, ax = _setup_figure((14, 7)) 
    
    # 1. Beam Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='none', zorder=10))
    
    # 2. Supports
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            x = row['x'] * 1000
            _draw_support_symbol(ax, x, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 3. Reinforcement & Stirrups
    x_cursor = 0
    top_y = h_mm - cover_mm - 15
    bot_y = cover_mm + 15

    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid_span = x_cursor + span_L/2
        
        # --- วาดเหล็กปลอก (Stirrups) ตามระยะจริง ---
        s_spacing = res['shear']['s']
        num_stirrups = int(span_L / s_spacing)
        stirrup_locs = np.linspace(x_cursor + 50, x_cursor + span_L - 50, num_stirrups)
        for sx in stirrup_locs:
            ax.plot([sx, sx], [cover_mm, h_mm-cover_mm], color=COLOR_STIRRUP, lw=0.6, alpha=0.5, zorder=2)

        # --- Top Reinforcement (Negative @ Supports) ---
        L_neg = span_L * 0.30 
        # Left Support Bar
        ax.plot([x_cursor, x_cursor + L_neg], [top_y, top_y], color=COLOR_TOP, lw=3, solid_capstyle='round', zorder=15)
        # Right Support Bar
        ax.plot([x_cursor + span_L - L_neg, x_cursor + span_L], [top_y, top_y], color=COLOR_TOP, lw=3, solid_capstyle='round', zorder=15)
        # Hanger Bar (ประคองปลอกช่วงกลาง)
        ax.plot([x_cursor + L_neg, x_cursor + span_L - L_neg], [top_y, top_y], color=COLOR_TOP, lw=1, ls='--', alpha=0.7, zorder=14)
        
        # Label Top
        ax.text(mid_span, h_mm + 80, f"{res['neg']['n']}-DB{int(res['top_db'])} (Top)", 
                color=COLOR_TOP, ha='center', fontweight='bold', fontsize=8, bbox=dict(fc='white', ec=COLOR_TOP, lw=0.5))

        # --- Bottom Reinforcement (Main @ Span) ---
        ax.plot([x_cursor + 50, x_cursor + span_L - 50], [bot_y, bot_y], color=COLOR_BOT, lw=3, solid_capstyle='round', zorder=15)
        ax.text(mid_span, bot_y + 40, f"{res['pos']['n']}-DB{int(res['bot_db'])}", 
                color=COLOR_BOT, ha='center', fontweight='bold', fontsize=8)

        # --- Stirrup Label ---
        stir_text = f"RB{int(res['stir_db'])}@{int(s_spacing)}"
        ax.annotate(stir_text, xy=(mid_span, cover_mm), xytext=(mid_span, -180),
                    arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP, lw=0.8),
                    ha='center', fontsize=8, color=COLOR_STIRRUP)

        # Section Cut Symbol
        ax.vlines(mid_span, -50, h_mm+50, colors='#9b59b6', linestyles='dashdot', lw=1, alpha=0.6)
        ax.text(mid_span, h_mm+250, f"SEC {i+1}", color='#9b59b6', ha='center', fontsize=9, fontweight='bold')
        
        x_cursor += span_L

    # Dimensions
    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"TOTAL L = {total_L/1000:.2f} m", offset=450)
    
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-600, total_L + 600)
    ax.set_ylim(-800, h_mm + 800)
    return fig
