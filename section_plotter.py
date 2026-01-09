import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np

# --- 🏗️ Engineering Standard Config ---
COLOR_CONCRETE = '#FFFFFF'
COLOR_DIM      = '#000000'
COLOR_STIRRUP  = '#2c3e50'
COLOR_TOP      = '#c0392b' # สีแดงเหล็กบน
COLOR_BOT      = '#27ae60' # สีเขียวเหล็กล่าง
FONT_MAIN      = 10
FONT_DIM       = 9

def _setup_figure(figsize):
    fig, ax = plt.subplots(figsize=figsize, dpi=120)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    return fig, ax

def _draw_dim_line(ax, p1, p2, text, offset=0, is_vert=False):
    """วาดเส้น Dimension (จากชุดเดิม)"""
    if is_vert:
        x_pos = p1[0] - offset
        mid_y = (p1[1] + p2[1]) / 2
        ax.annotate("", xy=(x_pos, p1[1]), xytext=(x_pos, p2[1]),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        ax.plot([p1[0], x_pos], [p1[1], p1[1]], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], x_pos], [p2[1], p2[1]], color=COLOR_DIM, lw=0.5)
        ax.text(x_pos - 15, mid_y, text, ha='right', va='center', rotation=90, fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))
    else:
        y_pos = p1[1] + offset
        mid_x = (p1[0] + p2[0]) / 2
        ax.annotate("", xy=(p1[0], y_pos), xytext=(p2[0], y_pos),
                    arrowprops=dict(arrowstyle='<|-|>', color=COLOR_DIM, lw=0.7))
        ax.plot([p1[0], p1[0]], [p1[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.plot([p2[0], p2[0]], [p2[1], y_pos], color=COLOR_DIM, lw=0.5)
        ax.text(mid_x, y_pos + 5, text, ha='center', va='bottom', fontsize=FONT_DIM,
                bbox=dict(facecolor='white', edgecolor='none', pad=2))

def _draw_support_symbol(ax, x, y, sup_type, sup_id):
    """วาดสัญลักษณ์ Support ให้ตั้งอยู่ใต้คาน (y=0 ลงไป)"""
    size = 200 
    ax.text(x, y - size - 150, f"S{sup_id}", ha='center', fontsize=9, fontweight='bold')
    
    if sup_type == 'Fixed':
        w, h = 80, 400
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor='#bdc3c7', edgecolor='black', hatch='///')
        ax.add_patch(rect)
    elif sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black')
        ax.add_patch(tri)
        ax.plot([x-size, x+size], [y-size, y-size], color='black', lw=2)
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black')
        ax.add_patch(tri)
        wheel_r = 30
        ax.add_patch(patches.Circle((x, y-size-wheel_r), wheel_r, fc='white', ec='black'))
        ax.plot([x-size, x+size], [y-size-2*wheel_r, y-size-2*wheel_r], color='black', lw=2)
    else:
        rect = patches.Rectangle((x-80, y-300), 160, 300, fc='#eee', ec='black')
        ax.add_patch(rect)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """รูปตัดขวาง: b=แนวนอน, h=แนวตั้ง"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((6, 7))
    
    # Concrete Outline
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    
    # Stirrup
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, fill=False, ls='--', zorder=2))
    
    # Rebars Placement
    def draw_bars(n, y_pos, db, color):
        if n <= 0: return b/2
        if n == 1:
            xs = [b/2]
        else:
            side_gap = st_off + 10 + db/2
            xs = np.linspace(side_gap, b - side_gap, int(n))
        for x in xs:
            ax.add_patch(patches.Circle((x, y_pos), db/2, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1]

    y_top = h - (cover_mm + 10 + db_top_mm/2)
    y_bot = cover_mm + 10 + db_bot_mm/2
    
    last_x_top = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    last_x_bot = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Annotations
    if n_top > 0:
        ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", xy=(last_x_top, y_top), 
                    xytext=(b+60, h-40), arrowprops=dict(arrowstyle='->', color=COLOR_TOP),
                    ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_TOP, fontweight='bold')
    if n_bot > 0:
        ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", xy=(last_x_bot, y_bot), 
                    xytext=(b+60, 40), arrowprops=dict(arrowstyle='->', color=COLOR_BOT),
                    ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_BOT, fontweight='bold')

    _draw_dim_line(ax, (0, 0), (b, 0), f"b={int(b)}", offset=-70)
    _draw_dim_line(ax, (0, 0), (0, h), f"h={int(h)}", offset=70, is_vert=True)

    ax.set_title(title, fontsize=12, fontweight='bold', pad=20)
    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-150, b + 250)
    ax.set_ylim(-150, h + 150)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """หน้าตัดตามยาว: แนวนอน(X) คือความยาวรวม, แนวตั้ง(Y) คือความหนาคาน"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    fig, ax = _setup_figure((14, 5)) 
    
    # 1. โครงคาน (วาดตามแนวนอน)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='none', zorder=10))
    
    # 2. Supports
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            x_pos = row['x'] * 1000
            _draw_support_symbol(ax, x_pos, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 3. Reinforcement Logic
    x_cursor = 0
    y_top_rebar = h_mm - cover_mm - 15
    y_bot_rebar = cover_mm + 15

    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid_span = x_cursor + span_L/2
        
        # วาดเหล็กปลอก (เส้นแนวตั้ง)
        s_spacing = res['shear']['s']
        num_stirrups = max(2, int(span_L / s_spacing))
        stir_x = np.linspace(x_cursor + 50, x_cursor + span_L - 50, num_stirrups)
        for sx in stir_x:
            ax.plot([sx, sx], [cover_mm, h_mm-cover_mm], color=COLOR_STIRRUP, lw=0.6, alpha=0.3)

        # วาดเหล็กเสริมพิเศษบน (Negative)
        L_neg = span_L * 0.30
        ax.plot([x_cursor, x_cursor + L_neg], [y_top_rebar, y_top_rebar], color=COLOR_TOP, lw=2.5, zorder=15)
        ax.plot([x_cursor + span_L - L_neg, x_cursor + span_L], [y_top_rebar, y_top_rebar], color=COLOR_TOP, lw=2.5, zorder=15)
        # Hanger Bar (เส้นบางประคองช่วงกลาง)
        ax.plot([x_cursor + L_neg, x_cursor + span_L - L_neg], [y_top_rebar, y_top_rebar], color=COLOR_TOP, lw=0.8, ls=':', alpha=0.5)

        # วาดเหล็กเสริมล่าง (Positive)
        ax.plot([x_cursor + 50, x_cursor + span_L - 50], [y_bot_rebar, y_bot_rebar], color=COLOR_BOT, lw=2.5, zorder=15)

        # Labels
        ax.text(mid_span, h_mm + 60, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=8, fontweight='bold')
        ax.text(mid_span, -100, f"RB{int(res['stir_db'])}@{int(s_spacing)}", color=COLOR_STIRRUP, ha='center', fontsize=8)
        ax.text(mid_span, y_bot_rebar + 30, f"{res['pos']['n']}-DB{int(res['bot_db'])}", color=COLOR_BOT, ha='center', fontsize=8)

        # Section Marker
        ax.vlines(mid_span, -50, h_mm + 120, colors='purple', linestyles='dashdot', lw=0.7)
        ax.text(mid_span, h_mm + 180, f"SEC {i+1}", color='purple', ha='center', fontsize=8, fontweight='bold')
        
        x_cursor += span_L

    # Total Dimension
    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=350)
    
    # บังคับการแสดงผลให้เป็นแนวนอนคาน
    ax.set_aspect('auto') # ปรับเพื่อให้ความยาวคานไม่โดนบีบจนเป็นแนวตั้ง
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 600)
    return fig
