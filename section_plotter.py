import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as mpath
import numpy as np

# --- 🏗️ Engineering Standard Config (Original) ---
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
    """วาดเส้น Dimension แบบมาตรฐาน - รักษา Logic เดิมของคุณ"""
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
    """วาดสัญลักษณ์ Support ตามหลักวิศวกรรม - ปรับตำแหน่งให้พ้นแนวคาน"""
    size = 250 
    ax.text(x, y - size - 150, str(sup_id), ha='center', fontsize=9, fontweight='bold')
    
    if sup_type == 'Fixed':
        w, h = 100, 450
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor='#bdc3c7', edgecolor='black', hatch='///', zorder=5)
        ax.add_patch(rect)
    elif sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=5)
        ax.add_patch(tri)
        ax.add_patch(patches.Circle((x, y), 20, fc='white', ec='black', zorder=10))
        ax.plot([x-size, x+size], [y-size, y-size], color='black', lw=2, zorder=6)
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=5)
        ax.add_patch(tri)
        wheel_r = 30
        ax.add_patch(patches.Circle((x-size/3, y-size-wheel_r), wheel_r, fc='white', ec='black', zorder=6))
        ax.add_patch(patches.Circle((x+size/3, y-size-wheel_r), wheel_r, fc='white', ec='black', zorder=6))
        ax.plot([x-size, x+size], [y-size-2*wheel_r, y-size-2*wheel_r], color='black', lw=2, zorder=5)
    else:
        rect = patches.Rectangle((x-100, y-350), 200, 350, fc='#eee', ec='black', zorder=4)
        ax.add_patch(rect)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """ Cross Section - แก้ไขทิศทางให้ b=แนวนอน, h=แนวตั้ง """
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((6, 6))
    
    # Concrete & Stirrup
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, ls='--', fill=False, zorder=2))
    
    def draw_bars(n, y, db, color):
        if n < 1: return b/2
        # บังคับให้เรียงเหล็กตามแนวนอน (แกน X)
        xs = np.linspace(st_off + 15 + db/2, b - st_off - 15 - db/2, int(n)) if n > 1 else [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1]

    y_top = h - cover_mm - 10 - (db_top_mm/2)
    y_bot = cover_mm + 10 + (db_bot_mm/2)
    
    lx_t = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    lx_b = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Labels (คงเดิมตามของคุณ)
    ax.annotate(f"Stirrup: {stir_text}", xy=(st_off, h/2), xytext=(-80, h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP),
                ha='right', va='center', fontsize=FONT_MAIN, color=COLOR_STIRRUP)

    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)} (Top)", xy=(lx_t, y_top), xytext=(b+60, h-30),
                arrowprops=dict(arrowstyle='->', color=COLOR_TOP, connectionstyle="arc3,rad=0.2"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_TOP, fontweight='bold')
    
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)} (Bot)", xy=(lx_b, y_bot), xytext=(b+60, 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_BOT, connectionstyle="arc3,rad=-0.2"),
                ha='left', va='center', fontsize=FONT_MAIN, color=COLOR_BOT, fontweight='bold')

    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-70)
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=70, is_vert=True)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-150, b + 250)
    ax.set_ylim(-150, h + 150)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """ Longitudinal Section - แก้ไขให้เป็นแนวนอนและไม่ทับกัน """
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # ปรับสัดส่วนรูปภาพให้ยาวตามความยาวคาน (Width > Height)
    fig_w = max(14, total_L / 400)
    fig, ax = _setup_figure((fig_w, 5)) 
    
    # 1. Beam Body (แนวนอน X)
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='#FFFFFF', zorder=1))
    
    # 2. Supports
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('type', 'Pin'), row.get('id', ''))

    # 3. Reinforcement
    x_cursor = 0
    y_top_main = h_mm - cover_mm - 20
    y_bot_main = cover_mm + 20

    for i, span_L in enumerate(spans_mm):
        if i >= len(design_res): break
        res = design_res[i]
        mid_span = x_cursor + span_L/2
        
        # เหล็กบน (Negative)
        L_neg = span_L * 0.25
        ax.plot([x_cursor, x_cursor + L_neg], [y_top_main, y_top_main], color=COLOR_TOP, lw=3, zorder=15)
        ax.plot([x_cursor + span_L - L_neg, x_cursor + span_L], [y_top_main, y_top_main], color=COLOR_TOP, lw=3, zorder=15)
        ax.plot([x_cursor + L_neg, x_cursor + span_L - L_neg], [y_top_main, y_top_main], color=COLOR_TOP, lw=1, ls=':', zorder=14)
        
        # เหล็กล่าง (Positive)
        ax.plot([x_cursor + 50, x_cursor + span_L - 50], [y_bot_main, y_bot_main], color=COLOR_BOT, lw=3, zorder=15)
        
        # Labels
        ax.text(mid_span, h_mm + 80, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontweight='bold', fontsize=8)
        ax.text(mid_span, y_bot_main + 50, f"{res['pos']['n']}-DB{int(res['bot_db'])}", color=COLOR_BOT, ha='center', fontweight='bold', fontsize=8)
        
        # Stirrup (ขยับลงด้านล่าง)
        s_val = res['shear']['s']
        ax.text(mid_span, -250, f"RB{int(res['stir_db'])}@{int(s_val)}", color=COLOR_STIRRUP, ha='center', fontsize=8)
        
        x_cursor += span_L

    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=400)

    # หัวใจสำคัญ: เปลี่ยนเป็น auto เพื่อให้คานนอนราบ
    ax.set_aspect('auto') 
    ax.axis('off')
    ax.set_xlim(-600, total_L + 600)
    ax.set_ylim(-800, h_mm + 800)
    return fig
