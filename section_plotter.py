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
    """วาดเส้น Dimension - คงเดิมทุกบรรทัด"""
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
    """วาด Support - แก้ไขตำแหน่งให้ไม่ทับคาน"""
    size = 250 
    ax.text(x, y - size - 150, f"S{sup_id}", ha='center', fontsize=9, fontweight='bold')
    
    # วาดสัญลักษณ์ Support (คง Logic เดิม แต่เน้น Z-order)
    if sup_type == 'Fixed':
        w, h = 120, 500
        rect = patches.Rectangle((x-w/2, y-h/2), w, h, facecolor='#bdc3c7', edgecolor='black', hatch='///', zorder=1)
        ax.add_patch(rect)
    elif sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=1)
        ax.add_patch(tri)
        ax.plot([x-size, x+size], [y-size, y-size], color='black', lw=2, zorder=1)
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-size/2, y-size], [x+size/2, y-size]], 
                              closed=True, facecolor='#bdc3c7', edgecolor='black', zorder=1)
        ax.add_patch(tri)
        wheel_r = 40
        ax.add_patch(patches.Circle((x, y-size-wheel_r), wheel_r, fc='white', ec='black', zorder=1))
        ax.plot([x-size, x+size], [y-size-2*wheel_r, y-size-2*wheel_r], color='black', lw=2, zorder=1)
    else:
        rect = patches.Rectangle((x-100, y-400), 200, 400, fc='#eee', ec='black', zorder=1)
        ax.add_patch(rect)

def plot_section(b_m, h_m, cover_mm, db_top_mm, db_bot_mm, n_top, n_bot, stir_text, fc, fy, title="SECTION A-A"):
    """รูปตัดขวาง - แก้ไขการเรียงเหล็กให้เป็นแนวนอนตามกว้าง b"""
    b, h = b_m * 1000.0, h_m * 1000.0
    fig, ax = _setup_figure((7, 6))
    
    # 1. Concrete (b=กว้าง, h=สูง)
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=2, ec='black', fc='#FAFAFA', zorder=1))
    
    # 2. Stirrup
    st_off = cover_mm
    ax.add_patch(patches.Rectangle((st_off, st_off), b-2*st_off, h-2*st_off, 
                                   lw=1.5, ec=COLOR_STIRRUP, fill=False, zorder=2))
    
    # 3. Bars - บังคับพิกัด X ให้กระจายตาม b
    def draw_bars(n, y, db, color):
        if n < 1: return b/2
        if n > 1:
            xs = np.linspace(st_off + 15 + db/2, b - st_off - 15 - db/2, int(n))
        else:
            xs = [b/2]
        for x in xs:
            ax.add_patch(patches.Circle((x, y), db/2, fc=color, ec='black', lw=0.8, zorder=10))
        return xs[-1]

    y_top = h - cover_mm - 10 - (db_top_mm/2)
    y_bot = cover_mm + 10 + (db_bot_mm/2)
    
    lx_t = draw_bars(n_top, y_top, db_top_mm, COLOR_TOP)
    lx_b = draw_bars(n_bot, y_bot, db_bot_mm, COLOR_BOT)
    
    # Annotations & Dims - คงเดิมจากต้นฉบับ
    ax.annotate(f"Stirrup: {stir_text}", xy=(st_off, h/2), xytext=(-120, h/2),
                arrowprops=dict(arrowstyle='->', color=COLOR_STIRRUP), ha='right', va='center')

    ax.annotate(f"{int(n_top)}-DB{int(db_top_mm)}", xy=(lx_t, y_top), xytext=(b+80, h-30),
                arrowprops=dict(arrowstyle='->', color=COLOR_TOP, connectionstyle="arc3,rad=0.1"),
                ha='left', fontweight='bold', color=COLOR_TOP)
    
    ax.annotate(f"{int(n_bot)}-DB{int(db_bot_mm)}", xy=(lx_b, y_bot), xytext=(b+80, 50),
                arrowprops=dict(arrowstyle='->', color=COLOR_BOT, connectionstyle="arc3,rad=-0.1"),
                ha='left', fontweight='bold', color=COLOR_BOT)

    _draw_dim_line(ax, (0, 0), (b, 0), f"{int(b)}", offset=-70)
    _draw_dim_line(ax, (0, 0), (0, h), f"{int(h)}", offset=70, is_vert=True)

    ax.axis('equal')
    ax.axis('off')
    ax.set_xlim(-200, b + 350)
    ax.set_ylim(-200, h + 200)
    return fig

def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_m, cover_mm):
    """รูปตัดตามยาว - แก้ไข Aspect Ratio ให้เป็นแนวนอนและวาดปลอกจริง"""
    spans_mm = [s * 1000 for s in spans]
    total_L = sum(spans_mm)
    h_mm = h_m * 1000
    
    # ปรับสัดส่วนภาพให้ยาวตามคาน
    fig_w = max(14, total_L / 450)
    fig, ax = _setup_figure((fig_w, 5)) 
    
    # 1. Beam Body
    ax.add_patch(patches.Rectangle((0, 0), total_L, h_mm, lw=2, ec='black', fc='none', zorder=10))
    
    # 2. Supports
    if not sup_df.empty:
        for _, row in sup_df.iterrows():
            _draw_support_symbol(ax, row['x']*1000, 0, row.get('type','Pin'), row.get('id',''))

    # 3. Reinforcement & Stirrups
    x_cur = 0
    y_t = h_mm - cover_mm - 15
    y_b = cover_mm + 15

    for i, span_L in enumerate(spans_mm):
        res = design_res[i]
        mid = x_cur + span_L/2
        
        # วาดเส้นเหล็กปลอกจริง (Stirrups)
        s_space = res['shear']['s']
        num_s = int(span_L / s_space)
        for sx in np.linspace(x_cur+50, x_cur+span_L-50, num_s):
            ax.plot([sx, sx], [cover_mm, h_mm-cover_mm], color=COLOR_STIRRUP, lw=0.5, alpha=0.4, zorder=2)

        # เหล็กเมนบนและล่าง (คงเดิมจากต้นฉบับ)
        ax.plot([x_cur, x_cur + span_L], [y_t, y_t], color=COLOR_TOP, lw=3, zorder=15)
        ax.plot([x_cur + 50, x_cur + span_L - 50], [y_b, y_b], color=COLOR_BOT, lw=3, zorder=15)
        
        # Labels
        ax.text(mid, h_mm + 80, f"{res['neg']['n']}-DB{int(res['top_db'])}", color=COLOR_TOP, ha='center', fontsize=8)
        ax.text(mid, y_b + 40, f"{res['pos']['n']}-DB{int(res['bot_db'])}", color=COLOR_BOT, ha='center', fontsize=8)
        ax.text(mid, -180, f"RB{int(res['stir_db'])}@{int(s_space)}", color=COLOR_STIRRUP, ha='center', fontsize=8)
        
        x_cur += span_L

    _draw_dim_line(ax, (0, h_mm), (total_L, h_mm), f"Total L = {total_L/1000:.2f} m", offset=400)
    
    ax.set_aspect('auto') # บังคับแนวนอน
    ax.axis('off')
    ax.set_xlim(-500, total_L + 500)
    ax.set_ylim(-600, h_mm + 700)
    return fig
