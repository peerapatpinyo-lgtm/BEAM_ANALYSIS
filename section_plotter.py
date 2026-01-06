import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info, fc=None, fy=None):
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    # Concrete
    rect = patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#2C3E50', facecolor='#E5E7E9')
    ax.add_patch(rect)
    
    cover = cover_mm / 1000
    db = db_mm / 1000
    
    # Stirrup
    stirrup_w = b - 2*cover
    stirrup_h = h - 2*cover
    if stirrup_w > 0 and stirrup_h > 0:
        rect_stir = patches.FancyBboxPatch((cover, cover), stirrup_w, stirrup_h,
                                           boxstyle="round,pad=0.0,rounding_size=0.01",
                                           linewidth=1.2, edgecolor='#C0392B', facecolor='none', linestyle='--')
        ax.add_patch(rect_stir)

    def draw_bars(n, y_pos, color):
        if n <= 0: return
        start_x = cover + db/2
        end_x = b - cover - db/2
        if n == 1: x_positions = [b/2]
        else:
            gap = (end_x - start_x) / (n - 1)
            x_positions = [start_x + i*gap for i in range(n)]
        for x in x_positions:
            circle = patches.Circle((x, y_pos), radius=db/2, edgecolor='black', facecolor=color, linewidth=0.8, zorder=10)
            ax.add_patch(circle)
    
    draw_bars(int(n_top), h - cover - db/2, '#E74C3C')
    draw_bars(int(n_bot), cover + db/2, '#2980B9')

    # Dimensions
    ax.annotate(f"{h:.2f}", xy=(-0.02, h/2), xytext=(-0.08, h/2), arrowprops=dict(arrowstyle='|-|'), ha='right', va='center', rotation=90)
    ax.annotate(f"{b:.2f}", xy=(b/2, -0.02), xytext=(b/2, -0.06), arrowprops=dict(arrowstyle='|-|'), ha='center', va='top')
    
    if n_top > 0: ax.text(b/2, h + 0.02, f"{n_top}-DB{db_mm}", ha='center', va='bottom', color='#E74C3C', fontweight='bold')
    if n_bot > 0: ax.text(b/2, -0.02, f"{n_bot}-DB{db_mm}", ha='center', va='top', color='#2980B9', fontweight='bold')
    
    ax.text(b + 0.02, h/2, str(stirrup_info), rotation=270, va='center', color='#C0392B')

    ax.set_xlim(-0.15, b + 0.15)
    ax.set_ylim(-0.15, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig

def draw_support_symbol(ax, x, y, sup_type, scale=1.0):
    sz = 0.2 * scale
    if sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-sz/2, y-sz], [x+sz/2, y-sz]], closed=True, edgecolor='black', facecolor='#BDC3C7')
        ax.add_patch(tri)
        ax.plot([x-sz, x+sz], [y-sz-0.02, y-sz-0.02], color='black', lw=1.5)
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-sz/2, y-sz*0.8], [x+sz/2, y-sz*0.8]], closed=True, edgecolor='black', facecolor='#BDC3C7')
        ax.add_patch(tri)
        ax.add_patch(patches.Circle((x-sz/3, y-sz*0.8-sz*0.15), sz*0.15, color='black'))
        ax.add_patch(patches.Circle((x+sz/3, y-sz*0.8-sz*0.15), sz*0.15, color='black'))
        ax.plot([x-sz, x+sz], [y-sz-sz*0.4, y-sz-sz*0.4], color='black', lw=1.5)
    elif sup_type == 'Fixed':
        ax.plot([x, x], [y-sz, y+sz], color='black', lw=2.5)
        for i in range(5):
            dy = (i - 2) * (sz*0.4)
            ax.plot([x, x-sz/3], [y+dy, y+dy-sz/3], color='black', lw=0.8)

def plot_longitudinal_section(spans, supports_df, design_data, h, cover_mm):
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    cover = cover_mm / 1000.0
    
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.add_patch(patches.Rectangle((0, 0), total_len, h, linewidth=1.5, edgecolor='black', facecolor='#FDFFE6'))
    
    y_top = h - cover
    y_bot = cover
    
    for i, span_len in enumerate(spans):
        start_x = cum_spans[i]
        end_x = cum_spans[i+1]
        data = design_data[i]
        
        # Bot Bars
        ax.plot([start_x + 0.1, end_x - 0.1], [y_bot, y_bot], color='#2980B9', linewidth=2.5)
        ax.text((start_x+end_x)/2, y_bot+0.1, f"{data['pos']['n']}-DB{data['db']}", color='#2980B9', ha='center', fontweight='bold', fontsize=8)
        
        # Top Bars (Simplified Logic)
        L_eff = span_len / 3.0
        if i == 0 and not supports_df.empty: # Check first support
             if supports_df.iloc[0]['type'] == 'Fixed':
                 ax.plot([start_x, start_x + L_eff], [y_top, y_top], color='#E74C3C', linewidth=2.5)

        if i < len(spans): # Intermediate and End
            ax.plot([end_x - L_eff, end_x], [y_top, y_top], color='#E74C3C', linewidth=2.5)
            if i < len(spans)-1:
                ax.plot([end_x, end_x + spans[i+1]/3], [y_top, y_top], color='#E74C3C', linewidth=2.5)
            
            # Label
            ax.text(end_x, y_top-0.1, f"{data['neg']['n']}-DB{data['db']}", color='#E74C3C', ha='center', fontweight='bold', fontsize=8)

    if not supports_df.empty:
        for _, sup in supports_df.iterrows():
            if sup['type'] != 'None':
                draw_support_symbol(ax, cum_spans[int(sup['id'])], 0, sup['type'], scale=h)

    ax.set_ylim(-0.5*h, h*1.5)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    plt.tight_layout()
    return fig
