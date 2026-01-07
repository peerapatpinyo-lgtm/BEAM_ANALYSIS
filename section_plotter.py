import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import numpy as np

# Set generic font style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info, fc=None, fy=None):
    """Draw Cross Section"""
    fig, ax = plt.subplots(figsize=(3.5, 4))
    
    # 1. Concrete Section
    rect = patches.Rectangle((0, 0), b, h, linewidth=1.5, edgecolor='#2C3E50', facecolor='#E5E7E9')
    ax.add_patch(rect)
    
    cover = cover_mm / 1000
    db = db_mm / 1000
    
    # 2. Stirrup
    stirrup_w = b - 2*cover
    stirrup_h = h - 2*cover
    if stirrup_w > 0 and stirrup_h > 0:
        rect_stir = patches.FancyBboxPatch((cover, cover), stirrup_w, stirrup_h,
                                           boxstyle="round,pad=0.0,rounding_size=0.01",
                                           linewidth=1.2, edgecolor='#C0392B', facecolor='none', linestyle='--')
        ax.add_patch(rect_stir)

    # 3. Main Bars
    def draw_bars(n, y_pos, color, label_prefix):
        if n <= 0: return
        start_x = cover + db/2
        end_x = b - cover - db/2
        
        if n == 1:
            x_positions = [b/2]
        else:
            gap = (end_x - start_x) / (n - 1)
            x_positions = [start_x + i*gap for i in range(n)]
            
        for x in x_positions:
            circle = patches.Circle((x, y_pos), radius=db/2, edgecolor='black', facecolor=color, linewidth=0.8, zorder=10)
            ax.add_patch(circle)
    
    # Top Bars
    y_top = h - cover - db/2
    draw_bars(int(n_top), y_top, '#E74C3C', 'Top')
    
    # Bot Bars
    y_bot = cover + db/2
    draw_bars(int(n_bot), y_bot, '#2980B9', 'Bot')

    # 4. Dimensions & Labels
    ax.annotate(f"{h:.2f}", xy=(-0.02, h/2), xytext=(-0.08, h/2),
                arrowprops=dict(arrowstyle='|-|', color='black', lw=0.8),
                ha='right', va='center', rotation=90, fontsize=9)
    
    ax.annotate(f"{b:.2f}", xy=(b/2, -0.02), xytext=(b/2, -0.06),
                arrowprops=dict(arrowstyle='|-|', color='black', lw=0.8),
                ha='center', va='top', fontsize=9)
    
    if n_top > 0:
        ax.text(b/2, h + 0.02, f"{n_top}-DB{db_mm}", ha='center', va='bottom', color='#E74C3C', fontweight='bold', fontsize=10)
    if n_bot > 0:
        ax.text(b/2, -0.02, f"{n_bot}-DB{db_mm}", ha='center', va='top', color='#2980B9', fontweight='bold', fontsize=10)

    try:
        s_val = stirrup_info.split('@')[1].strip()
        stir_lbl = f"Stir. @ {s_val}"
    except:
        stir_lbl = stirrup_info
    ax.text(b + 0.02, h/2, stir_lbl, rotation=270, va='center', color='#C0392B', fontsize=9)

    if fc and fy:
        spec_text = f"$f'_c$={fc:.0f}\n$f_y$={fy:.0f}"
        props = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray', linewidth=0.5)
        ax.text(b*0.5, h*0.4, spec_text, transform=ax.transData, fontsize=8,
                verticalalignment='center', horizontalalignment='center', bbox=props, alpha=0.6)

    ax.set_xlim(-0.15, b + 0.15)
    ax.set_ylim(-0.15, h + 0.15)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def draw_support_symbol(ax, x, y, sup_type, scale=1.0):
    sz = 0.2 * scale 
    
    if sup_type == 'Pin':
        tri = patches.Polygon([[x, y], [x-sz/2, y-sz], [x+sz/2, y-sz]], 
                              closed=True, edgecolor='black', facecolor='#BDC3C7', lw=1.5, zorder=5)
        ax.add_patch(tri)
        ax.plot([x-sz, x+sz], [y-sz-0.02, y-sz-0.02], color='black', lw=1.5)
        
    elif sup_type == 'Roller':
        tri = patches.Polygon([[x, y], [x-sz/2, y-sz*0.8], [x+sz/2, y-sz*0.8]], 
                              closed=True, edgecolor='black', facecolor='#BDC3C7', lw=1.5, zorder=5)
        ax.add_patch(tri)
        r_wheel = sz * 0.15
        c1 = patches.Circle((x-sz/3, y-sz*0.8-r_wheel), r_wheel, color='black')
        c2 = patches.Circle((x+sz/3, y-sz*0.8-r_wheel), r_wheel, color='black')
        ax.add_patch(c1)
        ax.add_patch(c2)
        ax.plot([x-sz, x+sz], [y-sz-0.05, y-sz-0.05], color='black', lw=1.5)
        
    elif sup_type == 'Fixed':
        h_wall = sz * 1.5
        ax.plot([x, x], [y-h_wall/2, y+h_wall/2], color='black', lw=2.5)
        for i in range(5):
            dy = (i - 2) * (h_wall/5)
            ax.plot([x, x-sz/3], [y+dy, y+dy-sz/3], color='black', lw=0.8)

def plot_longitudinal_section(spans, supports_df, design_data, h, cover_mm):
    """Draw Longitudinal Elevation"""
    total_len = sum(spans)
    cum_spans = [0] + list(np.cumsum(spans))
    cover = cover_mm / 1000.0
    
    fig_h = max(3, h * 3)
    fig_w = max(10, total_len * 1.2)
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # 1. Concrete Beam Outline
    beam_rect = patches.Rectangle((0, 0), total_len, h, linewidth=1.5, edgecolor='black', facecolor='#FDFFE6') 
    ax.add_patch(beam_rect)
    
    # 2. Rebar Visualization
    y_top = h - cover
    y_bot = cover
    
    for i, span_len in enumerate(spans):
        start_x = cum_spans[i]
        end_x = cum_spans[i+1]
        data = design_data[i]
        
        # Bottom Bars
        ax.plot([start_x + 0.05, end_x - 0.05], [y_bot, y_bot], color='#2980B9', linewidth=2.5, solid_capstyle='round')
        mid_x = (start_x + end_x) / 2
        ax.text(mid_x, y_bot + 0.05 * h, f"{data['pos']['n']}-DB{data['db']}", 
                color='#2980B9', ha='center', fontsize=9, fontweight='bold', backgroundcolor='white')
        
        # Top Bars (Schematic L/3)
        L_eff = span_len / 3.0 
        
        if i == 0: 
            sup_type = supports_df[supports_df['id'] == 0]['type'].values[0] if not supports_df.empty else 'Pin'
            if sup_type == 'Fixed':
                ax.plot([start_x, start_x + L_eff], [y_top, y_top], color='#E74C3C', linewidth=2.5)
        else: 
            ax.plot([start_x - L_eff, start_x + L_eff], [y_top, y_top], color='#E74C3C', linewidth=2.5)
            n_top_val = data['neg']['n'] 
            ax.text(start_x, y_top - 0.15 * h, f"{n_top_val}-DB{data['db']}", 
                    color='#E74C3C', ha='center', fontsize=9, fontweight='bold', backgroundcolor='white')

        if i == len(spans) - 1:
            sup_type = supports_df[supports_df['id'] == len(spans)]['type'].values[0] if not supports_df.empty else 'Pin'
            if sup_type == 'Fixed':
                ax.plot([end_x - L_eff, end_x], [y_top, y_top], color='#E74C3C', linewidth=2.5)

    # 3. Supports
    if not supports_df.empty:
        for _, sup in supports_df.iterrows():
            sx = cum_spans[int(sup['id'])]
            stype = sup['type']
            if stype != 'None':
                draw_support_symbol(ax, sx, 0, stype, scale=h)
                ax.text(sx, -0.3*h, f"{stype}", ha='center', fontsize=8, color='#555')

    # 4. Stirrups
    for i in range(len(spans)):
        sx = cum_spans[i] + 0.2
        ex = cum_spans[i+1] - 0.2
        for k in range(4):
            ax.plot([sx + k*0.15, sx + k*0.15], [cover, h-cover], color='#C0392B', lw=0.5, alpha=0.5)
            ax.plot([ex - k*0.15, ex - k*0.15], [cover, h-cover], color='#C0392B', lw=0.5, alpha=0.5)

    ax.set_ylim(-0.5*h, h*1.5)
    ax.set_xlim(-0.5, total_len + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title("Longitudinal Elevation (Reinforcement Profile)", loc='left', fontsize=11, fontweight='bold', pad=10)
    
    plt.tight_layout()
    return fig
