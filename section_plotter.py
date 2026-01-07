import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def plot_section(b, h, cover_mm, db_mm, n_top, n_bot, stirrup_info, fc, fy):
    fig, ax = plt.subplots(figsize=(3.5, 4))
    ax.add_patch(patches.Rectangle((0, 0), b, h, lw=1.5, ec='#2C3E50', fc='#E5E7E9'))
    
    c, db = cover_mm/1000, db_mm/1000
    if b-2*c > 0: ax.add_patch(patches.FancyBboxPatch((c, c), b-2*c, h-2*c, boxstyle="round,pad=0,rounding_size=0.01", lw=1.2, ec='#C0392B', fc='none', ls='--'))
    
    def draw_bars(n, y, color):
        if n <= 0: return
        xs = [b/2] if n==1 else np.linspace(c+db/2, b-c-db/2, n)
        for x in xs: ax.add_patch(patches.Circle((x, y), db/2, ec='black', fc=color, zorder=10))
            
    draw_bars(int(n_top), h-c-db/2, '#E74C3C')
    draw_bars(int(n_bot), c+db/2, '#2980B9')
    
    ax.text(b/2, -0.05, f"{n_bot}-DB{db_mm}", ha='center', color='#2980B9', weight='bold')
    ax.text(b/2, h+0.05, f"{n_top}-DB{db_mm}", ha='center', color='#E74C3C', weight='bold')
    ax.axis('off'); ax.set_aspect('equal')
    plt.tight_layout()
    return fig

def plot_longitudinal_section(spans, sup_df, design_data, h, cover_mm):
    cum_spans = [0] + list(np.cumsum(spans))
    total_len = cum_spans[-1]
    fig, ax = plt.subplots(figsize=(10, 3))
    
    ax.add_patch(patches.Rectangle((0, 0), total_len, h, lw=1.5, ec='black', fc='#FDFFE6'))
    c = cover_mm/1000.0
    
    for i, span in enumerate(spans):
        sx, ex = cum_spans[i], cum_spans[i+1]
        d = design_data[i]
        
        # Bot Bars with Hooks
        hook = 0.15
        ax.plot([sx+0.05, sx+0.05, ex-0.05, ex-0.05], [c+hook, c, c, c+hook], color='#2980B9', lw=2)
        ax.text((sx+ex)/2, c+0.1, f"{d['pos']['n']}-DB16", color='#2980B9', ha='center', size=8, weight='bold')
        
        # Top Bars
        Le = span/3
        if i==0: ax.plot([sx, sx+Le], [h-c, h-c], color='#E74C3C', lw=2)
        else: 
            ax.plot([sx-Le, sx+Le], [h-c, h-c], color='#E74C3C', lw=2)
            ax.text(sx, h-c-0.1, f"{d['neg']['n']}-DB16", color='#E74C3C', ha='center', size=8)

    # Supports
    for _, s in sup_df.iterrows():
        if s['type'] != 'None':
            x = cum_spans[int(s['id'])]
            ax.add_patch(patches.Polygon([[x,0], [x-0.1,-0.15], [x+0.1,-0.15]], closed=True, fc='gray'))

    ax.set_xlim(-0.5, total_len+0.5); ax.set_ylim(-0.3, h+0.2)
    ax.axis('off'); ax.set_aspect('equal')
    plt.tight_layout()
    return fig
def plot_longitudinal_section_detailed(spans, sup_df, design_res, h_beam, cover=40):
    """
    Plots a detailed longitudinal section of the beam with accurate supports and rebar.
    """
    fig, ax = plt.subplots(figsize=(12, 3))
    
    cum_dist = [0] + list(pd.Series(spans).cumsum())
    total_length = cum_dist[-1]
    h_m = h_beam
    cover_m = cover / 1000.0
    
    # 1. Draw Beam Body
    beam_rect = patches.Rectangle((0, 0), total_length, h_m, linewidth=1.5, edgecolor='black', facecolor='#FFFFF0')
    ax.add_patch(beam_rect)
    
    # 2. Draw Supports (Realistic)
    sup_types = sup_df.set_index('node_id')['type'].to_dict()
    for i, x in enumerate(cum_dist):
        sType = sup_types.get(i, 'Pin')
        
        # Support triangle geometry
        tri_w = total_length * 0.025
        tri_h = h_m * 0.2
        
        if sType == 'Pin':
            # Triangle
            triangle = patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray')
            ax.add_patch(triangle)
        elif sType == 'Roller':
            # Triangle + Circles underneath
            triangle = patches.Polygon([[x, 0], [x - tri_w/2, -tri_h], [x + tri_w/2, -tri_h]], closed=True, color='gray')
            ax.add_patch(triangle)
            # Rollers
            circle1 = patches.Circle((x - tri_w/3, -tri_h - tri_w/4), tri_w/4, color='gray')
            circle2 = patches.Circle((x + tri_w/3, -tri_h - tri_w/4), tri_w/4, color='gray')
            ax.add_patch(circle1)
            ax.add_patch(circle2)
            # Ground line
            ax.plot([x - tri_w, x + tri_w], [-tri_h - tri_w/2, -tri_h - tri_w/2], color='black', lw=1)
        elif sType == 'Fixed':
            # Hatch pattern for fixed wall
            ax.plot([x, x], [-tri_h, h_m + tri_h], color='black', lw=2) # Wall line
            # Hatches
            for h_y in np.arange(-tri_h, h_m + tri_h, tri_h/2):
                 ax.plot([x, x-tri_w/2], [h_y, h_y-tri_h/2], color='gray', lw=1)


    # 3. Draw Rebar (Realistic based on design_res)
    for i, d_data in enumerate(design_res):
        x_start = cum_dist[i]
        x_end = cum_dist[i+1]
        span_len = spans[i]
        
        # --- Bottom Bars (Positive Moment) ---
        n_bot = d_data['pos']['n']
        if n_bot > 0:
            # Draw from start to end of span, offset by cover
            y_bot = cover_m
            # Add small hook/anchorage offset at ends
            anchorage = span_len * 0.02 
            ax.plot([x_start + anchorage, x_end - anchorage], [y_bot, y_bot], 
                    color='#1f77b4', lw=2.5, solid_capstyle='round')
            # Label (Mid-span)
            ax.text(x_start + span_len/2, y_bot + 0.05*h_m, f"{n_bot}-DB{d_data.get('db',16)}", 
                    ha='center', va='bottom', color='#1f77b4', fontsize=9, fontweight='bold')

    # --- Top Bars (Negative Moment over interior supports) ---
    # Iterate through INTERIOR supports
    for i in range(1, len(cum_dist) - 1):
        x_sup = cum_dist[i]
        
        # Get design data for adjacent spans
        span_prev = design_res[i-1]
        span_next = design_res[i]
        
        # Use the maximum top bars required by either adjacent span
        n_top = max(span_prev['neg']['n'], span_next['neg']['n'])
        
        if n_top > 0:
            y_top = h_m - cover_m
            # Extend into adjacent spans (e.g., L/3 or L/4)
            len_prev = spans[i-1] / 3.5 
            len_next = spans[i] / 3.5
            
            ax.plot([x_sup - len_prev, x_sup + len_next], [y_top, y_top], 
                    color='#d62728', lw=2.5, solid_capstyle='round')
            # Label (Above support)
            ax.text(x_sup, y_top - 0.05*h_m, f"{n_top}-DB16", # Assuming DB16 for now
                    ha='center', va='top', color='#d62728', fontsize=9, fontweight='bold')

    # Setting Axes
    ax.set_xlim(-total_length*0.05, total_length*1.05)
    ax.set_ylim(-h_m*0.4, h_m*1.4)
    ax.set_aspect('equal')
    ax.axis('off') # Turn off axes for a clean diagram look
    
    plt.tight_layout()
    return fig
