# rc_load_processor.py
import pandas as pd
from rc_utils import normalize_section_units

def prepare_load_dataframe(raw_loads_df, n_spans, spans, params, f_dl, f_ll):
    """
    Helper function to prepare load dataframe for solver.
    Scales loads by Load Factors (f_dl, f_ll).
    Also handles Unit Normalization for Self-Weight calculation.
    """
    # Normalize inputs for self-weight calculation
    b_mm, h_mm = normalize_section_units(params['b'], params['h'])
    b_m = b_mm / 1000.0
    h_m = h_mm / 1000.0
    
    # 1. Self-weight (Calculated from dimensions in Meters)
    # Density approx 24 kN/m3
    w_sw_base_kN = b_m * h_m * 24.0      
    w_sw_factored_kN = w_sw_base_kN * f_dl
    
    # Initialize dictionary for Total UDL per span
    span_total_udl_N = {i: w_sw_factored_kN * 1000.0 for i in range(n_spans)} 
    combined_loads_list = []
    
    # 2. User Defined Loads from DataFrame
    if not raw_loads_df.empty:
        for _, row in raw_loads_df.iterrows():
            try:
                s_idx = int(row['span_index'])
                if s_idx >= n_spans: continue 
                
                l_type = row['type']
                # Determine factor based on case
                u_factor = f_dl if row['case'] == 'DL' else f_ll
                
                mag_base_kN = float(row['mag']) 
                mag_factored_N = mag_base_kN * u_factor * 1000.0 
                
                dist = float(row['dist'])
                d_start = float(row['d_start'])
                
                if l_type == 'P':
                    combined_loads_list.append({
                        'span_index': s_idx, 'type': 'P', 'mag': mag_factored_N, 
                        'd_start': d_start, 'dist': 0.0
                    })
                elif l_type == 'U':
                    # If Full Span UDL, add to the base accumulator
                    if d_start <= 0.01 and dist >= (spans[s_idx] - 0.01):
                        span_total_udl_N[s_idx] += mag_factored_N
                    else:
                        combined_loads_list.append({
                            'span_index': s_idx, 'type': 'U', 'mag': mag_factored_N, 
                            'd_start': d_start, 'dist': dist
                        })
            except Exception: continue
    
    # Add Self-weight + Full Span UDLs combined
    for i in range(n_spans):
        if span_total_udl_N[i] > 0:
            combined_loads_list.append({
                'span_index': i, 'type': 'U', 'mag': span_total_udl_N[i], 
                'd_start': 0.0, 'dist': spans[i]
            })
            
    return pd.DataFrame(combined_loads_list)
