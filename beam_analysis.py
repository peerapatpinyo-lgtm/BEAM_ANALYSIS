import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Direct Stiffness Method with EXACT Static Equilibrium Post-Processing.
    Corrects SFD/BMD inconsistency by using strict equilibrium equations 
    from node reactions.
    """

    # ==============================================================================
    # 1. SANITIZE INPUTS (Keep this to prevent string errors)
    # ==============================================================================
    safe_loads = []
    if isinstance(loads, pd.DataFrame):
        safe_loads = loads.to_dict('records') 
    elif isinstance(loads, list):
        safe_loads = loads
    
    safe_supports = []
    n_nodes = len(spans) + 1
    
    for i in range(n_nodes):
        raw_val = None
        if isinstance(supports, list):
            raw_val = supports[i] if i < len(supports) else None
        elif isinstance(supports, (pd.DataFrame, pd.Series)):
            try: raw_val = supports.iloc[i]
            except: raw_val = None
        
        sup_str = "None"
        if isinstance(raw_val, str): sup_str = raw_val
        elif isinstance(raw_val, dict): sup_str = raw_val.get('type', 'None')
        elif hasattr(raw_val, 'get'): sup_str = raw_val.get('type', 'None')
        elif hasattr(raw_val, 'type'): sup_str = raw_val.type
        safe_supports.append(sup_str)

    # ==============================================================================
    # 2. GLOBAL STIFFNESS & LOAD ASSEMBLY
    # ==============================================================================
    n_spans = len(spans)
    total_dof = n_nodes * 2
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)
    E = 2e6  
    I = 0.001 

    for i in range(n_spans):
        L = spans[i]
        L2 = L*L; L3 = L2*L
        
        # Element Stiffness (Standard Euler-Bernoulli)
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])

        # Assemble Global K
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_local[r, c]

        # FEM Calculation (Fixed End Actions)
        # Sign Convention for Matrix: Up (+), CCW (+)
        fem_vec = np.zeros(4)
        
        # Filter loads for this span
        span_loads = []
        for load in safe_loads:
            try: s_idx = int(load.get('span_idx', -999))
            except: continue
            if s_idx == i: span_loads.append(load)

        for load in span_loads:
            l_type = load.get('type', 'None')
            try: w = float(load.get('w', 0.0))  # Magnitude
            except: w = 0.0
            try: P = float(load.get('P', 0.0))  # Magnitude
            except: P = 0.0
            try: a = float(load.get('x', 0.0))
            except: a = 0.0

            # NOTE: We assume input loads are Gravity Loads (Acting Down)
            # FEM Reactions for Downward Load:
            # Left: Force Up (+), Moment CCW (+)
            # Right: Force Up (+), Moment CW (-)
            
            if l_type == 'U' and w != 0:
                # w is load intensity. FEM: wL^2/12
                fem_vec[0] += w * L / 2        # Fy1 (Up)
                fem_vec[1] += w * L2 / 12      # M1 (CCW)
                fem_vec[2] += w * L / 2        # Fy2 (Up)
                fem_vec[3] += -w * L2 / 12     # M2 (CW -> Negative)

            elif l_type == 'P' and P != 0:
                b = L - a
                fem_vec[0] += P * (b**2 * (3*a + b)) / L3 # Fy1
                fem_vec[1] += P * a * (b**2) / L2         # M1 (CCW)
                fem_vec[2] += P * (a**2 * (a + 3*b)) / L3 # Fy2
                fem_vec[3] += -P * (a**2) * b / L2        # M2 (CW)

        # Subtract FEM from Global Force Vector (F_nodes = F_ext - FEM)
        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # ==============================================================================
    # 3. SOLVE DISPLACEMENTS
    # ==============================================================================
    free_dof = np.ones(total_dof, dtype=bool)
    vis_supports_data = []

    for i in range(n_nodes):
        stype = safe_supports[i]
        vis_supports_data.append({'x': sum(spans[:i]), 'type': stype})
        dof_y, dof_m = 2*i, 2*i+1
        
        if stype == "Pin" or stype == "Roller":
            free_dof[dof_y] = False 
        elif stype == "Fixed":
            free_dof[dof_y] = False 
            free_dof[dof_m] = False 

    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    try:
        D_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        D_reduced = np.zeros_like(F_reduced) # Unstable structure fallback

    D_full = np.zeros(total_dof)
    D_full[free_dof] = D_reduced

    # ==============================================================================
    # 4. POST-PROCESSING (STATIC EQUILIBRIUM METHOD)
    # ==============================================================================
    x_coords, shear_vals, moment_vals, span_ids = [], [], [], []
    global_x = 0.0
    
    for i in range(n_spans):
        L = spans[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u_span = D_full[idx] # Local displacements [y1, th1, y2, th2]
        
        # 4.1 Re-calculate Member End Forces (Matrix Sign Convention)
        # F_member = k_local * u + FEM
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        # Re-compute FEM for this span to add back
        fem_span = np.zeros(4)
        span_loads = []
        for load in safe_loads:
            try: s_idx = int(load.get('span_idx', -999))
            except: continue
            if s_idx == i: span_loads.append(load)
            
        for load in span_loads:
            l_type = load.get('type')
            try: w = float(load.get('w', 0.0)); P = float(load.get('P', 0.0)); a = float(load.get('x', 0.0))
            except: w=0; P=0; a=0
            
            if l_type == 'U':
                fem_span[0] += w*L/2; fem_span[1] += w*L2/12
                fem_span[2] += w*L/2; fem_span[3] += -w*L2/12
            elif l_type == 'P':
                b = L - a
                fem_span[0] += P*(b**2*(3*a+b))/L3; fem_span[1] += P*a*(b**2)/L2
                fem_span[2] += P*(a**2*(a+3*b))/L3; fem_span[3] += -P*(a**2)*b/L2

        # Final Member End Forces (At Nodes)
        f_end = np.dot(k_local, u_span) + fem_span
        
        # Extract Start Forces
        # Fy_start: Up is Positive (Matrix) -> Shear Up is Positive (Beam)
        # M_start: CCW is Positive (Matrix) -> Hogging (Negative Moment) for Beam Left End
        V_start_node = f_end[0] 
        M_start_node = f_end[1] # CCW +

        # 4.2 Calculate Internal Forces at x (0 to L) using Equilibrium
        # V(x) = V_start_node - Sum(Downward Loads)
        # M(x) = -M_start_node + V_start_node*x - Sum(Moment from Downward Loads)
        
        # Create eval points (add points at loads to catch discontinuities)
        eval_x = np.linspace(0, L, 100)
        for load in span_loads:
             if load.get('type') == 'P':
                 lx = float(load.get('x', 0.0))
                 eval_x = np.append(eval_x, [lx - 1e-6, lx + 1e-6])
        eval_x = np.sort(np.unique(eval_x))
        eval_x = eval_x[(eval_x >= 0) & (eval_x <= L)]

        for x in eval_x:
            # Shear Calculation
            V_x = V_start_node
            
            # Moment Calculation (Sagging +)
            # Start with Moment from Node Reaction (CCW Moment at support = Hogging on Beam)
            M_x = -M_start_node + (V_start_node * x)
            
            # Subtract Load Effects
            for load in span_loads:
                l_type = load.get('type')
                try: w = float(load.get('w', 0.0)); P = float(load.get('P', 0.0)); px = float(load.get('x', 0.0))
                except: w=0; P=0; px=0

                if l_type == 'U':
                    # UDL starts at 0, goes to L (assuming full span for now based on input UI)
                    # Load total = w * x
                    V_x -= w * x
                    # Moment arm is x/2
                    M_x -= (w * x) * (x / 2)
                    
                elif l_type == 'P':
                    if x > px:
                        V_x -= P
                        M_x -= P * (x - px)
            
            x_coords.append(global_x + x)
            shear_vals.append(V_x)
            moment_vals.append(M_x)
            span_ids.append(i)
            
        global_x += L

    res_df = pd.DataFrame({
        'x': x_coords,
        'shear': shear_vals,
        'moment': moment_vals,
        'span_id': span_ids
    })
    
    return res_df, pd.DataFrame(vis_supports_data)
