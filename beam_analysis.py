import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    Direct Stiffness Method for Continuous Beam Analysis (1D).
    Robust handling for List/DataFrame supports and Point/UDL loads.
    """
    n_spans = len(spans)
    n_nodes = n_spans + 1
    dof_per_node = 2  # Vertical Displacement (v), Rotation (theta)
    total_dof = n_nodes * dof_per_node

    # 1. Global Stiffness Matrix (K) & Force Vector (F)
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof)

    E = 2e6  
    I = 0.001 

    # --- ASSEMBLE MATRICES ---
    for i in range(n_spans):
        L = spans[i]
        L2 = L * L
        L3 = L2 * L
        
        # Element Stiffness Matrix (Local)
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])

        # Map to Global Indices
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_local[r, c]

        # --- CALCULATE FIXED END ACTIONS (FEM) ---
        fem_vec = np.zeros(4) 
        span_loads = [l for l in loads if l.get('span_idx') == i]
        
        for load in span_loads:
            # === UNIFORM LOAD (UDL) ===
            if load['type'] == 'U' and abs(load['w']) > 0:
                w = -load['w']
                fem_vec[0] += w * L / 2       
                fem_vec[1] += w * L2 / 12     
                fem_vec[2] += w * L / 2       
                fem_vec[3] += -w * L2 / 12    

            # === POINT LOAD (P) ===
            elif load['type'] == 'P' and abs(load['P']) > 0:
                P = -load['P'] 
                a = load['x']  
                b = L - a
                
                if 0 < a < L:
                    fem_vec[0] += P * (b**2 * (3*a + b)) / L3  
                    fem_vec[1] += P * a * (b**2) / L2          
                    fem_vec[2] += P * (a**2 * (a + 3*b)) / L3  
                    fem_vec[3] += -P * (a**2) * b / L2         

        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # 2. Apply Boundary Conditions
    free_dof = np.ones(total_dof, dtype=bool)
    vis_supports = [] 

    for i in range(n_nodes):
        # *** FIX START: SUPER ROBUST DATA EXTRACTION ***
        sup_type = "None"
        
        if i < len(supports):
            # Step 1: Get the raw item (handle List vs DataFrame)
            if isinstance(supports, list):
                raw_item = supports[i]
            elif isinstance(supports, pd.DataFrame):
                raw_item = supports.iloc[i]
            elif isinstance(supports, pd.Series):
                 # Case where a single Series is passed
                 raw_item = supports.iloc[i] if i < len(supports) else "None"
            else:
                raw_item = "None"

            # Step 2: Extract string type (handle String vs Dict/Series)
            if isinstance(raw_item, str):
                sup_type = raw_item
            elif hasattr(raw_item, 'get'): # Dict or Series
                sup_type = raw_item.get('type', 'None')
            elif hasattr(raw_item, 'type'): # Object attribute
                sup_type = raw_item.type
        # *** FIX END ***
        
        vis_supports.append({'x': sum(spans[:i]), 'type': sup_type})

        dof_y = 2*i
        dof_m = 2*i + 1
        
        if sup_type == "Pin":
            free_dof[dof_y] = False 
        elif sup_type == "Roller":
            free_dof[dof_y] = False 
        elif sup_type == "Fixed":
            free_dof[dof_y] = False 
            free_dof[dof_m] = False 

    # 3. Solve for Displacements
    K_reduced = K_global[np.ix_(free_dof, free_dof)]
    F_reduced = F_global[free_dof]
    
    try:
        D_reduced = np.linalg.solve(K_reduced, F_reduced)
    except np.linalg.LinAlgError:
        D_reduced = np.zeros_like(F_reduced) 

    D_full = np.zeros(total_dof)
    D_full[free_dof] = D_reduced

    # 4. Post-Process: Internal Forces
    x_coords = []
    shear_vals = []
    moment_vals = []
    span_ids = []

    global_x = 0.0
    
    for i in range(n_spans):
        L = spans[i]
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u = D_full[idx] 
        
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        fem_vec = np.zeros(4)
        span_loads = [l for l in loads if l.get('span_idx') == i]
        
        for load in span_loads:
            if load['type'] == 'U':
                w = -load['w']
                fem_vec[0] += w * L / 2
                fem_vec[1] += w * L2 / 12
                fem_vec[2] += w * L / 2
                fem_vec[3] += -w * L2 / 12
            elif load['type'] == 'P':
                P = -load['P']
                a = load['x']
                b = L - a
                fem_vec[0] += P * (b**2 * (3*a + b)) / L3
                fem_vec[1] += P * a * (b**2) / L2
                fem_vec[2] += P * (a**2 * (a + 3*b)) / L3
                fem_vec[3] += -P * (a**2) * b / L2

        f_end = np.dot(k_local, u) + fem_vec
        V_start = f_end[0]
        M_start = f_end[1] 
        
        num_pts = 100
        x_span = np.linspace(0, L, num_pts)
        
        for load in span_loads:
            if load['type'] == 'P':
                x_span = np.append(x_span, [load['x'] - 1e-5, load['x'] + 1e-5])
        
        x_span = np.sort(np.unique(x_span))
        x_span = x_span[(x_span >= 0) & (x_span <= L)]

        for x in x_span:
            V_x = V_start
            M_x = M_start + V_start * x
            
            for load in span_loads:
                if load['type'] == 'U':
                    w_mag = load['w']
                    V_x -= w_mag * x
                    M_x -= w_mag * x**2 / 2
                
                elif load['type'] == 'P':
                    P_mag = load['P']
                    px = load['x']
                    if x > px:
                        V_x -= P_mag
                        M_x -= P_mag * (x - px)
            
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
    
    vis_supports_df = pd.DataFrame(vis_supports)
    
    return res_df, vis_supports_df
