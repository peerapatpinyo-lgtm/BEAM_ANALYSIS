beam_analysis.py
import numpy as np
import pandas as pd

def run_beam_analysis(spans, supports, loads):
    """
    CORRECTED Direct Stiffness Method for Continuous Beams.
    
    KEY CORRECTIONS:
    1. Consistent Sign Convention:
       - Matrix Level: Up (+), CCW (+)
       - Beam Level: Up (+), Sagging (+ for Bending Moment Diagram)
    2. Proper FEM mapping to Global Force Vector.
    3. Exact Member End Force calculation using f = k*u + FEM.
    4. Continuity guaranteed by global displacement vector.
    """

    # --- 1. DATA SANITIZATION (Safety First) ---
    safe_loads = []
    if isinstance(loads, pd.DataFrame): safe_loads = loads.to_dict('records')
    elif isinstance(loads, list): safe_loads = loads
    
    safe_supports = []
    n_nodes = len(spans) + 1
    for i in range(n_nodes):
        raw = None
        if isinstance(supports, list): raw = supports[i] if i < len(supports) else None
        elif isinstance(supports, (pd.DataFrame, pd.Series)): 
            try: raw = supports.iloc[i]
            except: raw = None
            
        if isinstance(raw, str): safe_supports.append(raw)
        elif isinstance(raw, dict): safe_supports.append(raw.get('type', 'None'))
        elif hasattr(raw, 'type'): safe_supports.append(raw.type)
        else: safe_supports.append('None')

    # --- 2. SYSTEM CONSTANTS & SETUP ---
    n_spans = len(spans)
    total_dof = n_nodes * 2
    K_global = np.zeros((total_dof, total_dof))
    F_global = np.zeros(total_dof) # Global Nodal Forces (Load Vector)
    
    # Constant EI (Assumed prismatic for analysis shape)
    E = 2.0e6  
    I = 1.0e-3 

    # --- 3. ASSEMBLY (Stiffness K & Load Vector F) ---
    for i in range(n_spans):
        L = spans[i]
        L2 = L*L; L3 = L2*L
        
        # 3.1 Local Stiffness Matrix (Bernoulli-Euler)
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])

        # 3.2 Map to Global K
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1] # [y1, m1, y2, m2]
        for r in range(4):
            for c in range(4):
                K_global[idx[r], idx[c]] += k_local[r, c]

        # 3.3 Fixed End Actions (FEM)
        # Convention: Reaction Forces due to Load
        # Downward Load -> Left: Up(+), CCW(+); Right: Up(+), CW(-)
        fem_vec = np.zeros(4) 
        
        # Get loads for this span
        current_span_loads = [l for l in safe_loads if int(l.get('span_idx', -999)) == i]
        
        for load in current_span_loads:
            l_type = load.get('type')
            try: 
                w = abs(float(load.get('w', 0.0))) # Always positive magnitude for formulas
                P = abs(float(load.get('P', 0.0)))
                a = float(load.get('x', 0.0))
            except: continue
            
            # Formulate FEM (Reactions at Fixed Ends)
            if l_type == 'U' and w > 0:
                # Load is DOWN (-y direction)
                fem_vec[0] += w * L / 2        # Fy_L (+)
                fem_vec[1] += w * L2 / 12      # M_L  (+) (CCW)
                fem_vec[2] += w * L / 2        # Fy_R (+)
                fem_vec[3] += -w * L2 / 12     # M_R  (-) (CW)

            elif l_type == 'P' and P > 0:
                # Load is DOWN
                b = L - a
                fem_vec[0] += P * (b**2 * (3*a + b)) / L3  # Fy_L
                fem_vec[1] += P * a * (b**2) / L2          # M_L (CCW)
                fem_vec[2] += P * (a**2 * (a + 3*b)) / L3  # Fy_R
                fem_vec[3] += -P * (a**2) * b / L2         # M_R (CW)

        # 3.4 Assemble Force Vector (F_nodes = F_ext - FEM)
        # Since we have no external nodal loads/moments, F_ext is 0.
        # So F_global = -FEM
        for r in range(4):
            F_global[idx[r]] -= fem_vec[r]

    # --- 4. APPLY BOUNDARY CONDITIONS & SOLVE ---
    free_dof = np.ones(total_dof, dtype=bool)
    vis_supports_data = []

    for i in range(n_nodes):
        stype = safe_supports[i]
        vis_supports_data.append({'x': sum(spans[:i]), 'type': stype})
        dof_y = 2*i
        dof_m = 2*i + 1
        
        if stype in ["Pin", "Roller"]:
            free_dof[dof_y] = False # Fix Y
        elif stype == "Fixed":
            free_dof[dof_y] = False # Fix Y
            free_dof[dof_m] = False # Fix Rotation

    # Partition and Solve
    K_free = K_global[np.ix_(free_dof, free_dof)]
    F_free = F_global[free_dof]
    
    D_free = np.zeros_like(F_free)
    try:
        if len(F_free) > 0:
            D_free = np.linalg.solve(K_free, F_free)
    except:
        pass # Unstable

    # Full Displacement Vector
    D_total = np.zeros(total_dof)
    D_total[free_dof] = D_free

    # --- 5. POST-PROCESSING (Internal Forces) ---
    # We compute Internal Forces by walking the beam, starting from
    # the CALCULATED Member End Forces.
    
    x_coords, shear_vals, moment_vals, span_ids = [], [], [], []
    global_x_start = 0.0

    for i in range(n_spans):
        L = spans[i]
        # Get Local Displacements
        idx = [2*i, 2*i+1, 2*(i+1), 2*(i+1)+1]
        u_local = D_total[idx]
        
        # Re-construct stiffness and FEM for this element
        L2 = L*L; L3 = L2*L
        k_local = (E * I / L3) * np.array([
            [12,      6*L,    -12,     6*L],
            [6*L,     4*L2,   -6*L,    2*L2],
            [-12,    -6*L,     12,    -6*L],
            [6*L,     2*L2,   -6*L,    4*L2]
        ])
        
        # Re-calc FEM Vector for this span
        fem_local = np.zeros(4)
        current_span_loads = [l for l in safe_loads if int(l.get('span_idx', -999)) == i]
        for load in current_span_loads:
            l_type = load.get('type')
            try: w=abs(float(load.get('w',0))); P=abs(float(load.get('P',0))); a=float(load.get('x',0))
            except: continue
            if l_type == 'U':
                fem_local[0] += w*L/2; fem_local[1] += w*L2/12
                fem_local[2] += w*L/2; fem_local[3] += -w*L2/12
            elif l_type == 'P':
                b = L - a
                fem_local[0] += P*(b**2*(3*a+b))/L3; fem_local[1] += P*a*(b**2)/L2
                fem_local[2] += P*(a**2*(a+3*b))/L3; fem_local[3] += -P*(a**2)*b/L2

        # 5.1 Calculate MEMBER END FORCES (Matrix Convention)
        # f_member = k * u + FEM
        f_member = np.dot(k_local, u_local) + fem_local
        
        # Extract Start Forces (Left Node of Span)
        # f_member[0] = Shear Force (Up is +)
        # f_member[1] = Moment (CCW is +)
        
        # CONVERT TO BEAM SIGN CONVENTION FOR PLOTTING
        # Shear: Upward force on left face is Positive Shear.
        # Moment: CCW Moment on left face causes HOGGING (Sad face) -> Negative Moment.
        
        V_start = f_member[0]       
        M_start = -f_member[1]      # !!! CRITICAL FIX: Matrix CCW (+) = Beam Hogging (-)
        
        # 5.2 Walk along the span to plot V(x) and M(x)
        # Use simple statics starting from the calculated End Forces
        
        # Define evaluation points
        eval_x = np.linspace(0, L, 100)
        for load in current_span_loads:
            if load.get('type') == 'P':
                lx = float(load.get('x', 0.0))
                eval_x = np.append(eval_x, [lx - 1e-5, lx + 1e-5])
        eval_x = np.sort(np.unique(eval_x))
        eval_x = eval_x[(eval_x >= 0) & (eval_x <= L)]
        
        for x in eval_x:
            # Shear V(x) = V_start - Sum(Downward Loads)
            # Moment M(x) = M_start + V_start*x - Sum(Moment from Loads) + (Sagging +)
            
            V_x = V_start
            M_x = M_start + V_start * x
            
            for load in current_span_loads:
                l_type = load.get('type')
                try: w=abs(float(load.get('w',0))); P=abs(float(load.get('P',0))); px=float(load.get('x',0))
                except: continue
                
                if l_type == 'U':
                    # Load applies from 0 to L. At position x, total load is w*x
                    if x > 0:
                        load_mag = w * x
                        V_x -= load_mag
                        # Moment arm of this block is x/2
                        M_x -= load_mag * (x/2)
                        
                elif l_type == 'P':
                    if x > px:
                        V_x -= P
                        M_x -= P * (x - px)
            
            x_coords.append(global_x_start + x)
            shear_vals.append(V_x)
            moment_vals.append(M_x)
            span_ids.append(i)
            
        global_x_start += L

    res_df = pd.DataFrame({
        'x': x_coords,
        'shear': shear_vals,
        'moment': moment_vals,
        'span_id': span_ids
    })
    
    return res_df, pd.DataFrame(vis_supports_data)
