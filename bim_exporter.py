# bim_exporter.py
import ifcopenshell
import ifcopenshell.guid
import ifcopenshell.api
import ifcopenshell.api.root
import ifcopenshell.api.unit
import ifcopenshell.api.context
import ifcopenshell.api.project
import ifcopenshell.api.material
import ifcopenshell.api.geometry
import ifcopenshell.api.aggregate # Import ตัวนี้เพิ่มเพื่อความชัวร์
import uuid

def create_guid():
    return ifcopenshell.guid.compress(uuid.uuid1().hex)

def generate_ifc_model(project_name, spans, params, design_results):
    """
    Generate IFC4 file for the continuous beam with design data properties.
    Fixed for ifcopenshell v0.7.0+ API compliance.
    """
    # 1. Initialize IFC File
    model = ifcopenshell.file(schema="IFC4")
    
    # 2. Project Setup
    project = model.create_entity("IfcProject", GlobalId=create_guid(), Name=project_name)
    ifcopenshell.api.run("unit.assign_unit", model)
    
    # 3. Context & Hierarchy
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model, 
                                context_type="Model", context_identifier="Body", 
                                target_view="MODEL_VIEW", parent=context)
    
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Project Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Main Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1")
    
    # --- FIX: Use 'products=[...]' (list) instead of 'product=...' ---
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])

    # 4. Define Material
    concrete_name = f"Concrete fc'{params['fc']} MPa"
    concrete = ifcopenshell.api.run("material.add_material", model, name=concrete_name)
    
    # 5. Create Beams Loop
    current_x = 0.0
    b_mm = float(params['b'])
    h_mm = float(params['h'])
    
    for i, span_len in enumerate(spans):
        beam_name = f"B{i+1}"
        length_mm = float(span_len) * 1000.0
        
        # Create Beam Entity
        beam = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBeam", name=beam_name)
        
        # --- 5.1 Geometry (Extrusion) ---
        # Create Profile (Rectangle)
        profile = model.create_entity("IfcRectangleProfileDef", 
                                      ProfileType="AREA", 
                                      XDim=b_mm, YDim=h_mm)
        
        # Create 3D Representation (Extrusion)
        # Note: add_profile_representation creates the shape
        representation = ifcopenshell.api.run("geometry.add_profile_representation", model, 
                                              context=body, profile=profile, depth=length_mm)
        
        ifcopenshell.api.run("geometry.assign_representation", model, product=beam, representation=representation)
        
        # --- 5.2 Placement (Matrix) ---
        # Shift X to current position. 
        # Note: In standard extrusion, Z is length. We need to rotate or just place it.
        # For simplicity in this viewer: We place them along X axis. 
        # (Standard IFC beams usually extrude along local X, but let's stick to simple placement)
        
        # Matrix logic: [x, y, z]
        ifcopenshell.api.run("geometry.edit_object_placement", model, product=beam, matrix=[current_x * 1000.0, 0.0, 0.0])
        
        # --- 5.3 Material ---
        ifcopenshell.api.run("material.assign_material", model, product=beam, material=concrete)
        
        # --- 5.4 Properties (Embedded Data) ---
        res = design_results[i]
        props = {
            "Span Length": float(span_len),
            "Concrete Grade": float(params['fc']),
            "Steel Grade": float(params['fy']),
            "Top Rebar Area (Req)": float(res['neg']['area']),
            "Bot Rebar Area (Req)": float(res['pos']['area']),
            "Stirrup Spacing": float(res['shear']['s']),
            "Stirrup DB": float(res['shear']['db']),
            "Design Status": "PASS" if (res['pos']['status'] and res['neg']['status']) else "FAIL"
        }
        
        ifcopenshell.api.run("pset.add_pset", model, product=beam, name="Pset_StructuralDesign", properties=props)
        
        # Assign to Storey (Fix: Use products=[beam])
        ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=[beam])
        
        current_x += span_len

    # 6. Serialize
    return model.to_string()
