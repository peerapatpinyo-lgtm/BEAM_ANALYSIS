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
import ifcopenshell.api.aggregate
import ifcopenshell.api.pset
import uuid
import numpy as np

def create_guid():
    return ifcopenshell.guid.compress(uuid.uuid1().hex)

def generate_ifc_model(project_name, spans, params, design_results):
    """
    Generate IFC4 file.
    Final Fix: All relationship functions now use 'products=[...]' list argument.
    """
    model = ifcopenshell.file(schema="IFC4")
    
    # Setup
    project = model.create_entity("IfcProject", GlobalId=create_guid(), Name=project_name)
    ifcopenshell.api.run("unit.assign_unit", model)
    
    context = ifcopenshell.api.run("context.add_context", model, context_type="Model")
    body = ifcopenshell.api.run("context.add_context", model, 
                                context_type="Model", context_identifier="Body", 
                                target_view="MODEL_VIEW", parent=context)
    
    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name="Project Site")
    building = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuilding", name="Main Building")
    storey = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1")
    
    # FIX 1: products=[...]
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=project, products=[site])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=site, products=[building])
    ifcopenshell.api.run("aggregate.assign_object", model, relating_object=building, products=[storey])

    concrete_name = f"Concrete fc'{params['fc']} MPa"
    concrete = ifcopenshell.api.run("material.add_material", model, name=concrete_name)
    
    current_x = 0.0
    b_mm = float(params['b'])
    h_mm = float(params['h'])
    
    for i, span_len in enumerate(spans):
        beam_name = f"B{i+1}"
        length_mm = float(span_len) * 1000.0
        
        beam = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcBeam", name=beam_name)
        
        # Geometry
        profile = model.create_entity("IfcRectangleProfileDef", ProfileType="AREA", XDim=b_mm, YDim=h_mm)
        representation = ifcopenshell.api.run("geometry.add_profile_representation", model, 
                                              context=body, profile=profile, depth=length_mm)
        # Note: assign_representation is direct attribute, usually accepts 'product' or 'products'. keeping 'product' as it passed before.
        ifcopenshell.api.run("geometry.assign_representation", model, product=beam, representation=representation)
        
        # Placement
        matrix = np.eye(4)
        matrix[0][3] = current_x * 1000.0
        ifcopenshell.api.run("geometry.edit_object_placement", model, product=beam, matrix=matrix)
        
        # Material (FIX 2: products=[...])
        ifcopenshell.api.run("material.assign_material", model, products=[beam], material=concrete)
        
        # Properties
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
        
        # FIX 3 (The Error You Saw): products=[...] instead of product=...
        ifcopenshell.api.run("pset.add_pset", model, products=[beam], name="Pset_StructuralDesign", properties=props)
        
        # Aggregate (FIX 4: products=[...])
        ifcopenshell.api.run("aggregate.assign_object", model, relating_object=storey, products=[beam])
        
        current_x += span_len

    return model.to_string()
