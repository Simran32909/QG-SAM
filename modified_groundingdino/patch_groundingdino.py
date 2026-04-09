import os
import sys

# Path to the GroundingDINO ms_deform_attn.py file
ms_deform_attn_path = "/ssd_scratch/jyothi.swaroopa/Simran/qgsam/Grounded-Segment-Anything/GroundingDINO/groundingdino/models/GroundingDINO/ms_deform_attn.py"

# Read our Python implementation
with open("/ssd_scratch/jyothi.swaroopa/Simran/qgsam/modified_groundingdino/ms_deform_attn_pytorch.py", "r") as f:
    python_impl = f.read()

# Read the original file
with open(ms_deform_attn_path, "r") as f:
    original_content = f.read()

# Find the MultiScaleDeformableAttnFunction class
start_idx = original_content.find("class MultiScaleDeformableAttnFunction")
if start_idx == -1:
    print("Could not find MultiScaleDeformableAttnFunction class in the original file")
    sys.exit(1)

# Find the forward method
forward_idx = original_content.find("def forward", start_idx)
if forward_idx == -1:
    print("Could not find forward method in MultiScaleDeformableAttnFunction")
    sys.exit(1)

# Find the next method or class after forward
next_method_idx = original_content.find("def ", forward_idx + 1)
if next_method_idx == -1:
    next_method_idx = len(original_content)

# Extract the forward method
forward_method = original_content[forward_idx:next_method_idx]

# Replace the C++ extension call with our Python implementation
if "_C.ms_deform_attn_forward" in forward_method:
    # Create the new forward method
    new_forward_method = """    @staticmethod
    def forward(ctx, value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights, im2col_step):
        ctx.im2col_step = im2col_step
        output = ms_deform_attn_core_pytorch(
            value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        ctx.save_for_backward(value, value_spatial_shapes, value_level_start_index, sampling_locations, attention_weights)
        return output"""

    # Replace the forward method in the original content
    new_content = original_content[:forward_idx] + new_forward_method + original_content[next_method_idx:]
    
    # Add our Python implementation at the top of the file
    import_end = new_content.find("import", new_content.find("import") + 1)
    if import_end == -1:
        import_end = new_content.find("class")
    
    new_content = new_content[:import_end] + "\n" + python_impl.split("class")[0] + "\n" + new_content[import_end:]
    
    # Write the modified file
    with open(ms_deform_attn_path, "w") as f:
        f.write(new_content)
    
    print(f"Successfully patched {ms_deform_attn_path} with Python-only implementation")
else:
    print("Could not find _C.ms_deform_attn_forward in the forward method")
    sys.exit(1)
