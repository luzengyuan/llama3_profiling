import torch
import torch.nn as nn
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

import os

# Load LLaMA 3 Model
model_name = "meta-llama/Llama-3.2-1B"  # Adjust model name as needed
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

print(model.model.embed_tokens)


tokenizer = AutoTokenizer.from_pretrained(model_name)

# Storage for profiling data
profiling_data = []

# Function to estimate FLOPs for a layer
def compute_flops(layer, inputs, output):
    try:
        input_shapes = [tuple(inp.shape) for inp in inputs]  # Extract input shapes
        output_shape = tuple(output.shape)

    #    if isinstance(layer, nn.Linear):  # Linear layer FLOPs = 2 * input_features * output_features
        return sum(2 * shape[-1] * output_shape[-1] for shape in input_shapes)
        
        return 0  # Default for unknown layers
    except:
        return 0  # Handle unexpected errors

# Hook function to capture input, output, parameter shapes, and FLOPs
def hook_fn(module, inputs, output, name):
    try:
        input_shapes = [tuple(inp.shape) for inp in inputs]  # Handle multiple inputs
    except:
        input_shapes = ["Error"]  # If input shape cannot be determined
    
    try:
        output_shape = tuple(output.shape)
    except:
        output_shape = 0

    # try:
    #     param_shapes = [tuple(p.shape) for p in module.parameters()]  # Extract parameter shapes
    # except:
    #     param_shapes = "Error"

    try:
        param_shapes = tuple(output.shape) # Extract parameter shapes
    except:
        param_shapes = 0

    try:
        flops = compute_flops(module, inputs, output)
    except:
        flops = 0  # Default in case of error

    # Append data to profiling list
    profiling_data.append({
        "Layer Name": name,
        "Layer Type": module.__class__.__name__,
        "Input Shapes": input_shapes,
        "Parameter Shapes": param_shapes,  # Track parameter shapes
        "Output Shape": output_shape,
        "FLOPs": flops
    })

# Register hooks for each layer
hooks = []
for name, layer in model.named_modules():
    #if isinstance(layer, (nn.Linear, nn.Embedding, nn.Conv2d)):  # Add more layer types if needed
        hooks.append(layer.register_forward_hook(lambda m, i, o, n=name: hook_fn(m, i, o, n)))

# Dummy input for profiling
input_text = "The quick brown fox jumps over the lazy dog."
inputs = tokenizer(input_text, return_tensors="pt")
input_ids = inputs.input_ids.to(model.device)
attention_mask = inputs.attention_mask.to(model.device)  # Second input

# Run a forward pass to trigger hooks
with torch.no_grad():
    _ = model(input_ids, attention_mask=attention_mask)

# Remove hooks
for hook in hooks:
    hook.remove()

# Extract filename without extension
filename = os.path.splitext(os.path.basename(__file__))[0]

# set the output file name
output_filename = filename + '.csv'

# Save profiling results to CSV
df = pd.DataFrame(profiling_data)
df.to_csv(output_filename, index=False)

print("Profiling data saved to :", output_filename)
print("Done!")

