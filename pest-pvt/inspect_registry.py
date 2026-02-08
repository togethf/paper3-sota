
import os
import warnings
from mmengine.registry import MODELS
from mmengine.config import Config

# Trigger imports
try:
    import mmdet.models
    print("Imported mmdet.models")
except ImportError as e:
    print(f"Failed to import mmdet.models: {e}")

try:
    import mmpretrain.models
    print("Imported mmpretrain.models")
except ImportError as e:
    print(f"Failed to import mmpretrain.models: {e}")

print("\n--- Searching for PVT in Registry ---")
found = []
for name in MODELS.module_dict.keys():
    if 'PVT' in name or 'PyramidVisionTransformer' in name:
        found.append(name)

print(f"Found {len(found)} matching modules:")
for name in sorted(found):
    module = MODELS.module_dict[name]
    print(f"Name: '{name}' -> {module}")

print("\n--- Testing Instantiation ---")
try:
    # Try instantiation with arch (mmpretrain style)
    cfg = dict(type='mmpretrain.PVTv2', arch='b2')
    print(f"Trying build matching: {cfg}")
    model = MODELS.build(cfg)
    print("Success building mmpretrain.PVTv2!")
except Exception as e:
    print(f"Failed building mmpretrain.PVTv2: {e}")

