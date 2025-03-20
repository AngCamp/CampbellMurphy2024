
This code allowed us to make the lists of sesssions.

Before running, activate your environment:
   `conda activate allensdk_env`

Then run the script:
   `python fetch_session_ids.py`

```python
import os
import numpy as np
import yaml
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

# Load configuration
config_path = os.environ.get('CONFIG_PATH', 'united_detector_config.yaml')
with open(config_path, "r") as f:
    config_content = f.read()
    full_config = yaml.safe_load(config_content)

# Get Visual Coding specific configuration
dataset_config = full_config["abi_visual_coding"]
sdk_cache_dir = dataset_config["sdk_cache_dir"]
manifest_path = os.path.join(sdk_cache_dir, "manifest.json")

# Initialize cache
print("Initializing EcephysProjectCache...")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# Get session table
print("Getting session table...")
session_table = cache.get_session_table()

# Get all session IDs
session_ids = session_table.index.values
print(f"Found {len(session_ids)} sessions")

# Save to NPZ file
output_file = os.path.join(sdk_cache_dir, "visual_coding_session_ids.npz")
np.savez(output_file, session_ids=session_ids)
print(f"Saved session IDs to {output_file}")

# Optional: Filter for sessions that have CA1 data
sessions_with_ca1 = []
print("Filtering for sessions with CA1 data...")
for session_id in session_ids:
    # Check if session has 'CA1' in structure_acronyms
    if "CA1" in session_table.loc[session_id, "ecephys_structure_acronyms"]:
        sessions_with_ca1.append(session_id)

print(f"Found {len(sessions_with_ca1)} sessions with CA1 data")

# Save CA1 sessions to separate NPZ file
np.savez("allen_viscoding_ca1_session_ids.npz", data=np.array(sessions_with_ca1))
```