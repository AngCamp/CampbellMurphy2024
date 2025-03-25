
### 📁 Session ID Files
This folder contains .npz files with precomputed session IDs for experiments with CA1 recordings across multiple datasets. Each file contains a single array under the key 'data', which can be easily loaded using NumPy.

### 🗃️ Files
all_ibl_sessions_with_ca1.npz
CA1 session IDs from the IBL (International Brain Laboratory) dataset.

allen_visbehave_ca1_session_ids.npz
CA1 session IDs from the Allen Institute's Visual Behavior Neuropixels dataset.

allen_viscoding_ca1_session_ids.npz
CA1 session IDs from the Allen Institute's Visual Coding Neuropixels dataset.

### 📥 Loading Example
```python
import numpy as np

# Load any of the .npz files
data = np.load("allen_visbehave_ca1_session_ids.npz")["data"]
print(data)
```

## 📌 Notes
All files contain a single key: 'data', which maps to a NumPy array of session IDs.

These files are intended for quick access to relevant session subsets without recomputing filters across the full dataset.

Useful for downstream processing, CA1-specific analyses, or fast lookups.

### Scripts used to make the lists

Here are the scripts used to make the lists using the api's.

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

# Get Visual Behaviour-specific configuration
dataset_config = full_config["aabi_visual_behaviour"]
sdk_cache_dir = dataset_config["sdk_cache_dir"]
manifest_path = os.path.join(sdk_cache_dir, "manifest.json")

# Initialize cache
print("Initializing EcephysProjectCache for visual behaviour...")
cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

# Get session table
print("Getting session table...")
session_table = cache.get_session_table()

# Get all session IDs
session_ids = session_table.index.values
print(f"Found {len(session_ids)} sessions in visual behaviour dataset")

# Save all session IDs
output_file = os.path.join(sdk_cache_dir, "visual_behaviour_session_ids.npz")
np.savez(output_file, session_ids=session_ids)
print(f"Saved session IDs to {output_file}")

# Optional: Filter for sessions that have CA1 data
sessions_with_ca1 = []
print("Filtering for sessions with CA1 data...")
for session_id in session_ids:
    if "CA1" in session_table.loc[session_id, "ecephys_structure_acronyms"]:
        sessions_with_ca1.append(session_id)

print(f"Found {len(sessions_with_ca1)} sessions with CA1 data")

# Save CA1 sessions to separate NPZ file
np.savez("allen_visbehaviour_ca1_session_ids.npz", data=np.array(sessions_with_ca1))
```