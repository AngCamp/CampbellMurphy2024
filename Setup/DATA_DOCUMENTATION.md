# Sharp Wave Ripple Dataset: Complete Column Documentation

This document provides comprehensive documentation for all data files in the Sharp Wave Ripple dataset, including column descriptions, units, data types, and computational notes.

## File Overview

The dataset contains several types of files:
- **CSV files**: Event data with detailed metrics
- **JSON files**: Metadata and processing parameters
- **Compressed files**: All data files are gzip compressed (.gz)

## Putative SWR Events (*_putative_swr_events.csv.gz)

### Event Timing Information
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `start_time` | The start time of the SWR event | seconds | float64 | |
| `end_time` | The end time of the SWR event | seconds | float64 | |
| `duration` | The duration of the SWR event | seconds | float64 | |
| `power_peak_time` | Time of peak power within the SWR event | seconds | float64 | |

### Power Z-scores
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `power_max_zscore` | Maximum z-score of power during the event | z-score | float64 | |
| `power_median_zscore` | Median z-score of power during the event | z-score | float64 | |
| `power_mean_zscore` | Mean z-score of power during the event | z-score | float64 | |
| `power_min_zscore` | Minimum z-score of power during the event | z-score | float64 | |
| `power_90th_percentile` | 90th percentile of power z-score during the event | z-score | float64 | |

### Sharp Wave Metrics
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `sw_exceeds_threshold` | Boolean indicating if sharp wave exceeds detection threshold | - | bool | |
| `sw_peak_power` | Peak power of the sharp wave component | μV² | float64 | |
| `sw_peak_time` | Time of peak sharp wave power | seconds | float64 | |

### Sharp Wave-Ripple Coupling
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `sw_ripple_plv` | Phase-locking value between sharp wave and ripple components | 0-1 | float64 | |
| `sw_ripple_mi` | Modulation index between sharp wave and ripple | - | float64 | |
| `sw_ripple_clcorr` | Cross-correlation between sharp wave and ripple | correlation coefficient | float64 | |

### Envelope Metrics
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `envelope_peak_time` | Time of peak envelope amplitude | seconds | float64 | |
| `envelope_max_thresh` | Maximum envelope threshold value | μV | float64 | |
| `envelope_mean_zscore` | Mean z-score of envelope amplitude | z-score | float64 | |
| `envelope_median_zscore` | Median z-score of envelope amplitude | z-score | float64 | |
| `envelope_max_zscore` | Maximum z-score of envelope amplitude | z-score | float64 | |
| `envelope_min_zscore` | Minimum z-score of envelope amplitude | z-score | float64 | |
| `envelope_area` | Area under the envelope curve | μV·s | float64 | |
| `envelope_total_energy` | Total energy of the envelope signal | μV²·s | float64 | |
| `envelope_90th_percentile` | 90th percentile of envelope amplitude | μV | float64 | |

### Gamma Overlap
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `overlaps_with_gamma` | Boolean indicating if SWR overlaps with gamma oscillations | - | bool | |
| `gamma_overlap_percent` | Percentage of SWR duration that overlaps with gamma | percentage | float64 | |

### Movement Overlap
| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `overlaps_with_movement` | Boolean indicating if SWR overlaps with movement periods | - | bool | |
| `movement_overlap_percent` | Percentage of SWR duration that overlaps with movement | percentage | float64 | |

## Movement Artifacts (*_movement_artifacts.csv.gz)

| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `Unnamed: 0` | Row index from original dataset | - | int64 | |
| `start_time` | The start time of the movement artifact | seconds | float64 | |
| `end_time` | The end time of the movement artifact | seconds | float64 | |
| `duration` | The duration of the movement artifact | seconds | float64 | |
| `max_thresh` | Maximum envelope threshold value | μV | float64 | |
| `mean_zscore` | Mean z-score of envelope amplitude | z-score | float64 | |
| `median_zscore` | Median z-score of envelope amplitude | z-score | float64 | |
| `max_zscore` | Maximum z-score of envelope amplitude | z-score | float64 | |
| `min_zscore` | Minimum z-score of envelope amplitude | z-score | float64 | |
| `area` | Area under the envelope curve | μV·s | float64 | |
| `total_energy` | Total energy of the envelope signal | μV²·s | float64 | |

## Gamma Band Events (*_gamma_band_events.csv.gz)

| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `Unnamed: 0` | Row index from original dataset | - | int64 | |
| `start_time` | The start time of the gamma event | seconds | float64 | |
| `end_time` | The end time of the gamma event | seconds | float64 | |
| `duration` | The duration of the gamma event | seconds | float64 | |

## Global SWR Events (session_*_global_swr_events.csv.gz)

| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `Unnamed: 0` | Row index from original dataset | - | int64 | |
| `start_time` | The start time of the global SWR event | seconds | float64 | |
| `end_time` | The end time of the global SWR event | seconds | float64 | |
| `duration` | The duration of the global SWR event | seconds | float64 | |

## Probe Metadata (*_probe_metadata.csv.gz)

| Column Name | Description | Units | Data Type | Notes |
|-------------|-------------|-------|-----------|-------|
| `probe_id` | Unique identifier for the probe | - | string | |
| `total_unit_count` | Total number of units recorded on this probe | count | int64 | |
| `good_unit_count` | Number of good quality units on this probe | count | int64 | |
| `ca1_total_unit_count` | Total CA1 units on this probe | count | int64 | |
| `ca1_good_unit_count` | Good quality CA1 units on this probe | count | int64 | |

## Hierarchical JSON Files

### Selection Metadata (*_selection_metadata.json.gz)
**Format**: JSONL (JSON Lines) - each line is a separate JSON object

**Structure per line**:
```json
{
  "probe_id": "string",
  "ripple_band": {
    "channel_ids": ["array of int64"],
    "depths": ["array of float64"],
    "skewness": ["array of float64"],
    "net_power": ["array of float64"],
    "selected_channel_id": "int64",
    "selection_method": "string"
  },
  "sharp_wave_band": {
    "channel_ids": ["array of int64"],
    "depths": ["array of float64"],
    "net_sw_power": ["array of float64"],
    "modulation_index": ["array of float64"],
    "circular_linear_corrs": ["array of float64"],
    "selected_channel_id": "int64",
    "selection_method": "string"
  }
}
```

| Field | Description | Units | Data Type | Notes |
|-------|-------------|-------|-----------|-------|
| `probe_id` | Unique identifier for the probe | - | string | |
| `channel_ids` | Array of channel IDs for this band | - | array[int64] | |
| `depths` | Electrode depths in micrometers | μm | array[float64] | |
| `skewness` | Signal skewness per channel | - | array[float64] | |
| `net_power` | Net power per channel | μV² | array[float64] | |
| `selected_channel_id` | ID of the selected channel | - | int64 | |
| `selection_method` | Method used for channel selection | - | string | |
| `net_sw_power` | Net sharp wave power per channel | μV² | array[float64] | |
| `modulation_index` | Cross-frequency coupling per channel | - | array[float64] | |
| `circular_linear_corrs` | Phase-amplitude coupling per channel | correlation coefficient | array[float64] | |

### Run Settings (*_run_settings.json.gz)
**Format**: Single JSON object

**Structure**:
```json
{
  "run_name": "string",
  "thresholds": {
    "gamma_event_thresh": "float64",
    "ripple_band_threshold": "float64",
    "movement_artifact_ripple_band_threshold": "float64",
    "merge_events_offset": "float64"
  },
  "global_swr_detection": {
    "min_ca1_units": "int64",
    "min_events_per_probe": "int64",
    "min_filtered_events": "int64",
    "min_sw_power": "float64",
    "merge_window": "float64",
    "min_probe_count": "int64",
    "exclude_gamma": "bool",
    "exclude_movement": "bool",
    "global_rip_label": "string"
  },
  "dataset": "string",
  "sampling_rates": {
    "target_fs": "float64"
  }
}
```

| Field | Description | Units | Data Type | Notes |
|-------|-------------|-------|-----------|-------|
| `run_name` | Name identifier for this processing run | - | string | |
| `gamma_event_thresh` | Z-score threshold for gamma event detection | z-score | float64 | |
| `ripple_band_threshold` | Z-score threshold for ripple detection | z-score | float64 | |
| `movement_artifact_ripple_band_threshold` | Z-score threshold for movement artifact detection | z-score | float64 | |
| `merge_events_offset` | Time window for merging nearby events | seconds | float64 | |
| `min_ca1_units` | Minimum CA1 units required | count | int64 | |
| `min_events_per_probe` | Minimum events per probe | count | int64 | |
| `min_filtered_events` | Minimum filtered events required | count | int64 | |
| `min_sw_power` | Minimum sharp wave power threshold | μV² | float64 | |
| `merge_window` | Time window for merging global events | seconds | float64 | |
| `min_probe_count` | Minimum number of probes required | count | int64 | |
| `exclude_gamma` | Whether to exclude gamma overlapping events | - | bool | |
| `exclude_movement` | Whether to exclude movement overlapping events | - | bool | |
| `global_rip_label` | Label for global ripple events | - | string | |
| `dataset` | Dataset identifier | - | string | |
| `target_fs` | Target sampling frequency | Hz | float64 | |


