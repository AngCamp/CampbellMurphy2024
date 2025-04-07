# ABI Loaders
import time
import os
import numpy as np
import yaml
from scipy import io, signal, stats
from scipy.signal import lfilter
import scipy.ndimage
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
from scipy import interpolate
from scipy.signal import hilbert, fftconvolve
import matplotlib.pyplot as plt
from allensdk.brain_observatory.behavior.behavior_project_cache import (
    VisualBehaviorNeuropixelsProjectCache,
)

class abi_visual_behaviour_loader:
    def __init__(self, session_id):
        """
        Initialize the ABI loader with a session ID.
        
        Parameters
        ----------
        session_id : int
            The ABI ecephys session ID
        """
        self.session_id = session_id
        self.cache = None
        self.session = None
        self.probe_id_list = None
        self.probes_of_interest = None
        self.sw_channel_info = None
        
    def set_up(self, cache_directory=None):
        """
        Sets up the EcephysProjectCache and loads the session.
        
        Parameters
        ----------
        cache_directory : str, optional
            Directory where to store the cache. If None, uses default.
            
        Returns
        -------
        self : abi_loader
            Returns the instance for method chaining.
        """
        # Set up the cache
        config_path = os.environ.get('CONFIG_PATH', 'expanded_config.yaml')
        
        with open(config_path, "r") as f:
            raw_content = f.read()
            # Replace environment variables
            for key, value in os.environ.items():
                raw_content = raw_content.replace(f"${key}", value)
            full_config = yaml.safe_load(raw_content)
        dataset_config = full_config["abi_visual_behaviour"]
        sdk_cache_dir = dataset_config["sdk_cache_dir"]
        manifest_path = os.path.join(sdk_cache_dir, "manifest.json")
        self.cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=sdk_cache_dir)

        # provide the sharp wave component band filter
        self.sw_component_filter_path = full_config['filters']['sw_component_filter']
        
        # Load the session
        self.session = self.cache.get_ecephys_session(ecephys_session_id=self.session_id)
        self.session.channels = self.session.get_channels()
        
        print(f"Session {self.session_id} loaded")
        return self
    
    def has_ca1_channels(self):
        """
        Checks if the session includes CA1 channels.
        
        Returns
        -------
        bool
            True if CA1 channels exist, False otherwise
        """
        has_ca1 = np.isin("CA1", list(self.session.channels.structure_acronym.unique()))
        
        if not has_ca1:
            print(f"Session {self.session_id} does not have CA1 channels")
            
        return has_ca1
    
    def get_probes_with_ca1(self):
        """
        Gets the list of probes that have CA1 channels.
        
        Returns
        -------
        list
            List of probe IDs with CA1 channels
        """
        # Get probes with LFP data
        probes_table_df = self.cache.get_probe_table()
        valid_lfp = probes_table_df[probes_table_df["has_lfp_data"]]
        
        # Get probes for this session
        self.probe_id_list = list(
            valid_lfp[valid_lfp.ecephys_session_id == self.session_id].index
        )
        
        # Find probes with CA1 channels
        self.probes_of_interest = []
        for probe_id in self.probe_id_list:
            has_ca1_and_exists = np.isin(
                "CA1",
                list(
                    self.session.channels[
                        self.session.channels.probe_id == probe_id
                    ].structure_acronym.unique()
                ),
            )
            if has_ca1_and_exists:
                self.probes_of_interest.append(probe_id)
        
        print(f"Found {len(self.probes_of_interest)} probes with CA1 channels")
        return self.probes_of_interest

    def _compute_modulation_index(self, sw_phase, ripple_amp, valid_mask, n_bins=18):
        """
        Compute the circular-linear correlation between sharp wave phase and ripple amplitude.

        Parameters
        ----------
        sw_phase : np.ndarray
            Instantaneous phase of the sharp wave filtered signal (in radians).
        ripple_amp : np.ndarray
            Instantaneous amplitude envelope of the ripple filtered signal.
        valid_mask : np.ndarray of bool
            Boolean mask selecting valid time points to include (e.g., high power, no artifacts).

        Returns
        -------
        float
            Circular-linear correlation coefficient. Returns np.nan if insufficient data is available.
        """

        sw_phase = sw_phase[valid_mask]
        ripple_amp = ripple_amp[valid_mask]
        if ripple_amp.size < 10:
            return np.nan

        bins = np.linspace(-np.pi, np.pi, n_bins + 1)
        digitized = np.digitize(sw_phase, bins) - 1
        amp_by_bin = np.array([
            ripple_amp[digitized == j].mean() if np.any(digitized == j) else 0
            for j in range(n_bins)
        ])
        if amp_by_bin.sum() == 0:
            return np.nan

        P = amp_by_bin / amp_by_bin.sum()
        H = -np.sum(P * np.log(P + 1e-10))
        H_max = np.log(n_bins)
        return (H_max - H) / H_max


    def _compute_circular_linear_corr(self, sw_phase, ripple_amp, valid_mask):
        """
        Compute the circular-linear correlation between sharp wave phase and ripple amplitude.

        Parameters
        ----------
        sw_phase : np.ndarray
            Instantaneous phase of the sharp wave filtered signal (in radians).
        ripple_amp : np.ndarray
            Instantaneous amplitude envelope of the ripple filtered signal.
        valid_mask : np.ndarray of bool
            Boolean mask selecting valid time points to include (e.g., high power, no artifacts).

        Returns
        -------
        float
            Circular-linear correlation coefficient. Returns np.nan if insufficient data is available.
        """
        sw_phase = sw_phase[valid_mask]
        ripple_amp = ripple_amp[valid_mask]
        if ripple_amp.size < 10:
            return np.nan

        weighted_cos = np.sum(ripple_amp * np.cos(sw_phase))
        weighted_sin = np.sum(ripple_amp * np.sin(sw_phase))
        R = np.sqrt(weighted_cos ** 2 + weighted_sin ** 2)
        norm = np.sqrt(np.sum(ripple_amp ** 2))
        return R / norm


    def select_sharpwave_channel(
            self,
            ca1_lfp,
            lfp_time_index,
            ca1_chan_ids,
            this_chan_id,
            channel_positions,
            ripple_filtered,
            running_exclusion_periods=None,
            selection_metric='modulation_index'):
        """
        Select the optimal sharp wave channel based on phase-amplitude coupling with ripple activity.

        Parameters:
        [existing parameters]

        Returns:
        tuple
            (best_channel_id, best_channel_lfp) - The best channel ID and its raw LFP data
        """
        filter_data = np.load(self.sw_component_filter_path)
        sharpwave_filter = filter_data['sharpwave_componenet_8to40band_1500hz_band']

        mask = (lfp_time_index > 3.5) & (lfp_time_index < lfp_time_index[-1] - 3.5)
        if running_exclusion_periods:
            for start, end in running_exclusion_periods:
                mask &= ~((lfp_time_index >= start) & (lfp_time_index <= end))

        ripple_an = hilbert(ripple_filtered)
        ripple_phase = np.angle(ripple_an)
        ripple_amp = np.abs(ripple_an)
        ripple_power = ripple_amp ** 2
        ripple_power_z = (ripple_power - ripple_power[mask].mean()) / ripple_power[mask].std()

        ref_depth = channel_positions.loc[this_chan_id]
        below_ids = channel_positions[channel_positions > ref_depth].index
        id_to_idx = {cid: i for i, cid in enumerate(ca1_chan_ids)}
        below_idx = [id_to_idx[cid] for cid in below_ids if cid in id_to_idx]

        # Create dictionary to store detailed results for selection logic
        results = {}
        
        # Create a separate dictionary for JSON-serializable summary stats
        self.sw_component_summary_stats_dict = {
            'channel_ids': [],
            'modulation_indices': [],
            'circular_linear_corrs': [],
            'depths': [],
            'selection_metric': selection_metric,
            'best_channel_id': None  # Will be filled in after selection
        }

        for cid, idx in zip(below_ids, below_idx):
            sw_filt = fftconvolve(ca1_lfp[:, idx], sharpwave_filter, mode='same')
            sw_an = hilbert(sw_filt)
            sw_phase = np.angle(sw_an)
            sw_power = np.abs(sw_an) ** 2
            sw_power_z = (sw_power - sw_power[mask].mean()) / sw_power[mask].std()

            high_mask = (ripple_power_z > 1) & (sw_power_z > 1) & mask

            modulation_index = self._compute_modulation_index(sw_phase, ripple_amp, high_mask)
            circular_linear_corr = self._compute_circular_linear_corr(sw_phase, ripple_amp, high_mask)

            # Store detailed results for selection
            results[cid] = {
                'modulation_index': modulation_index,
                'circular_linear_corr': circular_linear_corr,
                'idx': idx,
                'sw_power_z': sw_power_z
            }
            
            # Store JSON-serializable summary stats
            self.sw_component_summary_stats_dict['channel_ids'].append(int(cid))
            self.sw_component_summary_stats_dict['modulation_indices'].append(float(modulation_index) if not np.isnan(modulation_index) else None)
            self.sw_component_summary_stats_dict['circular_linear_corrs'].append(float(circular_linear_corr) if not np.isnan(circular_linear_corr) else None)
            self.sw_component_summary_stats_dict['depths'].append(float(channel_positions.loc[cid]))

        # Select best channel
        best_cid = max(
            results,
            key=lambda k: results[k][selection_metric] if not np.isnan(results[k][selection_metric]) else -np.inf
        )

        best_idx = results[best_cid]['idx']
        best_lfp = ca1_lfp[:, best_idx]
        
        # Store best channel ID in summary stats
        self.sw_component_summary_stats_dict['best_channel_id'] = int(best_cid)
        
        # Return the best channel ID and its raw LFP
        return best_cid, best_lfp

    def process_probe(self, probe_id, filter_ripple_band_func=None):
        """
        Processes a single probe to extract CA1 and control channels.
        
        Parameters
        ----------
        probe_id : int
            ID of the probe to process
        filter_ripple_band_func : function, optional
            Function to filter for ripple band
            
        Returns
        -------
        dict
            Dictionary with processing results
        """
        print(f"Processing probe: {probe_id}")
        
        # Get LFP for the probe
        lfp = self.session.get_lfp(probe_id)
        og_lfp_obj_time_vals = lfp.time.values
        # Get control channels outside hippocampus
        idx = self.session.channels.probe_id == probe_id
        organisedprobechans = self.session.channels[idx].sort_values(
            by="probe_vertical_position"
        )
        organisedprobechans = organisedprobechans[
            np.isin(organisedprobechans.index.values, lfp.channel.values)
        ]
        
        # Find channels outside hippocampus
        not_a_ca1_chan = np.logical_not(
            np.isin(
                organisedprobechans.structure_acronym,
                ["CA3", "CA2", "CA1", "HPF", "EC", "DG"],
            )
        )
        
        # Choose two random channels
        take_two = np.random.choice(
            organisedprobechans.index[not_a_ca1_chan], 2, replace=False
        )
        control_channels = []
        
        # Get LFP for control channels
        for channel_outside_hp in take_two:
            movement_control_channel = lfp.sel(channel=channel_outside_hp)
            movement_control_channel = movement_control_channel.to_numpy()
            # Resample to match CA1 data
            movement_control_channel, lfp_time_index = self.resample_signal(movement_control_channel, lfp.time.values, 1500.0)
            # needed for ripple detector method
            #movement_control_channel = interp_func(lfp_time_index)
            movement_control_channel = movement_control_channel[:, None]
            control_channels.append(movement_control_channel)
        
        # Get CA1 channels for this probe
        ca1_chans = self.session.channels.probe_channel_number[
            (self.session.channels.probe_id == probe_id)
            & (self.session.channels.structure_acronym == "CA1")
        ]
        ca1_idx = np.isin(lfp.channel.values, ca1_chans.index.values)
        ca1_idx = lfp.channel.values[ca1_idx]
        
        # Select CA1 channels
        lfp_ca1 = lfp.sel(channel=ca1_idx)
        del lfp
        lfp_ca1 = lfp_ca1.to_pandas()
        lfp_ca1_chans = lfp_ca1.columns
        lfp_ca1 = lfp_ca1.to_numpy()
        
        # Check for NaNs
        if np.isnan(lfp_ca1).any():
            print(f"NaN detected in LFP data for probe {probe_id}, skipping")
            return None
        
        # Resample to 1500 Hz
        lfp_ca1, lfp_time_index = self.resample_signal(
            lfp_ca1, og_lfp_obj_time_vals, 1500.0
        )
        
        # Find channel with highest ripple power if function provided
        if filter_ripple_band_func is not None:
            lfp_ca1_rippleband = filter_ripple_band_func(lfp_ca1)
            highest_rip_power = np.abs(signal.hilbert(lfp_ca1_rippleband)) ** 2
            highest_rip_power = highest_rip_power.max(axis=0)
            
            # Get channel with highest ripple power
            peak_chan_idx = highest_rip_power.argmax()
            this_chan_id = int(lfp_ca1_chans[peak_chan_idx])
            peakrippleband = lfp_ca1_rippleband[:, peak_chan_idx]
            peakripchan_lfp_ca1 = lfp_ca1[:, lfp_ca1_chans == this_chan_id]

            # Get channel positions for CA1 channels
            ca1_channel_positions = self.session.channels.loc[lfp_ca1_chans, 'probe_vertical_position']

            best_sw_chan_id, best_sw_chan_lfp = self.select_sharpwave_channel(ca1_lfp=lfp_ca1,
                                                                                lfp_time_index=lfp_time_index,  
                                                                                ca1_chan_ids=lfp_ca1_chans,
                                                                                this_chan_id=this_chan_id,
                                                                                channel_positions=ca1_channel_positions,
                                                                                ripple_filtered=peakrippleband)

            # Extract sharpwave channel information
            best_sw_power_z = scipy.stats.zscore(best_sw_chan_lfp)
        else:
            peak_chan_idx = None
            this_chan_id = None
            peakrippleband = None
            peakripchan_lfp_ca1 = None
            best_sw_chan_id = None
            best_sw_chan_lfp = None
            best_sw_power_z = None
        del lfp_ca1

        
        # Collect results
        results = {
            'probe_id': probe_id,
            'lfp_time_index': lfp_time_index,
            'ca1_chans': lfp_ca1_chans,
            'control_lfps': control_channels,
            'control_channels': take_two,
            'peak_ripple_chan_idx': peak_chan_idx,
            'peak_ripple_chan_id': this_chan_id,
            'peak_ripple_chan_raw_lfp': peakripchan_lfp_ca1,
            'chan_id_string': str(this_chan_id) if this_chan_id is not None else None,
            'rippleband': peakrippleband,
            'sharpwave_chan_id': best_sw_chan_id,
            'sharpwave_chan_raw_lfp': best_sw_chan_lfp,
            'sharpwave_power_z': best_sw_power_z
        }
        
        return results

    def resample_signal(self, signal_data, time_values, target_fs=1500.0):
        """
        Resamples a signal to the target sampling frequency.
        
        Parameters
        ----------
        signal_data : numpy.ndarray
            Signal data to resample
        time_values : numpy.ndarray
            Time values corresponding to the signal data
        target_fs : float, optional
            Target sampling frequency
            
        Returns
        -------
        tuple
            (resampled_signal, new_time_values)
        """
        # Create new time index
        t_start = time_values[0]
        t_end = time_values[-1]
        dt_new = 1.0 / target_fs
        n_samples = int(np.ceil((t_end - t_start) / dt_new))
        new_time_values = t_start + np.arange(n_samples) * dt_new
        
        # Resample signal
        if signal_data.ndim == 1:
            # For 1D signals
            interp_func = interpolate.interp1d(
                time_values, signal_data, bounds_error=False, fill_value="extrapolate"
            )
            resampled = interp_func(new_time_values)
        else:
            # For multi-channel signals
            #resampled = np.zeros((signal_data.shape[0], len(new_time_values)))
            resampled = np.zeros((len(new_time_values), signal_data.shape[1]))
            for i in range(signal_data.shape[1]):
                interp_func = interpolate.interp1d(
                    time_values, signal_data[:, i], bounds_error=False, fill_value="extrapolate"
                )
                resampled[:, i] = interp_func(new_time_values)
        
        return resampled, new_time_values
    
    def cleanup(self):
        """
        Cleans up resources to free memory.
        """
        self.session = None
        
def session_id_with_ca1_generator():
    sessions = cache.get_ecephys_session_table()
    # Iterate over each session_id in the DataFrame's index
    for session_id in sessions.index:
        # Check if 'CA1' is in the structure_acronyms list for the current session_id
        if "CA1" in sessions.loc[session_id, "structure_acronyms"]:
            # If 'CA1' is found, append the session_id to the list
            sessions_with_CA1.append(session_id)