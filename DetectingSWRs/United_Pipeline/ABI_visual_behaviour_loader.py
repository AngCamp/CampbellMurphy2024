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

    def select_sharpwave_channel(self, ca1_lfp, lfp_time_index, ca1_chan_ids, 
                                this_chan_id, channel_positions, ripple_filtered, running_exclusion_periods=None):
        """
        Selects the optimal sharp wave channel based on correlation with ripple power.
        
        Parameters
        ----------
        ca1_lfp : numpy.ndarray
            LFP data for CA1 channels
        lfp_time_index : numpy.ndarray
            Time index for the LFP data
        ca1_chan_ids : list
            List of CA1 channel IDs
        this_chan_id : int
            Channel ID of the selected ripple channel
        channel_positions : pandas.Series
            Vertical positions of channels
        ripple_filtered : numpy.ndarray
            Already filtered signal in the ripple band
        running_exclusion_periods : list, optional
            List of (start, end) tuples for periods to exclude
            
        Returns
        -------
        dict
            Dictionary with best sharpwave channel information
        """
        
        # Load the pre-designed filter
        filter_data = np.load(self.sw_component_filter_path)
        sharpwave_filter = filter_data['sharpwave_componenet_8to40band_1500hz_band']
        
        # Create exclusion mask
        mask = np.ones_like(lfp_time_index, dtype=bool)
        # Exclude first 3.5 seconds and last 3.5 seconds
        mask &= (lfp_time_index > 3.5) & (lfp_time_index < (lfp_time_index[-1] - 3.5))

        if running_exclusion_periods:
            for start_time, end_time in running_exclusion_periods:
                mask &= ~((lfp_time_index >= start_time) & (lfp_time_index <= end_time))
        
        # Find channels below the reference
        this_chan_position = channel_positions.loc[this_chan_id]
        below_channels_mask = channel_positions > this_chan_position
        below_channel_ids = channel_positions[below_channels_mask].index.values
        
        # Map channel IDs to indices
        chan_id_to_idx = {chan_id: i for i, chan_id in enumerate(ca1_chan_ids)}
        below_channel_indices = [chan_id_to_idx[chan_id] for chan_id in below_channel_ids if chan_id in chan_id_to_idx]
        
        # Calculate ripple power from the filtered signal
        ripple_power = np.abs(hilbert(ripple_filtered)) ** 2
        
        # Z-score the ripple power
        ripple_power_z = (ripple_power - np.mean(ripple_power[mask])) / np.std(ripple_power[mask])
        
        # Get analytic signal and extract instantaneous phase of ripple band
        ripple_analytic = hilbert(ripple_filtered)
        ripple_phase = np.angle(ripple_analytic)
        
        # Create a dictionary to store results for each channel
        channel_results = {}
        
        # Process all channels
        for i, idx in enumerate(below_channel_indices):
            chan_id = below_channel_ids[i]
            
            # Apply sharpwave filter using convolve
            sw_filtered = fftconvolve(ca1_lfp[:, idx].reshape(-1), sharpwave_filter, mode="same")
            
            # Get power
            sw_power = np.abs(hilbert(sw_filtered)) ** 2
            
            # Z-score the sharpwave power
            sw_power_z = (sw_power - np.mean(sw_power[mask])) / np.std(sw_power[mask])
            
            # Calculate correlation for all valid points
            valid_sw = sw_power[mask]
            valid_ripple = ripple_power[mask]
            correlation = np.corrcoef(valid_sw, valid_ripple)[0, 1]
            
            # Create masks for high-power periods
            high_ripple_mask = (ripple_power_z > 1) & mask
            high_sw_mask = (sw_power_z > 1) & mask
            high_both_mask = high_ripple_mask & high_sw_mask
            
            # Calculate phase coherence during high power periods (both signals > 1 SD)
            # Get instantaneous phase of sharpwave component
            sw_analytic = hilbert(sw_filtered)
            sw_phase = np.angle(sw_analytic)
            
            # Calculate phase coherence during high power periods (both signals > 1 SD)
            phase_diff = ripple_phase - sw_phase
            if np.sum(high_both_mask) > 10:  # Need at least some points for meaningful coherence
                phase_coherence_high_power = np.abs(np.mean(np.exp(1j * phase_diff[high_both_mask])))
            else:
                phase_coherence_high_power = np.nan
            
            # Store results in dictionary without storing the filtered signal
            channel_results[chan_id] = {
                'vertical_position': channel_positions.loc[chan_id],
                'correlation': correlation,
                'phase_coherence_high_power': phase_coherence_high_power,
                'idx': idx,  # Store the index for later use
                'sw_power_z': sw_power_z  # Store z-scored power
            }
        
        # Find the channel with best phase coherence during high ripple power
        best_channel_id = max(
            channel_results.keys(),
            key=lambda k: channel_results[k]['phase_coherence_high_power'] 
                        if not np.isnan(channel_results[k]['phase_coherence_high_power']) 
                        else -float('inf')
        )
        
        # Get the raw LFP for the best channel and re-compute filtered signal
        best_channel_idx = channel_results[best_channel_id]['idx']
        sharp_wave_lfp = ca1_lfp[:, best_channel_idx]
        
        # Re-apply sharpwave filter for the best channel
        sharpwave_filtered = fftconvolve(sharp_wave_lfp.reshape(-1), sharpwave_filter, mode="same")
        
        # Get the z-scored power
        best_channel_power_z = channel_results[best_channel_id]['sw_power_z']
        
        # Create a results dictionary
        sw_results = {
            'best_channel_id': best_channel_id,
            'sharp_wave_lfp': sharp_wave_lfp,
            'sw_power_z': best_channel_power_z  # Include z-scored power of the selected channel
        }
        
        self.sw_channel_info
        
        return sw_results

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

            sw_results = self.select_sharpwave_channel(
                ca1_lfp=lfp_ca1,
                lfp_time_index=lfp_time_index,  
                ca1_chan_ids=lfp_ca1_chans,
                this_chan_id=this_chan_id,
                channel_positions=ca1_channel_positions,
                ripple_filtered=peakrippleband
            )

            # Extract sharpwave channel information
            best_sw_chan_id = sw_results['best_channel_id']
            best_sw_chan_lfp = sw_results['sharp_wave_lfp']
            best_sw_power_z = sw_results['sw_power_z']
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