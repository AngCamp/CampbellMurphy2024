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
import matplotlib.pyplot as plt
from scipy import signal, interpolate
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache


class abi_visual_coding_loader:
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
        config_path = os.environ.get('CONFIG_PATH', 'united_detector_config.yaml')
        with open(config_path, "r") as f:
            config_content = f.read()
            full_config = yaml.safe_load(config_content)
        dataset_config = full_config["abi_visual_coding"]
        sdk_cache_dir = dataset_config["sdk_cache_dir"]
        manifest_path = os.path.join(sdk_cache_dir, "manifest.json")
        self.cache = EcephysProjectCache.from_warehouse(manifest=manifest_path)

        # Load the session
        self.session = self.cache.get_session_data(self.session_id)
        
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
        has_ca1 = np.isin("CA1", list(self.session.channels.ecephys_structure_acronym.unique()))
        
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
        
        # Get probes for this session
        self.probe_id_list = [int(item) for item in self.session.channels.probe_id.unique()]
        
        # Find probes with CA1 channels
        self.probes_of_interest = []
        for probe_id in self.probe_id_list:
            has_ca1_and_exists = np.isin(
                "CA1",
                list(
                    self.session.channels[
                        self.session.channels.probe_id == probe_id
                    ].ecephys_structure_acronym.unique()
                ),
            )
            if has_ca1_and_exists:
                self.probes_of_interest.append(probe_id)
        
        print(f"Found {len(self.probes_of_interest)} probes with CA1 channels")
        return self.probes_of_interest
    
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
                organisedprobechans.ecephys_structure_acronym,
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
        else:
            peak_chan_idx = None
            this_chan_id = None
            peakrippleband = None
            peakripchan_lfp_ca1 = None
        del lfp_ca1

        
        # Collect results
        results = {
            'probe_id': probe_id,
            #'lfp_ca1': lfp_ca1,
            'lfp_time_index': lfp_time_index,
            'ca1_chans': lfp_ca1_chans,
            'control_lfps': control_channels,
            'control_channels': take_two,
            'peak_ripple_chan_idx': peak_chan_idx,
            'peak_ripple_chan_id': this_chan_id,
            'peak_ripple_chan_raw_lfp': peakripchan_lfp_ca1,
            'chan_id_string': str(this_chan_id) if this_chan_id is not None else None,
            'rippleband': peakrippleband
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