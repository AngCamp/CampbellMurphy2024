# Filter Design for SWR Detection

This document describes the process and settings used to design the FIR filters used in the SWR detection pipeline, specifically the ripple band and sharp-wave component filters.

The filters were designed using the MNE-Python library.

## Environment Setup

To design your own filters using the provided Jupyter notebook (`PowerBandFIlters/mne_filter_design_1500hz.ipynb` or similar), you need a Python environment with MNE installed. Follow the instructions on the MNE website for installation. A common way is using Conda:

```bash
# Create a new conda environment (e.g., named 'mne')
conda create --name mne --channel=conda-forge --override-channels python=3.9 mne matplotlib scipy notebook ipython pandas

# Activate the environment
conda activate mne
```

Refer to the official MNE installation guide for the most up-to-date instructions: [https://mne.tools/stable/install/manual_install.html](https://mne.tools/stable/install/manual_install.html)

## Filter Design Principles

*(Placeholder: Describe the general approach from the notebook - e.g., using `mne.filter.create_filter`, FIR design principles, windowing method used, etc.)*

## Filter Specifications

The target sampling frequency for the LFP data after resampling in the main pipeline is 1500 Hz. Filters were designed with this sampling frequency in mind.

### Ripple Band Filter (e.g., `ripplefilter.mat`)

*   **Type:** Band-pass FIR filter
*   **Passband:** [150 Hz - 250 Hz] *(Placeholder: Verify exact frequencies from notebook)*
*   **Stopband Attenuation:** *(Placeholder: Specify from notebook, e.g., 60 dB)*
*   **Passband Ripple:** *(Placeholder: Specify from notebook, e.g., 1 dB)*
*   **Transition Width:** *(Placeholder: Specify from notebook, e.g., 20 Hz)*
*   **Filter Length/Order:** *(Placeholder: Specify from notebook)*
*   **Design Method:** *(Placeholder: Specify from notebook, e.g., Hamming window, firwin, remez)*

*(Placeholder: Include the filter plot image or key MNE code snippet from the notebook if possible/desired)*

### Sharp-Wave Component Filter (e.g., `sharpwavefilter.mat`)

*   **Type:** Band-pass FIR filter
*   **Passband:** [8 Hz - 40 Hz] *(Placeholder: Verify exact frequencies from notebook)*
*   **Stopband Attenuation:** *(Placeholder: Specify from notebook)*
*   **Passband Ripple:** *(Placeholder: Specify from notebook)*
*   **Transition Width:** *(Placeholder: Specify from notebook)*
*   **Filter Length/Order:** *(Placeholder: Specify from notebook)*
*   **Design Method:** *(Placeholder: Specify from notebook)*

*(Placeholder: Include the filter plot image or key MNE code snippet from the notebook if possible/desired)*

## Saving Filters

The designed filter coefficients (typically the numerator `b` for FIR filters) were saved into `.mat` files (e.g., using `scipy.io.savemat`) for use by the main detection script.

```python
# Example using scipy (Placeholders - adapt from notebook)
import scipy.io

# Assuming filter_coeffs_ripple is the numpy array of coefficients
scipy.io.savemat('ripplefilter.mat', {'ripplefilter': filter_coeffs_ripple})

# Assuming filter_coeffs_sw is the numpy array of coefficients
scipy.io.savemat('sharpwavefilter.mat', {'sharpwavefilter': filter_coeffs_sw})
``` 