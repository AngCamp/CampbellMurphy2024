# plots QC metrics on filtered rippels to filter probes
# causing excess kurtosis in lognormal distribution of number of ripples per probe
# can be merged with filtering, but for now is a separate script
# also may need to be adjusted when considering different hippocampal regions

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy.stats import kurtosis
import os

def compute_kurtosis_threshold(data):
    threshold = np.min(data)
    while True:
        mask = data > threshold
        filtered_data = data[mask]
        kurt = kurtosis(filtered_data)
        if kurt<3:
            return threshold
        threshold += 0.1


filtered_events_folder_path = '/space/scratch/allen_visbehave_swr_data/allen_visbehave_swr_5sdthresh/filtered_swrs'

# eventspersession_df needs to be read in
events_file_path = os.path.join(filtered_events_folder_path, 'eventspersession_df.csv')
eventspersession_df = pd.read_csv(events_file_path, index_col=0)  

# Create a density plot
#plt.figure(figsize=(10, 6))
sns.kdeplot(data=eventspersession_df, x='ripple_number')

# Set the x-axis limits
plt.xlim(0, None) 

# Calculate the median
median_ripple_number = eventspersession_df['ripple_number'].median()
mean_ripple_number = eventspersession_df.ripple_number.mean()
lower_bound = eventspersession_df.ripple_number.quantile(0.125)
upper_bound = eventspersession_df.ripple_number.quantile(0.875)
lower_bound_5th = eventspersession_df.ripple_number.quantile(0.05)
upper_bound_95th = eventspersession_df.ripple_number.quantile(0.95)

# Add a vertical line for the median
plt.axvline(median_ripple_number, color='black', linestyle='-')
plt.axvline(mean_ripple_number, color='red', linestyle='-')

# Add vertical lines for the lower and upper bounds
plt.axvline(lower_bound, color='grey', linestyle='--')
plt.axvline(upper_bound, color='grey', linestyle='--')
plt.axvline(lower_bound_5th, color='grey', linestyle='--')
plt.axvline(upper_bound_95th, color='grey', linestyle='--')

# Create legend handles
median_patch = mpatches.Patch(color='black', label='Median')
mean_patch = mpatches.Patch(color='red', label='Mean')
bounds_patch = mpatches.Patch(color='grey', label='Bounds')

# Add labels for the bounds
# Add labels for the bounds
ymax = plt.gca().get_ylim()[1]
plt.text(lower_bound, ymax*0.1, '12.5th percentile', horizontalalignment='left')
plt.text(upper_bound, ymax*0.1, '87.5th percentile', horizontalalignment='left')
plt.text(lower_bound_5th, ymax*0.2, '5th percentile', horizontalalignment='left')
plt.text(upper_bound_95th, ymax*0.2, '95th percentile', horizontalalignment='left')


# Add the legend to the plot
plt.legend(handles=[median_patch, mean_patch, bounds_patch])

# Set plot title and labels
plt.title('Number of Ripples per Probe')
plt.xlabel('Ripple Number')

# Show the plot
plt.show()



# Create a density plot
#plt.figure(figsize=(10, 6))
eventspersession_df['ripple_number_logep1'] = np.log10(eventspersession_df['ripple_number']+1)

sns.kdeplot(data=eventspersession_df, x='ripple_number_logep1')

# Set the x-axis limits
plt.xlim(0, None) 
plt.figure(figsize=(10, 10))
# Show the plot
plt.savefig('Ripple_per_probe_distribution_ABI.png')


# Calculate the median
median_ripple_number = eventspersession_df.ripple_number_logep1.median()
mean_ripple_number = eventspersession_df.ripple_number_logep1.mean()
lower_bound = eventspersession_df.ripple_number_logep1.quantile(0.125)
upper_bound = eventspersession_df.ripple_number_logep1.quantile(0.875)
lower_bound_5th = eventspersession_df.ripple_number_logep1.quantile(0.05)
upper_bound_95th = eventspersession_df.ripple_number_logep1.quantile(0.95)

# Add a vertical line for the median
plt.axvline(median_ripple_number, color='black', linestyle='-')
plt.axvline(mean_ripple_number, color='red', linestyle='-')

# Add vertical lines for the lower and upper bounds
plt.axvline(lower_bound, color='grey', linestyle='--')
plt.axvline(upper_bound, color='grey', linestyle='--')
plt.axvline(lower_bound_5th, color='grey', linestyle='--')
plt.axvline(upper_bound_95th, color='grey', linestyle='--')

# Create legend handles
median_patch = mpatches.Patch(color='black', label='Median')
mean_patch = mpatches.Patch(color='red', label='Mean')
bounds_patch = mpatches.Patch(color='grey', label='Bounds')

# Add labels for the bounds
# Add labels for the bounds
ymax = plt.gca().get_ylim()[1]
plt.text(lower_bound, ymax*0.1, '12.5th percentile', horizontalalignment='right')
plt.text(upper_bound, ymax*0.1, '87.5th percentile', horizontalalignment='left')
plt.text(lower_bound_5th, ymax*0.2, '5th percentile', horizontalalignment='left')
plt.text(upper_bound_95th, ymax*0.2, '95th percentile', horizontalalignment='left')


# Add the legend to the plot
plt.legend(handles=[median_patch, mean_patch, bounds_patch])

# Set plot title and labels
plt.title('Number of Ripples per Probe')
plt.xlabel('Ripple Number log10(x+1)')

plt.figure(figsize=(10, 10))
# Show the plot
plt.savefig('Ripple_per_probe_distribution_log10_ABI.png')


# filtering out ripple outside the bounds
thresh = compute_kurtosis_threshold(eventspersession_df.ripple_number_logep1)

threshold_log10_df = eventspersession_df[eventspersession_df.ripple_number_logep1>thresh]
sns.kdeplot(data=threshold_log10_df, x='ripple_number_logep1')

# Set the x-axis limits
plt.xlim(0, None) 

# Calculate the median
median_ripple_number = threshold_log10_df.ripple_number_logep1.median()
mean_ripple_number = threshold_log10_df.ripple_number_logep1.mean()
lower_bound = threshold_log10_df.ripple_number_logep1.quantile(0.125)
upper_bound = threshold_log10_df.ripple_number_logep1.quantile(0.875)
lower_bound_5th_new= threshold_log10_df.ripple_number_logep1.quantile(0.05)
upper_bound_95th_new = threshold_log10_df.ripple_number_logep1.quantile(0.95)

# Add a vertical line for the median
plt.axvline(median_ripple_number, color='black', linestyle='-')
plt.axvline(mean_ripple_number, color='red', linestyle='-')

# Add vertical lines for the lower and upper bounds
plt.axvline(lower_bound, color='grey', linestyle='--')
plt.axvline(upper_bound, color='grey', linestyle='--')
plt.axvline(lower_bound_5th_new, color='grey', linestyle='--')
plt.axvline(upper_bound_95th_new, color='grey', linestyle='--')

# Create legend handles
median_patch = mpatches.Patch(color='black', label='Median')
mean_patch = mpatches.Patch(color='red', label='Mean')
bounds_patch = mpatches.Patch(color='grey', label='Bounds')

# Add labels for the bounds
ymax = plt.gca().get_ylim()[1]
plt.text(lower_bound, ymax*0.1, '12.5th percentile', horizontalalignment='right')
plt.text(upper_bound, ymax*0.1, '87.5th percentile', horizontalalignment='left')
plt.text(lower_bound_5th_new, ymax*0.2, '5th percentile', horizontalalignment='right')
plt.text(upper_bound_95th_new, ymax*0.2, '95th percentile', horizontalalignment='left')

# Add the legend to the plot
plt.legend(handles=[median_patch, mean_patch, bounds_patch])

# Set plot title and labels
plt.title('Number of Ripples per Probe')
plt.xlabel('Ripple Number log10(x+1)')

plt.figure(figsize=(10, 10))
# Show the plot

plt.show()

mask = (eventspersession_df.ripple_number_logep1>lower_bound_5th_new)
print(sum(mask))

eventspersession_df['within_lognormal_bounds'] = mask

# write it back to file with probe anotated as being worth keeping or not
eventspersession_df.to_csv(events_file_output_path, index=True)
