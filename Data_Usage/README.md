In the @/home/acampbell/NeuropixelsLFPOnRamp/Data Usage folder I would like  a .py script set up for copy pasting into a notebook.  The stuff that shuold go into markdown can be put inside """ """ coments so that I cna just cp[yt paste them  into the notebook.  Import all the lirbaries, mention at the top this is for the ABI Visual Behavious and requires the allensdk_env conda environment to be loaded.

I want you to use the code in @/home/acampbell/NeuropixelsLFPOnRamp/Figures_and_Technical_Validation/Relating_SWR_to_other_data/specific_swr_spike_raster_plot.py   and @/home/acampbell/NeuropixelsLFPOnRamp/Figures_and_Technical_Validation/Relating_SWR_to_other_data/swrs_pupile_and_running_plot.py

I want you to start by loading our events data.  Explain the folder structure here as well. 


Then filtering it based on the threshold I layed out.  3SD<power_max_zscore<10SD, no gamma or movement, and a minimum sw_peak_power >1SD.

Then we show how to align it to the CA1 units.

Then we show how to align it to running speed.  Generally speaking keep the plots simple.

Note that the pupil data seems to have lots of gaps in it.  Running can as well.  This needs to be checked session by session.  

Key things to highlight is that time is the unifying factor here.  All the other objects in the session can be aligned by the time.  When loading a piece of data it is useful to print it out.
