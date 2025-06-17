Create a new python script in this folder.  The script will identify sessions and events with the most spiking in it to make a raster plot of spiking.

1) Identifying the best sessions to use.
We first identify sessions with enough good units in all the regions we are interested in for our analysis.  This can be done from the cache level units table.  From this session we then compute the neurons that have the most significant during event vs baseline manwhitney U test results based on 0.01s binned spike counts from times during sessions and times not within 0.2s of any event.  We count the bins as samples then we take the man whitney u test of the during events vs baseline samples.  Save the direction of the effect, and the pvalue.  Then we save this pvalue.  

Once we have done this for every unit in every session we use the benjanimin hochberg correction on the pvalues to find units that are stil significant FDR <0.05.  Then we sum the number of significant good units session and return the session with the most number of units on it.

2) Finding the best event to plot.  Find the event that has the greatest number of spikes from all the units with a significant score within it vs times +/- 0.5s around the event.

3) The plot will go from the unit with the greatest effect to the unit with the lowest effect top to bottom for each region.  We then plot these units for each region.  We specific the order of the regions, top to bottom by their order int the TARGET_REGIONS list.

