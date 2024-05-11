# load the dataframe containing the putative ripples, filter for ones matching cirteria
    putative_ripples_df = pd.read_csv(os.path.join(session_path, probe_file_list[probe_num]))
    print('putative ripples df')

    probe_id = probe_id_list[probe_num]
    # check if the events overlap with each other
    # check if they overlap with gamma events

    
    # check if they overlap with gamma events
    gamma_event_file = get_files_with_substrings(session_path, substring1='probe_' + str(probe_id), substring2='gamma_band_events')
    gamma_events = pd.read_csv(os.path.join(session_path, gamma_event_file[0]), index_col=0)
    putative_ripples_df['Overlaps_with_gamma'] = check_overlap(putative_ripples_df, gamma_events)
    print('Gamma events')

    # now check if the overlapping non hippocampal HFEs also overlap with the hippocampal HFEs
    # if they do we mark them in the dataframe
    # check if the HFE events in the non hippocampal channels overlap
    movement_channels_files = get_files_with_substrings(session_path, substring1='probe_' + str(probe_id), substring2='movement_artifacts')
    movement_channel_1 = pd.read_csv(os.path.join(session_path, movement_channels_files[0]))
    movement_channel_2 = pd.read_csv(os.path.join(session_path, movement_channels_files[1]))
    if movement_channel_1.shape[0] > movement_channel_2.shape[0]:
        overlapping_artifacts = movement_channel_1[check_overlap(movement_channel_1, movement_channel_2)]
    elif movement_channel_1.shape[0] < movement_channel_2.shape[0]:
        overlapping_artifacts = movement_channel_2[check_overlap(movement_channel_2, movement_channel_1)]
    print('Check for overlapping movement artifacts')
    putative_ripples_df['Overlaps_with_movement'] = check_overlap(putative_ripples_df, overlapping_artifacts)
    
    print('Filtering the putative ripples...')
    #filtered_df = putative_ripples_df[(putative_ripples_df['Overlaps_with_gamma'] == False) & (putative_ripples_df['Overlaps_with_movement'] == False)]
    filtered_df = putative_ripples_df
    # this line filtered by max lfp ampiplitude which is pointless, should be removed
    #filtered_df = filtered_df[filtered_df.Peak_Amplitude_lfpzscore > lfp_amplidude_threshold]
    print('Writing filtered events to file')
    csv_filename = f"session_{session_id}_probe_{probe_id}_filtered_swrs.csv"
    csv_path = os.path.join(session_subfolder, csv_filename)