# previous_version_channels.py
"""
Reads the channel choices made by a previous version of the pipeline, so a new
run can reuse them instead of choosing again.

Control channels are picked at random, so the only way to get the same ones in a
new version is to read them back from the previous version's output. The ripple
and sharp wave channels can be reused the same way.

Where each choice is read from, in the previous version's session folder
(<previous dataset dir>/swrs_session_<session_id>/):
  - ripple  : probe_<id>_channel_<chan>_putative_swr_events.csv.gz (file name),
              else ripple_band.selected_channel_id in the channel selection metadata
  - control : probe_<id>_channel_<chan>_movement_artifacts.csv.gz (two file names),
              else control_channels.selected_channel_ids in the channel selection metadata
  - sw      : sharp_wave_band.selected_channel_id in
              probe_<id>_channel_selection_metadata.json.gz (the only place it is recorded)
File names are preferred because they name the channels detection was actually run on.
"""
import os
import re
import gzip
import json

# The kinds of channel choice that can be kept from a previous version
CHANNEL_KINDS = ('control', 'ripple', 'sw')


def parse_keep_list(value):
    """
    Turn 'control,ripple' (or a list) into a validated list of channel kinds.
    'all' keeps every kind, 'none' or empty keeps nothing.
    """
    if value is None:
        return []
    entries = value if isinstance(value, (list, tuple)) else str(value).split(',')
    entries = [str(entry).strip().lower() for entry in entries if str(entry).strip()]
    if 'none' in entries:
        return []
    if 'all' in entries:
        return list(CHANNEL_KINDS)
    unknown = [entry for entry in entries if entry not in CHANNEL_KINDS]
    if unknown:
        raise ValueError(f"Unknown channel kind(s) to keep: {unknown}. Valid: {list(CHANNEL_KINDS)}, 'all' or 'none'.")
    return [kind for kind in CHANNEL_KINDS if kind in entries]


def resolve_previous_dataset_dir(previous_dir, swr_output_dir_name):
    """
    Find the previous version's dataset folder (the one holding swrs_session_* folders).
    previous_dir may be that folder itself, or a folder above it such as the previous
    run's OUTPUT_DIR or an unzipped download. In that case the folder named after the
    dataset's swr_output_dir is searched for, up to max_depth levels down.
    """
    max_depth = 4
    if not os.path.isdir(previous_dir):
        raise FileNotFoundError(f"Previous version directory does not exist: {previous_dir}")
    if any(name.startswith('swrs_session_') for name in os.listdir(previous_dir)):
        return previous_dir
    previous_dir = previous_dir.rstrip(os.sep)
    for root, dirs, _ in os.walk(previous_dir):
        dirs.sort()
        if swr_output_dir_name in dirs:
            return os.path.join(root, swr_output_dir_name)
        if root[len(previous_dir):].count(os.sep) >= max_depth - 1:
            dirs[:] = []
        else:
            dirs[:] = [name for name in dirs if not name.startswith('swrs_session_')]
    raise FileNotFoundError(
        f"No swrs_session_* folders in {previous_dir} and no dataset folder '{swr_output_dir_name}' within {max_depth} levels below it"
    )


def _channel_number(text):
    """Channel id from a file name fragment, e.g. '850245983' or 'channelsrawInd_123'."""
    match = re.search(r'(\d+)$', str(text))
    return int(match.group(1)) if match else None


class PreviousVersionChannels:
    """
    Channel choices of one session in a previous version of the pipeline.

    Parameters
    ----------
    previous_dataset_dir : str
        Previous version's dataset folder (holds swrs_session_* folders).
    session_id : str or int
    keep : list of str
        Which kinds of channel choice to reuse, from CHANNEL_KINDS.
    strict : bool
        If True, a kept choice that cannot be reused raises an error instead of
        falling back to a fresh choice.
    """
    def __init__(self, previous_dataset_dir, session_id, keep, strict=False):
        self.previous_dataset_dir = previous_dataset_dir
        self.session_folder = os.path.join(previous_dataset_dir, f"swrs_session_{session_id}")
        self.keep = parse_keep_list(keep)
        self.strict = strict
        self._choices = None  # {probe_id: {'ripple': int, 'sw': int, 'control': [int, int]}}

    @classmethod
    def from_config(cls, config, session_id):
        """Build from the pipeline config, or return None if no previous version is in use."""
        settings = config.get('previous_version') or {}
        if not settings.get('dir') or not settings.get('keep'):
            return None
        return cls(settings['dir'], session_id, settings['keep'], settings.get('strict', False))

    def _load(self):
        """Read every probe's channel choices from the previous session folder, once."""
        choices = {}
        if not os.path.isdir(self.session_folder):
            return choices

        from_file_names = {}
        from_metadata = {}
        for name in sorted(os.listdir(self.session_folder)):
            match = re.match(r'probe_(.+?)_channel_(.+)_putative_swr_events\.csv\.gz$', name)
            if match:
                from_file_names.setdefault(match.group(1), {})['ripple'] = _channel_number(match.group(2))
                continue
            match = re.match(r'probe_(.+?)_channel_(.+)_movement_artifacts\.csv\.gz$', name)
            if match:
                from_file_names.setdefault(match.group(1), {}).setdefault('control', []).append(_channel_number(match.group(2)))
                continue
            match = re.match(r'probe_(.+)_channel_selection_metadata\.json\.gz$', name)
            if match:
                try:
                    with gzip.open(os.path.join(self.session_folder, name), 'rt', encoding='utf-8') as f:
                        metadata = json.load(f)
                except (OSError, EOFError, ValueError):
                    continue  # unreadable metadata is treated as missing
                from_metadata[match.group(1)] = {
                    'ripple': (metadata.get('ripple_band') or {}).get('selected_channel_id'),
                    'sw': (metadata.get('sharp_wave_band') or {}).get('selected_channel_id'),
                    'control': (metadata.get('control_channels') or {}).get('selected_channel_ids'),
                }

        for probe_id in set(from_file_names) | set(from_metadata):
            names = from_file_names.get(probe_id, {})
            metadata = from_metadata.get(probe_id, {})
            ripple = names.get('ripple')
            if ripple is None:
                ripple = metadata.get('ripple')
            control = names.get('control') or metadata.get('control')
            # Exactly two distinct control channels are needed, anything else counts as missing
            if control is None or None in control or len(set(control)) != 2:
                control = None
            sw = metadata.get('sw')
            choices[probe_id] = {
                'ripple': int(ripple) if ripple is not None else None,
                'sw': int(sw) if sw is not None else None,
                'control': sorted(int(chan) for chan in control) if control else None,
            }
        return choices

    def get(self, probe_id, kind):
        """
        Previous choice for this probe: an int for 'ripple' and 'sw', a list of two
        ints for 'control', or None if the previous version has no record of it.
        """
        if kind not in CHANNEL_KINDS:
            raise ValueError(f"Unknown channel kind '{kind}'. Valid: {list(CHANNEL_KINDS)}")
        if self._choices is None:
            self._choices = self._load()
        return self._choices.get(str(probe_id), {}).get(kind)
