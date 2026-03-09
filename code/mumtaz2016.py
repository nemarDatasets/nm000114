README_CONTENT = """## Introduction

This dataset contains resting-state and task-based EEG recordings from patients diagnosed with Major Depressive Disorder (MDD) and healthy control participants (H). The data was collected to investigate differences in brain electrical activity between MDD patients and healthy individuals across different mental states. The dataset includes 34 participants (19 healthy controls and 15 MDD patients) with recordings during eyes-closed rest, eyes-open rest, and an auditory oddball P300 task. This dataset enables research on neurophysiological biomarkers of depression, comparative studies of brain activity patterns between clinical and healthy populations, and investigation of attentional processing differences in MDD.

## Overview of the experiment

Participants underwent three recording conditions: (1) eyes-closed resting state, (2) eyes-open resting state, and (3) an auditory oddball P300 task. During the resting-state conditions, participants were instructed to sit quietly with either their eyes closed (EC) or eyes open (EO) for the duration of the recording. In the P300 task, participants were presented with auditory stimuli consisting of frequent standard tones (80% probability) and infrequent target tones (20% probability), and were required to mentally count the target tones. EEG was recorded using a 19-channel monopolar EEG system with electrodes positioned according to the International 10-20 system, referenced to linked ears (A1+A2). The sampling rate was 256 Hz. Hardware filters included a high-pass filter at 0.5 Hz and a low-pass filter at 70 Hz, with a 50 Hz notch filter to remove power line noise. All electrode impedances were maintained below 5 kΩ. The recordings were conducted in a controlled environment to minimize external artifacts. One participant (MDD S15) had two separate recording sessions, resulting in duplicate recordings for this subject.

## Description of the preprocessing if any

The original EDF files have been converted to BIDS format. Channel names have been standardized by extracting the electrode names from the original "EEG <electrode>-<reference>" format. Channels originally referenced to the left ear (LE) are now labeled with just the electrode name, while other bipolar derivations (e.g., A2-A1, 23A-23R, 24A-24R) retain their bipolar notation in the format "<electrode1>-<electrode2>". The dataset includes 19 standard EEG channels plus three additional bipolar channels. Subject IDs have been prefixed with their diagnostic group ("H" for healthy controls, "MDD" for Major Depressive Disorder patients) to facilitate group comparisons. All recordings were artifact-free segments selected from longer recording sessions, with epochs containing excessive oculographic or myographic artifacts excluded during initial data collection.

## Description of the event values if any

No events.tsv files are provided as the recordings represent continuous resting-state or task conditions without discrete trial markers. The experimental condition for each recording is indicated by the "task" field in the BIDS filename:
- "eyesClosed": eyes-closed resting state
- "eyesOpen": eyes-open resting state  
- "P300": auditory oddball task (continuous recording during the entire task block)

For the P300 task recordings, while individual stimulus onsets are not marked in events.tsv files, the entire recording represents the period during which participants performed the auditory oddball counting task.

## Citation

When using this dataset, please cite:

1. Mumtaz, W., Xia, L., Ali, S. S. A., Yasin, M. A. M., Hussain, M., & Malik, A. S. (2017). Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD). Biomedical Signal Processing and Control, 31, 108-115. https://doi.org/10.1016/j.bspc.2016.07.006

2. Mumtaz, Wajid (2016). MDD Patients and Healthy Controls EEG Data (New). figshare. Dataset. https://doi.org/10.6084/m9.figshare.4244171.v2

**Data curators:**
Pierre Guetschel (BIDS conversion)

Original data collection team:
- Wajid Mumtaz (Universiti Teknologi PETRONAS)
- Likun Xia (Universiti Teknologi PETRONAS)
- Syed Saad Azhar Ali (Universiti Teknologi PETRONAS)
- Mohd Azhar Mohd Yasin (Universiti Teknologi PETRONAS)
- Mazhar Hussain (Universiti Teknologi PETRONAS)
- Aamir Saeed Malik (Universiti Teknologi PETRONAS)
"""

DATASET_NAME = "MDD Patients and Healthy Controls EEG Data"

from pathlib import Path
import re
import shutil

import pandas as pd
from mne.io import read_raw_edf
from mne_bids import BIDSPath, write_raw_bids, make_dataset_description, make_report

pathology_map = {
    "H": "healthy",
    "MDD": "major depressive disorder",
}
task_map = {
    "EC": "eyesClosed",
    "EO": "eyesOpen",
    "TASK": "P300",
}
CH_NAME_REGEX = r"EEG (?P<ch_name>[a-zA-Z0-9]+)-(?P<ref_name>[a-zA-Z0-9]+)"


def _get_records(source_root: Path):

    # example record: subject 1, eyes closed, Healthy Control
    # "H S1 EC.edf"
    record_regex = r"(?P<record_id>[0-9]*)_?(?P<pathology>[A-Z]+) (?P<subject>S[0-9]+) +(?P<task>[A-Z]+)\.edf"

    for file in source_root.glob("*.edf"):
        if file.name.startswith("._"):
            continue  # skip MacOS hidden files
        match = re.match(record_regex, file.name)
        assert match is not None, f"Record {file} does not match expected format"

        subject = match.group("subject")
        pathology = match.group("pathology")
        task = match.group("task")
        record_id = match.group("record_id")
        assert pathology in pathology_map, pathology
        assert task in task_map, task

        run = None
        if record_id:
            assert subject == "S15"
            if record_id == "6921143":
                run = 1
            elif record_id == "6921959":
                run = 2
            else:
                raise ValueError(f"Unexpected record id {record_id}")

        # source_path = source_root / record
        bids_path = BIDSPath(
            subject=f"{pathology}{subject}",
            task=task_map[task],
            run=run,
            suffix="eeg",
            datatype="eeg",
            extension=".edf",
        )
        yield file, bids_path, pathology_map[pathology]


def main(
    source_root: Path,
    bids_root: Path,
    overwrite: bool = False,
    finalize_only: bool = False,
):
    """Convert the MDD dataset to BIDS format.

    Parameters
    ----------
    source_root : Path
        Path to the root folder of the MDD dataset.
    bids_root : Path
        Path to the root of the BIDS dataset to create.
    overwrite : bool
        If True, overwrite existing BIDS files.
    """
    source_root = Path(source_root).expanduser()
    bids_root = Path(bids_root).expanduser()

    if finalize_only:
        _finalize_dataset(bids_root, overwrite=overwrite)
        return

    records = list(_get_records(source_root))

    # Add bids root:
    bids_root.mkdir(parents=True, exist_ok=True)
    for _, bids_path, _ in records:
        bids_path = bids_path.update(root=bids_root)

    # sanity check: no duplicate bids paths
    bids_paths = [bids_path.fpath for _, bids_path, _ in records]
    assert len(bids_paths) == len(set(bids_paths)), "Duplicate BIDS paths found"

    for source_path, bids_path, pathology in records:
        raw = read_raw_edf(source_path, preload=False, verbose=False)
        raw.info["subject_info"] = {"his_id": bids_path.subject}
        raw.info["description"] = pathology

        # set proper channel names
        ch_names_map = {}
        for ch in raw.info["chs"]:
            match = re.match(CH_NAME_REGEX, ch["ch_name"])
            assert match is not None
            ch_name = match.group("ch_name")
            ref_name = match.group("ref_name")
            ch_names_map[ch["ch_name"]] = (
                ch_name if ref_name == "LE" else f"{ch_name}-{ref_name}"
            )
        raw.rename_channels(ch_names_map)

        write_raw_bids(
            raw,
            bids_path,
            overwrite=overwrite,
            verbose=False,
        )

    _finalize_dataset(bids_root, overwrite=overwrite)


def _finalize_dataset(bids_root: Path, overwrite: bool = False):
    # save script
    script_path = Path(__file__)
    script_dest = bids_root / "code" / script_path.name
    script_dest.parent.mkdir(exist_ok=True)
    shutil.copy2(script_path, script_dest)
    description_file = bids_root / "dataset_description.json"
    if description_file.exists() and overwrite:
        description_file.unlink()
    make_dataset_description(
        path=bids_root,
        name=DATASET_NAME,
        dataset_type="derivative",
        source_datasets=[
            {"DOI": "https://doi.org/10.6084/m9.figshare.4244171.v2"},
        ],
        authors=[
            "Wajid Mumtaz",
            "Likun Xia",
            "Syed Saad Azhar Ali",
            "Mohd Azhar Mohd Yasin",
            "Mazhar Hussain",
            "Aamir Saeed Malik",
        ],
        acknowledgements="Pierre Guetschel updated the data to BIDS format.",
        overwrite=overwrite,
        data_license="CC-BY-4.0",
    )
    # cleanup macos hidden files
    for macos_file in bids_root.rglob("._*"):
        macos_file.unlink()

    report_str = make_report(bids_root)
    print(report_str)

    # overwrite README
    readme_path = bids_root / "README.md"
    with open(readme_path, "w") as f:
        f.write(
            f"# {DATASET_NAME}\n\n{README_CONTENT}\n\n---\n\n"
            f"## Automatic report\n\n*Report automatically generated by `mne_bids.make_report()`.*\n\n> {report_str}"
        )

    # Remove participants.json if it exists
    participants_json = bids_root / "participants.json"
    if participants_json.exists():
        participants_json.unlink()
        print(f"Removed {participants_json}")

    # Clean up participants.tsv by removing columns where all values are "n/a"
    participants_tsv = bids_root / "participants.tsv"
    if participants_tsv.exists():
        df = pd.read_csv(participants_tsv, sep="\t")
        # Find columns where all non-participant_id values are "n/a"
        cols_to_drop = []
        for col in df.columns:
            if col != "participant_id" and (df[col] == "n/a").all():
                cols_to_drop.append(col)
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            df.to_csv(participants_tsv, sep="\t", index=False)
            print(
                f"Removed columns with all 'n/a' values from {participants_tsv}: {cols_to_drop}"
            )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)
    # python bids_maker/datasets/mumtaz2016.py --source_root ~/data/mdd_mumtaz2016/ --bids_root ~/data/bids/mdd_mumtaz2016/ --overwrite
