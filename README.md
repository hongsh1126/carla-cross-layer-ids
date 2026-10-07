# Cross-layer intrusion detection in CARLA (CAN + sensor attacks)

Research code for a cross-layer intrusion-detection study that combines in-vehicle CAN-bus traffic with perception-level signals. **This is work in progress; the results are preliminary and the associated manuscript is unpublished.**

## What is here
- `carla_cl_ids.py` - generator that drives a CARLA 0.9.16 scenario and writes a synthetic CAN stream plus perception/sensor logs, with four scenario families: normal, CAN-only attacks (DoS, fuzzy, plain and rate-preserving stealth spoofing), sensor-only attacks (GPS drift, LiDAR and camera phantom or hide), and combined CAN plus sensor attacks. The CAN bridge is one-way and nothing is sent to a real bus
- `realistic_can.py` - a more realistic CAN bus model (fixed IDs, ECU clock skew and jitter, event frames, 500 kbit/s arbitration with bit stuffing)
- `run_collection.py`, `run_supplement.py` - batch drivers for the main and supplementary runs
- `analyze_*.py`, `make_bus_tables.py` - window-level features and Random-Forest baselines (CAN-only vs fused)
- `road_*.py` - validation scripts for the real-vehicle [ROAD dataset](https://zenodo.org/records/10462796) (the dataset is **not** included; download it from Zenodo)
- `results/` - aggregate result files from the analysis scripts

## Requirements
CARLA 0.9.16 (server), Python with `carla`, `numpy`, `pandas`, `scikit-learn`, `pillow`, `matplotlib`.

## Known limitations
- The CAN traffic is generated in simulation, not recorded from a real vehicle (the ROAD scripts are the real-data check).
- The CAN bridge is one-way, and some feature definitions were changed after seeing early results; this is disclosed in the draft manuscript.
- Raw logs and captures are not included because of size.

## License
MIT (see `LICENSE`). CARLA and the ROAD dataset have their own licenses.
