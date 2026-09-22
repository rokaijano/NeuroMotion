"""Generate a corrected multi-muscle NeuroMotion recording from an explicit YAML protocol."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parents[1] / "BioMime")]

from BioMime.models.generator import Generator
from BioMime.utils.basics import load_generator, setup_seed, update_config
from NeuroMotion.EMGSyn import EMGSynthesiser
from NeuroMotion.MNPoollib.MNPoolStatus import MotoneuronPoolStatus
from NeuroMotion.MNPoollib.mn_params import NUM_MUS, mn_default_settings


def load_condition(path, condition):
    protocol = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    selected = protocol["conditions"][condition]
    muscles = tuple(protocol["muscles"])
    fs = int(protocol.get("sampling_rate_hz", 2048))
    phases = selected["phases"]
    for phase in phases:
        if set(phase["activation"]) != set(muscles) or set(phase["length"]) != set(muscles):
            raise ValueError("every phase must specify activation and length for every configured muscle")
    counts = [round(float(phase["duration_s"]) * fs) for phase in phases]
    if not all(counts):
        raise ValueError("each phase must contain at least one sample")
    def trajectory(key):
        values = {}
        for muscle in muscles:
            previous = phases[0][key][muscle]
            segments = []
            for phase, count in zip(phases, counts):
                target = phase[key][muscle]
                segments.append(np.linspace(previous, target, count, endpoint=False))
                previous = target
            values[muscle] = np.concatenate(segments)
        return values

    activation = trajectory("activation")
    length = trajectory("length")
    return protocol, selected, muscles, fs, activation, length


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--cfg", default="upstream/NeuroMotion/ckp/config.yaml")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--subject-seed", required=True, type=int)
    parser.add_argument("--trial-seed", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol, condition, muscles, fs, activation, length = load_condition(args.protocol, args.condition)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if any(muscle not in NUM_MUS for muscle in muscles):
        raise ValueError("protocol names a muscle without a NeuroMotion motor-unit pool")

    setup_seed(args.subject_seed)
    cfg = update_config(args.cfg)
    generator = load_generator(args.model_pth, Generator(cfg.Model.Generator), args.device)
    generator.eval()
    pools = {muscle: {"N": NUM_MUS[muscle], "ms_name": muscle, **mn_default_settings} for muscle in muscles}
    synthesiser = EMGSynthesiser(MotoneuronPoolStatus, pools, generator, cfg, fs=fs, device=args.device)

    setup_seed(args.trial_seed)
    sample_count = len(next(iter(activation.values())))
    emg = np.stack([synthesiser.update_emg({muscle: length[muscle][sample] for muscle in muscles}, {muscle: activation[muscle][sample] for muscle in muscles}) for sample in range(sample_count)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, emg=np.moveaxis(emg, 0, -1), sampling_rate_hz=fs)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps({
        "generator": "corrected_multimuscle_protocol_v1",
        "movement_name": args.condition,
        "protocol": str(Path(args.protocol)),
        "protocol_version": protocol.get("version"),
        "phase_labels": [phase["name"] for phase in condition["phases"]],
        "muscles": muscles,
        "source_sampling_rate_hz": fs,
        "tensor_axis_order": ["axial_row", "circumferential_column", "time"],
        "subject_seed": args.subject_seed,
        "trial_seed": args.trial_seed,
        "muap_latent_policy": "persistent_per_virtual_subject",
        "capture_SHA256": digest,
        "voltage_unit": "UNKNOWN",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
