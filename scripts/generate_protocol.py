"""Generate a corrected multi-muscle NeuroMotion recording from an explicit YAML protocol."""

import argparse
import gc
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import yaml

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT.parent / "BioMime")]

from BioMime.models.generator import Generator
from BioMime.utils.basics import load_generator, setup_seed, update_config
from NeuroMotion.MNPoollib.MNPool import MotoneuronPool
from NeuroMotion.MNPoollib.mn_params import NUM_MUS, mn_default_settings
from NeuroMotion.MNPoollib.mn_utils import generate_emg_mu, normalise_physical


def revision():
    return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()


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

    return protocol, selected, muscles, fs, trajectory("activation"), trajectory("length")


def dynamic_muaps(generator, pool, latent, length, steps, device, batch_size):
    properties = pool.get_properties()
    count = pool.get_num_mu()
    indexes = np.minimum((np.arange(steps) * len(length) / steps).astype(int), len(length) - 1)
    muaps = []
    for index in indexes:
        factor = length[index]
        condition = torch.from_numpy(np.column_stack((
            normalise_physical(properties["num"], "num"),
            normalise_physical(properties["depth"] / np.sqrt(factor), "depth"),
            normalise_physical(properties["angle"], "angle"),
            normalise_physical(properties["iz"], "iz"),
            normalise_physical(properties["cv"] / factor, "cv"),
            normalise_physical(properties["len"] * factor, "len"),
        ))).float()
        if device == "cuda":
            condition, latent = condition.cuda(), latent.cuda()
        batches = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                end = min(start + batch_size, count)
                sampled = generator.sample(end - start, condition[start:end], condition.device, latent[start:end])
                batches.append(sampled.permute(0, 2, 3, 1).cpu().numpy().astype(np.float32))
        muaps.append(np.concatenate(batches, axis=0))
    return np.transpose(np.asarray(muaps), (1, 0, 2, 3, 4))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--model-pth", required=True)
    parser.add_argument("--cfg", default="upstream/NeuroMotion/ckp/config.yaml")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--subject-seed", required=True, type=int)
    parser.add_argument("--trial-seed", required=True, type=int)
    parser.add_argument("--condition-rate-hz", type=float, default=5)
    parser.add_argument("--generator-batch-size", type=int, default=64)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    protocol, condition, muscles, fs, activation, length = load_condition(args.protocol, args.condition)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.condition_rate_hz <= 0 or args.generator_batch_size <= 0 or any(muscle not in NUM_MUS for muscle in muscles):
        raise ValueError("invalid condition rate or muscle without a NeuroMotion motor-unit pool")

    setup_seed(args.subject_seed)
    cfg = update_config(args.cfg)
    generator = load_generator(args.model_pth, Generator(cfg.Model.Generator), args.device)
    generator.eval()
    pools, latents = {}, {}
    for muscle in muscles:
        pool = MotoneuronPool(NUM_MUS[muscle], muscle, **mn_default_settings)
        pool.assign_properties(normalise=False)
        pool.init_twitches(fs)
        pool.init_quisistatic_ef_model()
        pools[muscle] = pool
        latents[muscle] = torch.randn(pool.get_num_mu(), cfg.Model.Generator.Latent)

    setup_seed(args.trial_seed)
    time_samples = len(next(iter(activation.values())))
    steps = max(1, round(time_samples * args.condition_rate_hz / fs))
    emg = np.zeros((10, 32, time_samples + 96), dtype=np.float32)
    for muscle in muscles:
        muaps = dynamic_muaps(generator, pools[muscle], latents[muscle], length[muscle], steps, args.device, args.generator_batch_size)
        spikes = pools[muscle].generate_spike_trains(activation[muscle])[1]
        for motor_unit, train in enumerate(spikes):
            emg += generate_emg_mu(muaps[motor_unit], train, time_samples).astype(np.float32)
        del muaps
        gc.collect()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, emg=emg[:, :, :time_samples], sampling_rate_hz=fs)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_suffix(".json").write_text(json.dumps({
        "generator": "corrected_multimuscle_protocol_v1",
        "NeuroMotion_SHA": revision(),
        "movement_name": args.condition,
        "protocol": str(Path(args.protocol)),
        "protocol_version": protocol.get("version"),
        "phase_labels": [phase["name"] for phase in condition["phases"]],
        "muscles": muscles,
        "source_sampling_rate_hz": fs,
        "condition_rate_hz": args.condition_rate_hz,
        "generator_batch_size": args.generator_batch_size,
        "tensor_axis_order": ["axial_row", "circumferential_column", "time"],
        "subject_seed": args.subject_seed,
        "trial_seed": args.trial_seed,
        "muap_latent_policy": "persistent_per_virtual_subject",
        "capture_SHA256": digest,
        "voltage_unit": "UNKNOWN",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
