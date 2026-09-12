"""Convert official teacher weights; this isolated process never trains our actor.

Modern TFP renamed its saved TypeSpec registry keys. Register only the two old
keys used by the official 2024 SavedModel, then check the restored function itself.
No TensorFlow/TFP dependency is needed in the embodied-fly runtime.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.core.protobuf import saved_model_pb2
from tensorflow.python.framework import type_spec_registry


def export(source: Path, output: Path):
    for name in ("Normal", "Independent"):
        getattr(tfp.distributions, name)  # Materialize lazy distribution modules.
        old = f"tensorflow_probability.python.distributions.{name.lower()}.{name}_ACTTypeSpec"
        type_spec_registry._NAME_TO_TYPE_SPEC[old] = type_spec_registry.lookup(
            f"tfp.distributions.{name}_ACTTypeSpec"
        )
    policy = tf.saved_model.load(str(source))
    function = policy.__call__.concrete_functions[0]
    specification = function.structured_input_signature[0][0]
    rng = np.random.default_rng(44001)
    inputs = {
        key: rng.normal(size=(32, *spec.shape[1:])).astype(np.float32)
        for key, spec in sorted(specification.items())
    }
    result = policy({key: tf.constant(value) for key, value in inputs.items()})
    proto = saved_model_pb2.SavedModel()
    proto.ParseFromString((source / "saved_model.pb").read_bytes())
    epsilons = []
    for f in proto.meta_graphs[0].graph_def.library.function:
        for node in f.node_def:
            if node.name.endswith("layer_norm/batchnorm/add/y"):
                epsilons.append(float(tf.make_ndarray(node.attr["value"].tensor)))
    if not epsilons or len(set(epsilons)) != 1:
        raise ValueError("Unrecognized teacher LayerNorm epsilon")
    arrays = {f"v{i}": v.numpy() for i, v in enumerate(policy._variables)}
    arrays.update(
        golden_input=np.concatenate([v.reshape(32, -1) for v in inputs.values()], 1),
        golden_mean=result.mean().numpy(),
        golden_std=result.stddev().numpy(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **arrays)
    manifest = {
        "source": f"Official FlyBody trained-fly-policies.zip / {source.name}",
        "saved_model_sha256": hashlib.sha256(
            (source / "saved_model.pb").read_bytes()
        ).hexdigest(),
        "export_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "tensorflow": tf.__version__,
        "tensorflow_probability": tfp.__version__,
        "layer_norm_epsilon": epsilons[0],
        "observation_shapes": {k: list(v.shape[1:]) for k, v in inputs.items()},
        "variables": [{"name": v.name, "shape": list(v.shape)} for v in policy._variables],
        "golden_samples": 32,
        "golden_seed": 44001,
        "purpose": "Training-only learned teacher; not part of deployed MaleCNS actor",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {k: manifest[k] for k in ("export_sha256", "layer_norm_epsilon", "golden_samples")}
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.source, args.output)
