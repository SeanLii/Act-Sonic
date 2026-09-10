# ACT-Sonic

This repository trains a classic ACT policy on the Unitree G1 SONIC dataset and
serves the result through the same ZMQ PolicyServer interface used by
Isaac-GR00T. The official `gear_sonic/scripts/run_vla_inference.py` client can
therefore use an ACT checkpoint without client-side changes.

## SONIC contract

The LeRobot dataset stores a concatenated 78-D action. It must not be sent to
SONIC as one raw tensor. `ACTSonicPolicy` inverse-normalizes and splits it into:

- `motion_token`: `float32[B, T, 64]`
- `left_hand_joints`: `float32[B, T, 7]`
- `right_hand_joints`: `float32[B, T, 7]`

The state fields are concatenated in the GR00T `unitree_g1_sonic` order to form
the ACT 46-D qpos input. The default chunk/action horizon is 40, matching the
official SONIC inference client.

Install the runtime and training dependencies first:

```bash
python -m pip install -r requirements.txt
```

## Train

Run from this directory:

```bash
python train.py \
  --dataset-dir /path/to/io_g1_box_cotransport2_sonic_joint \
  --output-dir outputs/act_sonic \
  --chunk-size 40
```

Checkpoints contain the model configuration, mean/std normalization values, and
the SONIC field contract. Existing checkpoints created before the contract field
was added remain loadable as long as they contain `model_config` and
`normalization`.

## Serve to SONIC

The included server implements the small ZMQ/msgpack/NumPy subset of GR00T's
PolicyServer used by the official `PolicyClient`; it does not load the GR00T
model or Transformers stack:

```bash
python serve.py \
  --checkpoint outputs/act_sonic/best.pt \
  --device cuda \
  --port 5550
```

Then run the official client from GR00T-WholeBodyControl:

```bash
python gear_sonic/scripts/run_vla_inference.py \
  --host <act-server-ip> \
  --port 5550 \
  --embodiment-tag unitree_g1_sonic \
  --action-horizon 40 \
  --prompt "pick up the cup"
```

The checkpoint's `chunk_size` must equal the client's `--action-horizon`.
The ACT architecture in this repository does not consume language, so changing
`--prompt` does not condition its predictions.

## Test

```bash
pytest -q
```
