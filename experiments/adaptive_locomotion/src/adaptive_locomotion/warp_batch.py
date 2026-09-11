"""Optional CUDA physics bridge for the existing NumPy locomotion environment.

Only initialization/reset/explicit forward uploads poses. Live stepping uploads
controls and dirty actuator parameters, then advances MuJoCo Warp on the GPU.
Host mirrors are intentional: rewards/reference matching still run on CPU.
"""

import time

import mujoco
import numpy as np


class WarpBatch:
    MODEL_FIELDS = ("actuator_gainprm", "actuator_biasprm", "actuator_forcerange")
    DATA_FIELDS = (
        "qpos",
        "qvel",
        "ctrl",
        "actuator_force",
        "sensordata",
        "geom_xpos",
        "time",
    )

    def __init__(self, model, n, device="cuda:0", nconmax=64, njmax=256):
        try:
            import mujoco_warp as mjw
            import warp as wp
        except ImportError as error:
            raise RuntimeError(
                "Install the optional stack with uv sync --extra warp"
            ) from error
        wp.init()
        if not wp.is_cuda_available() or not wp.get_device(device).is_cuda:
            raise RuntimeError(
                "Warp physics requires an NVIDIA CUDA device; use mjbatch on Mac"
            )
        self.wp, self.mjw, self.device = wp, mjw, wp.get_device(device)
        self.model, self.n = model, n
        self.host, self.buffers, self.graphs = {}, {}, {}
        self.compile_seconds = 0.0
        self.model_dirty = True
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model, batch_sizes={k: n for k in self.MODEL_FIELDS})
            # Warp 1.17 occupancy lookup loads the default 256-thread variant;
            # switching the same CCD kernel to 64 threads then looking it up
            # again can leave a stale symbol hash. Keep both launch widths equal.
            # This changes GPU scheduling only, not collision geometry/solver.
            self.m.block_dim.convex_ccd = 256
            self.d = mjw.make_data(model, nworld=n, nconmax=nconmax, njmax=njmax)
            for key in self.DATA_FIELDS:
                array = getattr(self.d, key)
                buffer = wp.empty(
                    array.shape, dtype=array.dtype, device="cpu", pinned=True
                )
                self.buffers[key], self.host[key] = buffer, buffer.numpy()
            for key in self.MODEL_FIELDS:
                array = getattr(self.m, key)
                buffer = wp.empty(
                    array.shape, dtype=array.dtype, device="cpu", pinned=True
                )
                self.buffers[key], self.host[key] = buffer, buffer.numpy()
                self.host[key][:] = np.asarray(getattr(model, key))
            self.reset_mask = wp.zeros(n, dtype=bool)
            # Capture reset once; the mask contents vary without recapturing.
            started = time.perf_counter()
            with wp.ScopedCapture() as capture:
                mjw.reset_data(self.m, self.d, self.reset_mask)
            self.reset_graph = capture.graph
            self.compile_seconds += time.perf_counter() - started
        self.host["warning"] = np.zeros((n, mujoco.mjtWarning.mjNWARNING, 2), dtype=int)
        self.download()

    def bind(self, name):
        return self.host[name]

    def expand(self, name):
        return self.host[name]

    def sensor(self, name):
        sensor = self.model.sensor(name)
        address, size = int(sensor.adr[0]), int(sensor.dim[0])
        return self.host["sensordata"][:, address : address + size]

    def mark_model_dirty(self):
        self.model_dirty = True

    def upload_controls(self):
        wp = self.wp
        with wp.ScopedDevice(self.device):
            wp.copy(self.d.ctrl, self.buffers["ctrl"])
            if self.model_dirty:
                for key in self.MODEL_FIELDS:
                    wp.copy(getattr(self.m, key), self.buffers[key])
                self.model_dirty = False

    def graph(self, nstep):
        if nstep not in self.graphs:
            started = time.perf_counter()
            with self.wp.ScopedDevice(self.device), self.wp.ScopedCapture() as capture:
                for _ in range(nstep):
                    self.mjw.step(self.m, self.d)
            self.graphs[nstep] = capture.graph
            self.compile_seconds += time.perf_counter() - started
        return self.graphs[nstep]

    def step_device(self, nstep=1):
        """GPU-only launch for an explicitly labeled physics throughput benchmark."""
        with self.wp.ScopedDevice(self.device):
            self.wp.capture_launch(self.graph(nstep))

    def download(self, ids=None):
        wp = self.wp
        previous = None
        if ids is not None:
            # Recomputing derived fields for reset worlds must not change the
            # host observation timing of other worlds in the same topology.
            keep = np.ones(self.n, dtype=bool)
            keep[ids] = False
            previous = {k: self.host[k][keep].copy() for k in self.DATA_FIELDS}
        with wp.ScopedDevice(self.device):
            for key in self.DATA_FIELDS:
                wp.copy(self.buffers[key], getattr(self.d, key))
            wp.synchronize_device(self.device)
            overflow = self.d.overflow.numpy()
        if overflow.any():
            raise FloatingPointError(
                f"MuJoCo Warp capacity overflow: {np.unique(overflow).tolist()}"
            )
        if any(not np.isfinite(self.host[k]).all() for k in self.DATA_FIELDS):
            raise FloatingPointError("Nonfinite MuJoCo Warp state/sensor data")
        if np.max(np.abs(self.host["qvel"])) > 1e10:
            raise FloatingPointError(
                "MuJoCo Warp velocity exceeds MuJoCo numerical envelope"
            )
        if previous is not None:
            for key, values in previous.items():
                self.host[key][keep] = values

    def step(self, nstep=1):
        self.upload_controls()
        self.step_device(nstep)
        self.download()

    def reset(self, ids=None):
        mask = np.zeros(self.n, dtype=bool)
        mask[:] = ids is None
        if ids is not None:
            mask[ids] = True
        with self.wp.ScopedDevice(self.device):
            self.reset_mask.assign(mask)
            self.wp.capture_launch(self.reset_graph)
        self.download(ids)

    def forward(self, ids=None):
        # Called only after explicit initialization/reset/state assignment.
        with self.wp.ScopedDevice(self.device):
            if ids is None:
                for key in ("qpos", "qvel"):
                    self.wp.copy(getattr(self.d, key), self.buffers[key])
            else:
                # Merge only reset rows so live device state stays authoritative.
                for key in ("qpos", "qvel"):
                    values = getattr(self.d, key).numpy()
                    values[ids] = self.host[key][ids]
                    getattr(self.d, key).assign(values)
            self.upload_controls()
            if "forward" not in self.graphs:
                started = time.perf_counter()
                with self.wp.ScopedCapture() as capture:
                    self.mjw.forward(self.m, self.d)
                self.graphs["forward"] = capture.graph
                self.compile_seconds += time.perf_counter() - started
            self.wp.capture_launch(self.graphs["forward"])
        self.download(ids)
