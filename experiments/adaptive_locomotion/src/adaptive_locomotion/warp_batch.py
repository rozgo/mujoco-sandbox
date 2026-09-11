"""Optional CUDA physics bridge for the existing NumPy locomotion environment.

Only initialization/reset/explicit forward uploads poses. Live stepping uploads
controls and dirty actuator parameters, then advances MuJoCo Warp on the GPU.
Host mirrors are intentional: rewards/reference matching still run on CPU.
"""

import functools
import time

import mujoco
import numpy as np


def configure_ccd_module_width(wp, mjw):
    """Version-scoped fix for occupancy/launch variant identity in Warp 1.17.

    The upstream CCD builder specializes launch_bounds with block_dim but leaves
    its module default at 256. Occupancy queries then load the wrong variant.
    Set the public module option to the builder's existing width; no kernel body,
    geometry, contacts, iteration limits or physics parameters are modified.
    """
    if (mjw.__version__, wp.__version__) != ("3.13.0", "1.17.0"):
        raise RuntimeError(
            "Review the CCD compatibility hook before changing Warp versions"
        )
    from mujoco_warp._src import collision_convex

    builder = collision_convex.ccd_kernel_builder
    if getattr(builder, "_adaptive_width_configured", False):
        return

    @functools.wraps(builder)
    def configured(*args, **kwargs):
        kernel = builder(*args, **kwargs)
        width = kwargs["block_dim"] if "block_dim" in kwargs else args[6]
        if wp.get_module_options(kernel.module)["block_dim"] != width:
            wp.set_module_options({"block_dim": width}, kernel.module)
        return kernel

    configured._adaptive_width_configured = True
    collision_convex.ccd_kernel_builder = configured


def configure_sensor_bvh(mjw):
    """Route this bridge's range sensors through MJWarp's public BVH ray API."""
    from mujoco_warp._src import ray

    original = ray.rays
    if getattr(original, "_adaptive_bvh_configured", False):
        return

    @functools.wraps(original)
    def accelerated(m, d, *args, **kwargs):
        context = getattr(m, "_adaptive_sensor_bvh", None)
        if context is not None and "rc" not in kwargs and len(args) == 8:
            mjw.refit_bvh(m, d, context)
            kwargs["rc"] = context
        return original(m, d, *args, **kwargs)

    accelerated._adaptive_bvh_configured = True
    ray.rays = accelerated


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
        configure_ccd_module_width(wp, mjw)
        configure_sensor_bvh(mjw)
        self.wp, self.mjw, self.device = wp, mjw, wp.get_device(device)
        self.model, self.n = model, n
        self.host, self.buffers, self.graphs = {}, {}, {}
        self.compile_seconds = 0.0
        self.model_dirty = True
        self.stream = wp.Stream(self.device)
        with wp.ScopedDevice(self.device):
            self.m = mjw.put_model(model, batch_sizes={k: n for k in self.MODEL_FIELDS})
            self.d = mjw.make_data(model, nworld=n, nconmax=nconmax, njmax=njmax)
            if self.m.nrangefinder:
                # Build all mesh/scene BVHs without allocating camera images.
                # Keep every geom group and the public ray API's culling rules.
                self.m._adaptive_sensor_bvh = mjw.create_render_context(
                    model,
                    nworld=n,
                    cam_active=[],
                    enabled_geom_groups=list(range(6)),
                    use_textures=False,
                    use_fast_math=False,
                )
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
            array = self.d.overflow
            self.buffers["overflow"] = wp.empty(
                array.shape, dtype=array.dtype, device="cpu", pinned=True
            )
            self.host["overflow"] = self.buffers["overflow"].numpy()
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
            self.enqueue_download()
            wp.synchronize_device(self.device)
        self.check_download()
        if previous is not None:
            for key, values in previous.items():
                self.host[key][keep] = values

    def enqueue_download(self):
        for key in (*self.DATA_FIELDS, "overflow"):
            self.wp.copy(self.buffers[key], getattr(self.d, key))

    def check_download(self):
        overflow = self.host["overflow"]
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

    def step_async(self, nstep=1):
        """Queue this independent topology without blocking the other batches."""
        with self.wp.ScopedStream(self.stream):
            self.upload_controls()
            self.step_device(nstep)
            self.enqueue_download()

    def wait_step(self):
        self.wp.synchronize_stream(self.stream)
        self.check_download()

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
