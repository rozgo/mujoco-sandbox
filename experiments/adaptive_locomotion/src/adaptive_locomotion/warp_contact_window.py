"""Optional on-device contact window; no rigid-body state is modified."""

import warp as wp


@wp.kernel
def accumulate(
    sensors: wp.array2d(dtype=wp.float32),
    peaks: wp.array2d(dtype=wp.float32),
    impulses: wp.array2d(dtype=wp.float32),
    first: int,
    dt: float,
):
    world, sensor = wp.tid()
    address = first + 4 * sensor + 1
    force = wp.vec3(
        sensors[world, address],
        sensors[world, address + 1],
        sensors[world, address + 2],
    )
    magnitude = wp.length(force)
    peaks[world, sensor] = wp.max(peaks[world, sensor], magnitude)
    impulses[world, sensor] += magnitude * dt
