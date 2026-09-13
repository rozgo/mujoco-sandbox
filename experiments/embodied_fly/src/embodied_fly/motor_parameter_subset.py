"""Training-only gradient selection inside the existing final motor layer."""

import numpy as np
import torch


class WingOutputSubset:
    def __init__(self, actor, channels):
        channels = np.asarray(channels, dtype=np.int64)
        self.layer = actor.motor_decoder[3]
        if (
            channels.shape != (6,)
            or len(np.unique(channels)) != 6
            or (channels.min() < 0 or channels.max() >= self.layer.out_features)
        ):
            raise ValueError("Select the six distinct existing wing actuator rows")
        self.actor = actor
        self.channels = torch.as_tensor(channels, device=self.layer.weight.device)
        self.other = torch.as_tensor(
            np.setdiff1d(np.arange(self.layer.out_features), channels),
            device=self.layer.weight.device,
        )
        self.initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        actor.zero_grad(set_to_none=True)
        actor.requires_grad_(False)
        self.layer.requires_grad_(True)
        mask = torch.zeros_like(self.layer.bias)
        mask[self.channels] = 1
        self.handles = [
            self.layer.weight.register_hook(lambda grad: grad * mask[:, None]),
            self.layer.bias.register_hook(lambda grad: grad * mask),
        ]

    def gradient_audit(self):
        result = {}
        for name, parameter in (("weight", self.layer.weight), ("bias", self.layer.bias)):
            grad = parameter.grad
            if grad is None or not torch.isfinite(grad).all() or grad[self.other].any():
                raise RuntimeError("Wing selection must produce finite, wing-only gradients")
            norm = float(grad[self.channels].norm())
            if norm <= 0:
                raise RuntimeError("Wing output parameters must receive learning gradients")
            result[name] = {"selected_gradient_l2": norm, "other_rows_gradient_zero": True}
        if any(
            p.grad is not None
            for name, p in self.actor.named_parameters()
            if not name.startswith("motor_decoder.3.")
        ):
            raise RuntimeError("Upstream parameters must remain outside this training graph")
        return result

    def verify_and_report(self):
        current = self.actor.state_dict()
        for name, value in current.items():
            if name in ("motor_decoder.3.weight", "motor_decoder.3.bias"):
                if not torch.equal(value[self.other], self.initial[name][self.other]):
                    raise RuntimeError("Non-wing output parameters changed")
            elif not torch.equal(value, self.initial[name]):
                raise RuntimeError(f"Frozen upstream state changed: {name}")
        selected = len(self.channels) * (self.layer.in_features + 1)
        return {
            "mode": "wing-output",
            "selected_actuator_rows": self.channels.cpu().tolist(),
            "effective_trainable_parameters": selected,
            "tensor_requires_grad_parameters": sum(
                p.numel() for p in self.actor.parameters() if p.requires_grad
            ),
            "upstream_parameters_and_buffers_unchanged": True,
            "nonwing_output_rows_unchanged": True,
            "selected_parameter_changes_l2": {
                name: float(
                    (current[name][self.channels] - self.initial[name][self.channels]).norm()
                )
                for name in ("motor_decoder.3.weight", "motor_decoder.3.bias")
            },
            "runtime_module": False,
            "runtime_action_mask": False,
            "scope": "Gradient selection only; changed wing feedback can still change physical ground behavior",
        }


class WingResidualSubset:
    """Online learning of the existing optional decoder, with original state frozen."""

    def __init__(self, actor, channels):
        if actor.wing_residual is None or list(channels) != list(range(14, 20)):
            raise ValueError("An enabled canonical wing readout is required")
        self.actor = actor
        self.initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
        actor.zero_grad(set_to_none=True)
        actor.requires_grad_(False)
        actor.wing_residual.requires_grad_(True)

    def gradient_audit(self):
        audit = {}
        for name, parameter in self.actor.named_parameters():
            if name.startswith("wing_residual."):
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise RuntimeError("Wing readout must receive finite gradients")
                audit[name] = {"gradient_l2": float(parameter.grad.norm())}
            elif parameter.grad is not None:
                raise RuntimeError("Original actor parameters received gradients")
        if not any(item["gradient_l2"] > 0 for item in audit.values()):
            raise RuntimeError("No wing readout learning gradient")
        return audit

    def verify_and_report(self):
        current = self.actor.state_dict()
        eligible = {name for name, p in self.actor.named_parameters() if p.requires_grad}
        for name, value in current.items():
            if name not in eligible and not torch.equal(value, self.initial[name]):
                raise RuntimeError(f"Frozen original state or normalization changed: {name}")
        return {
            "mode": "wing-residual",
            "effective_trainable_parameters": sum(
                p.numel() for p in self.actor.parameters() if p.requires_grad
            ),
            "upstream_parameters_and_buffers_unchanged": True,
            "nonwing_output_rows_unchanged": True,
            "all_original_actor_state_unchanged": True,
            "feature_normalization_unchanged": True,
            "selected_parameter_changes_l2": {
                name: float((current[name] - self.initial[name]).norm())
                for name in sorted(eligible)
            },
            "runtime_action_mask": False,
            "runtime_readout": "same optional feedforward decoder in the actor",
        }
