"""Within-episode ground command changes; training requests, never a gait."""

import numpy as np


class GroundCommandCurriculum:
    def __init__(self, tasks, worlds, seconds):
        self.tasks, self.env = tasks, tasks.env
        if worlds < 2 or worlds % 2 or not np.isfinite(seconds) or seconds <= 0:
            raise ValueError(
                "An even positive transition-world count and interval are required"
            )
        self.period = round(seconds / self.env.control_dt)
        if self.period < 2:
            raise ValueError("Command phases must contain at least two physical actions")
        self.initial_tasks = tasks.task_ids.copy()
        groups = [np.flatnonzero(self.initial_tasks == i) for i in (0, 1)]
        if any(len(group) <= worlds // 2 for group in groups):
            raise ValueError("Keep fixed standing and walking rehearsal worlds")
        self.ids = np.concatenate([group[: worlds // 2] for group in groups])
        self.events = []

    def advance(self, memory):
        phase = self.env.ages[self.ids] // self.period
        desired = self.initial_tasks[self.ids] ^ (phase % 2)
        changed = self.ids[desired != self.tasks.task_ids[self.ids]]
        for i in changed:
            previous = int(self.tasks.task_ids[i])
            current = int(self.initial_tasks[i] ^ ((self.env.ages[i] // self.period) % 2))
            self.events.append(
                {
                    "world": int(i),
                    "episode_action": int(self.env.ages[i]),
                    "episode_seconds": float(self.env.ages[i] * self.env.control_dt),
                    "from_command_cm_s": previous,
                    "to_command_cm_s": current,
                    "neural_state_l2_at_boundary": float(memory[:, i].norm().detach()),
                    "physical_or_neural_reset": False,
                }
            )
            self.tasks.task_ids[i] = current
            self.env.command[i] = (current, 0, 0)
        return changed

    def prepare_reset(self, ids):
        """Restore requested initial tasks only before a real episode reset."""
        self.tasks.task_ids[ids] = self.initial_tasks[ids]

    def supervised_actions(self, parent_actions, teacher):
        """Reference labels for switching worlds; frozen-parent fixed rehearsal.

        The parent fails command switching, so copying it in those worlds would
        teach the wrong behavior. No returned label is a runtime motor override.
        """
        labels = teacher.posture.targets(
            teacher.ground.act().cpu().numpy(), self.tasks.task_ids
        )
        result = parent_actions.copy()
        result[self.ids] = labels[self.ids]
        return result

    def report(self):
        return {
            "enabled": True,
            "transition_world_ids": self.ids.tolist(),
            "transition_worlds": len(self.ids),
            "phase_seconds": self.period * self.env.control_dt,
            "fixed_rehearsal_worlds": {
                name: int(
                    np.sum(self.initial_tasks == i) - (len(self.ids) // 2 if i < 2 else 0)
                )
                for i, name in enumerate(("stand", "walk", "hover"))
            },
            "initial_worlds_by_task": np.bincount(self.initial_tasks, minlength=3).tolist(),
            "command_events": self.events,
            "neural_resets_at_commands": 0,
            "physical_resets_at_commands": 0,
            "transition_labels": "current-state inherited walking reference; full initial-form stand",
            "fixed_ground_labels": "frozen parent plus initial-form/resting-wing targets",
            "hover_labels": "unchanged measured-state hover reference",
            "runtime_module": False,
        }
