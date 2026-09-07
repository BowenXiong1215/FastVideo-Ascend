# SPDX-License-Identifier: Apache-2.0
"""Dense four-step DMD2 bring-up for joint MiniMax-H3 video/audio.

This is an engineering recipe for exercising the complete student/teacher/
critic path before the FastH3 quality recipe is public.  It deliberately keeps
all three roles on dense attention and requires teacher guidance scale 1.0.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import torch
import torch.nn.functional as F

from fastvideo.train.methods.distribution_matching.dmd2 import DMD2Method
from fastvideo.train.models.base import ModelBase
from fastvideo.train.utils.config import get_optional_float

JointLatents = tuple[torch.Tensor, torch.Tensor]

_BASE_TIMESTEP_MAX = 999
_VIDEO_SHIFT = 12.0
_AUDIO_SHIFT = 3.0


def _shift_noise_amount(amount: torch.Tensor, shift: float) -> torch.Tensor:
    return shift * amount / (1.0 + (shift - 1.0) * amount)


class MiniMaxH3DMD2Method(DMD2Method):
    """DMD2 over an ordered ``(video, stereo-audio)`` latent pair."""

    @staticmethod
    def _is_progress_rank() -> bool:
        dist = torch.distributed
        return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0

    def _emit_progress(self, message: str) -> None:
        if not self._is_progress_rank():
            return
        started = getattr(self, "_progress_started_at", None)
        elapsed = ""
        if isinstance(started, float):
            elapsed = f" elapsed={time.perf_counter() - started:.1f}s"
        print(
            f"[MiniMax-H3 DMD2][{time.strftime('%H:%M:%S')}] "
            f"step={getattr(self, '_progress_iteration', '?')} {message}{elapsed}",
            flush=True,
        )

    def _set_progress_phase(self, phase: str) -> None:
        self._progress_phase = phase
        self._emit_progress(phase)

    def _heartbeat_loop(self, stop_event: threading.Event) -> None:
        while not stop_event.wait(60.0):
            self._emit_progress(
                "still working: " + str(getattr(self, "_progress_phase", "training")),
            )

    def _start_progress(self, iteration: int) -> None:
        previous = getattr(self, "_progress_stop_event", None)
        if isinstance(previous, threading.Event):
            previous.set()
        self._progress_iteration = int(iteration)
        self._progress_started_at = time.perf_counter()
        self._progress_phase = "starting"
        if self._is_progress_rank():
            stop_event = threading.Event()
            self._progress_stop_event = stop_event
            threading.Thread(
                target=self._heartbeat_loop,
                args=(stop_event, ),
                name="minimax-h3-dmd2-heartbeat",
                daemon=True,
            ).start()
        self._set_progress_phase("training step started; preparing batch")

    def _stop_progress(self) -> None:
        stop_event = getattr(self, "_progress_stop_event", None)
        if isinstance(stop_event, threading.Event):
            stop_event.set()

    def single_train_step(
        self,
        batch: dict[str, Any],
        iteration: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
        self._start_progress(iteration)
        try:
            result = super().single_train_step(batch, iteration)
        except BaseException:
            self._set_progress_phase("training step failed or was interrupted")
            self._stop_progress()
            raise
        self._set_progress_phase("all forward losses assembled")
        return result

    def backward(
        self,
        loss_map: dict[str, torch.Tensor],
        outputs: dict[str, Any],
        *,
        grad_accum_rounds: int = 1,
    ) -> None:
        grad_accum_rounds = max(1, int(grad_accum_rounds))
        backward_ctx = outputs.get("_fv_backward")
        if not isinstance(backward_ctx, dict):
            self._set_progress_phase("combined backward")
            super().backward(
                loss_map,
                outputs,
                grad_accum_rounds=grad_accum_rounds,
            )
            self._set_progress_phase("combined backward complete")
            return

        if bool(backward_ctx.get("update_student", False)):
            student_ctx = backward_ctx.get("student_ctx")
            if student_ctx is None:
                raise RuntimeError("Missing student backward context")
            self._set_progress_phase("student backward (activation recomputation may run)")
            self.student.backward(
                loss_map["generator_loss"],
                student_ctx,
                grad_accum_rounds=grad_accum_rounds,
            )
            self._set_progress_phase("student backward complete")

        critic_ctx = backward_ctx.get("critic_ctx")
        if critic_ctx is None:
            raise RuntimeError("Missing critic backward context")
        self._set_progress_phase("critic backward (activation recomputation may run)")
        self.critic.backward(
            loss_map["fake_score_loss"],
            critic_ctx,
            grad_accum_rounds=grad_accum_rounds,
        )
        self._set_progress_phase("critic backward complete")

    def optimizers_schedulers_step(self, iteration: int) -> None:
        self._set_progress_phase("student and critic optimizer steps")
        super().optimizers_schedulers_step(iteration)
        self._set_progress_phase("optimizer steps complete")

    def checkpoint_state(self) -> dict[str, Any]:
        self._set_progress_phase("assembling DCP checkpoint state")
        return super().checkpoint_state()

    def _parse_score_timestep_bounds(self) -> tuple[int, int]:
        min_ratio = get_optional_float(
            self.method_config,
            "min_timestep_ratio",
            where="method.min_timestep_ratio",
        )
        max_ratio = get_optional_float(
            self.method_config,
            "max_timestep_ratio",
            where="method.max_timestep_ratio",
        )
        min_ratio = 0.0 if min_ratio is None else float(min_ratio)
        max_ratio = 1.0 if max_ratio is None else float(max_ratio)
        if not 0.0 <= min_ratio <= max_ratio <= 1.0:
            raise ValueError("method min/max_timestep_ratio must satisfy 0 <= min <= max <= 1")
        return int(min_ratio * _BASE_TIMESTEP_MAX), int(max_ratio * _BASE_TIMESTEP_MAX)

    def _sample_score_timestep(self, device: torch.device) -> torch.Tensor:
        return torch.randint(
            self._score_min_timestep,
            self._score_max_timestep + 1,
            (1, ),
            device=device,
            dtype=torch.long,
            generator=self.cuda_generator,
        )

    @staticmethod
    def _noise_amounts(base_timestep: torch.Tensor, *, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        base = base_timestep.to(device=device, dtype=torch.float32).reshape(1)
        base = (base / float(_BASE_TIMESTEP_MAX)).clamp(0.0, 1.0)
        return (
            _shift_noise_amount(base, _VIDEO_SHIFT),
            _shift_noise_amount(base, _AUDIO_SHIFT),
        )

    def _set_joint_model_input(
        self,
        batch: Any,
        latents: JointLatents,
        base_timestep: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video, audio = latents
        video_sigma, audio_sigma = self._noise_amounts(base_timestep, device=video.device)
        batch.noisy_model_input = video.permute(0, 2, 1, 3, 4)
        batch.audio_noisy_model_input = audio
        batch.sigmas = video_sigma.to(video.dtype).view(1, 1, 1, 1, 1)
        batch.audio_sigmas = audio_sigma.to(audio.dtype).view(1, 1, 1, 1)
        batch.timesteps = 1.0 - video_sigma
        batch.audio_timesteps = 1.0 - audio_sigma
        return video_sigma, audio_sigma

    def _predict_joint_x0(
        self,
        model: ModelBase,
        latents: JointLatents,
        base_timestep: torch.Tensor,
        batch: Any,
    ) -> JointLatents:
        video, audio = latents
        video_sigma, audio_sigma = self._set_joint_model_input(batch, latents, base_timestep)
        prediction = model.predict_noise(
            video,
            batch.timesteps,
            batch,
            conditional=True,
            cfg_uncond=None,
            attn_kind="dense",
        )
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise TypeError("MiniMaxH3DMD2Method requires a (video, audio) prediction")
        video_prediction, audio_prediction = prediction
        video_x0 = video - video_sigma.to(video.dtype).view(1, 1, 1, 1, 1) * video_prediction
        audio_x0 = audio - audio_sigma.to(audio.dtype).view(1, 1, 1, 1) * audio_prediction
        return video_x0, audio_x0

    def _add_joint_noise(
        self,
        clean: JointLatents,
        noise: JointLatents,
        base_timestep: torch.Tensor,
    ) -> JointLatents:
        clean_video, clean_audio = clean
        noise_video, noise_audio = noise
        video_sigma, audio_sigma = self._noise_amounts(base_timestep, device=clean_video.device)
        video_sigma = video_sigma.to(clean_video.dtype).view(1, 1, 1, 1, 1)
        audio_sigma = audio_sigma.to(clean_audio.dtype).view(1, 1, 1, 1)
        return (
            (1.0 - video_sigma) * clean_video + video_sigma * noise_video,
            (1.0 - audio_sigma) * clean_audio + audio_sigma * noise_audio,
        )

    def _random_joint_like(self, latents: JointLatents) -> JointLatents:
        video, audio = latents
        return (
            torch.randn(video.shape, device=video.device, dtype=video.dtype, generator=self.cuda_generator),
            torch.randn(audio.shape, device=audio.device, dtype=audio.dtype, generator=self.cuda_generator),
        )

    def _student_rollout(
        self,
        batch: Any,
        *,
        with_grad: bool,
    ) -> JointLatents:
        if batch.audio_latents is None:
            raise RuntimeError("MiniMax-H3 DMD2 requires TrainingBatch.audio_latents")
        clean_shapes = (batch.latents, batch.audio_latents)
        step_list = self._get_denoising_step_list(batch.latents.device)
        target_index = int(
            torch.randint(
                0,
                len(step_list),
                (1, ),
                device=batch.latents.device,
                generator=self.cuda_generator,
            ).item())

        step_values = [int(value) for value in step_list.detach().cpu().tolist()]
        rollout_name = "student-gradient rollout" if with_grad else "critic-target rollout"
        total_forwards = target_index + 1
        target_timestep = step_values[target_index]
        self._set_progress_phase(
            f"{rollout_name}: sampled target t={target_timestep}; "
            f"running {total_forwards} student forward(s)",
        )

        current = self._random_joint_like(clean_shapes)
        with torch.no_grad():
            for step_index in range(target_index):
                self._set_progress_phase(
                    f"{rollout_name}: student forward {step_index + 1}/{total_forwards} "
                    f"at t={step_values[step_index]}",
                )
                predicted_clean = self._predict_joint_x0(
                    self.student,
                    current,
                    step_list[step_index],
                    batch,
                )
                current = self._add_joint_noise(
                    predicted_clean,
                    self._random_joint_like(clean_shapes),
                    step_list[step_index + 1],
                )

        self._set_progress_phase(
            f"{rollout_name}: student forward {total_forwards}/{total_forwards} "
            f"at t={target_timestep}{' with grad' if with_grad else ''}",
        )
        if with_grad:
            result = self._predict_joint_x0(self.student, current, step_list[target_index], batch)
        else:
            with torch.no_grad():
                result = self._predict_joint_x0(self.student, current, step_list[target_index], batch)
        batch.dmd_latent_vis_dict["generator_timestep"] = step_list[target_index].float().detach()
        self._set_progress_phase(f"{rollout_name} complete")
        return result

    def _critic_flow_matching_loss(
        self,
        batch: Any,
    ) -> tuple[torch.Tensor, Any, dict[str, Any]]:
        with torch.no_grad():
            generated = self._student_rollout(batch, with_grad=False)
        timestep = self._sample_score_timestep(generated[0].device)
        noise = self._random_joint_like(generated)
        noisy = self._add_joint_noise(generated, noise, timestep)
        self._set_joint_model_input(batch, noisy, timestep)
        self._set_progress_phase(
            "critic flow-matching forward at sampled score timestep",
        )
        prediction = self.critic.predict_noise(
            noisy[0],
            batch.timesteps,
            batch,
            conditional=True,
            cfg_uncond=None,
            attn_kind="dense",
        )
        if not isinstance(prediction, tuple) or len(prediction) != 2:
            raise TypeError("MiniMax-H3 critic must return video and audio predictions")
        video_target = noise[0] - generated[0]
        audio_target = noise[1] - generated[1]
        video_loss = F.mse_loss(prediction[0].float(), video_target.float())
        audio_loss = F.mse_loss(prediction[1].float(), audio_target.float())
        loss = video_loss + audio_loss
        outputs = {
            "fake_score_latent_vis_dict": {
                "generator_pred_video": generated[0],
                "fake_score_timestep": timestep,
            },
            "video_fake_score_loss": video_loss.detach(),
            "audio_fake_score_loss": audio_loss.detach(),
        }
        self._set_progress_phase("critic flow-matching loss assembled")
        return loss, (batch.timesteps, batch.attn_metadata), outputs

    def _dmd_loss(
        self,
        generated: JointLatents,
        batch: Any,
    ) -> torch.Tensor:
        guidance_scale = get_optional_float(
            self.method_config,
            "real_score_guidance_scale",
            where="method.real_score_guidance_scale",
        )
        guidance_scale = 1.0 if guidance_scale is None else float(guidance_scale)
        if guidance_scale != 1.0:
            raise ValueError("MiniMax-H3 bring-up DMD2 requires real_score_guidance_scale=1.0")

        with torch.no_grad():
            timestep = self._sample_score_timestep(generated[0].device)
            noisy = self._add_joint_noise(generated, self._random_joint_like(generated), timestep)
            self._set_progress_phase("DMD fake-score critic forward at sampled score timestep")
            fake = self._predict_joint_x0(self.critic, noisy, timestep, batch)
            self._set_progress_phase("DMD real-score teacher forward at the same score timestep")
            real = self._predict_joint_x0(self.teacher, noisy, timestep, batch)
            self._set_progress_phase("DMD teacher and critic score forwards complete")
            video_denom = torch.abs(generated[0] - real[0]).mean().clamp_min(1e-6)
            audio_denom = torch.abs(generated[1] - real[1]).mean().clamp_min(1e-6)
            video_grad = torch.nan_to_num((fake[0] - real[0]) / video_denom)
            audio_grad = torch.nan_to_num((fake[1] - real[1]) / audio_denom)

        video_loss = 0.5 * F.mse_loss(
            generated[0].float(),
            (generated[0].float() - video_grad.float()).detach(),
        )
        audio_loss = 0.5 * F.mse_loss(
            generated[1].float(),
            (generated[1].float() - audio_grad.float()).detach(),
        )
        self._set_progress_phase("DMD generator loss assembled")
        return video_loss + audio_loss


__all__ = ["MiniMaxH3DMD2Method"]
