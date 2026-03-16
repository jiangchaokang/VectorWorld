import numpy as np
import torch
from torch import nn

from vectorworld.utils.diffusion_helpers import cosine_beta_schedule, extract
from vectorworld.utils.losses import GeometricLosses
from vectorworld.networks.egr_dit import EGRDiT
from configs.config import BEFORE_PARTITION


# -------------------------------------------------------------------------
# Diffusion LDM(original)DDPMVersion)
# -------------------------------------------------------------------------
class LDM(nn.Module):
    """Original diffusion-based Latent Diffusion Model."""

    def __init__(self, cfg):
        super(LDM, self).__init__()

        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset
        self.model = EGRDiT(cfg)

        n_timesteps = self.cfg_model.n_diffusion_timesteps
        betas = cosine_beta_schedule(n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        # For diffusion onlyLDMRandomity of the sampling phase
        self.lane_sampling_temperature = self.cfg_model.lane_sampling_temperature

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # q(x_t | x_{t-1})And its derivatives.
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1),
        )

        # Auditq(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (
            1.0 - alphas_cumprod
        )
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        loss_type = self.cfg.train.loss_type
        self.lane_loss_fn = GeometricLosses[loss_type]((1, 2))
        self.agent_loss_fn = GeometricLosses[loss_type]((1, 2))

    # ---------------------------------------------------------------------
    # q / p helpers
    # ---------------------------------------------------------------------
    def predict_start_from_noise(self, x_t, t, noise):
        """Given x_t and predicted noise, recover x_0."""
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """Posterior q(x_{t-1} | x_t, x_0)is the difference between the mean and the log."""
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x_agent, x_lane, data, t_agent, t_lane):
        """Predict mean/logvar of p(x_{t-1} | x_t)."""
        # Noise prediction (with/ without conditions) for classifier-free guidance
        conditional_epsilon_agent, conditional_epsilon_lane = self.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=False
        )
        unconditional_epsilon_agent, unconditional_epsilon_lane = self.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=True
        )

        s = self.cfg.train.guidance_scale
        epsilon_agent = unconditional_epsilon_agent + s * (
            conditional_epsilon_agent - unconditional_epsilon_agent
        )
        epsilon_lane = unconditional_epsilon_lane + s * (
            conditional_epsilon_lane - unconditional_epsilon_lane
        )

        t_agent = t_agent.detach().to(torch.int64)
        t_lane = t_lane.detach().to(torch.int64)

        # Noise predictions x_0, after which parameters are obtained
        x_agent_recon = self.predict_start_from_noise(
            x_agent, t=t_agent, noise=epsilon_agent
        )
        x_lane_recon = self.predict_start_from_noise(
            x_lane, t=t_lane, noise=epsilon_lane
        )

        model_mean_agent, posterior_log_variance_agent = self.q_posterior(
            x_start=x_agent_recon, x_t=x_agent, t=t_agent
        )
        model_mean_lane, posterior_log_variance_lane = self.q_posterior(
            x_start=x_lane_recon, x_t=x_lane, t=t_lane
        )

        return (
            model_mean_agent,
            posterior_log_variance_agent,
            model_mean_lane,
            posterior_log_variance_lane,
        )

    # ---------------------------------------------------------------------
    # Sampling (DDPM reverse process)
    # ---------------------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_agent, x_lane, data, t_agent, t_lane):
        """Single reverse step p(x_{t-1} | x_t)."""
        b_agent = t_agent.shape[0]
        b_lane = t_lane.shape[0]

        (
            model_mean_agent,
            model_log_variance_agent,
            model_mean_lane,
            model_log_variance_lane,
        ) = self.p_mean_variance(x_agent, x_lane, data, t_agent, t_lane)

        noise_agent = torch.randn_like(x_agent)
        noise_lane = torch.randn_like(x_lane)

        # t= 0 No more noise
        nonzero_mask_agent = (1 - (t_agent == 0).float()).reshape(
            b_agent, *((1,) * (len(x_agent.shape) - 1))
        )
        nonzero_mask_lane = (1 - (t_lane == 0).float()).reshape(
            b_lane, *((1,) * (len(x_lane.shape) - 1))
        )

        next_x_agent = (
            model_mean_agent
            + nonzero_mask_agent * (model_log_variance_agent).exp().sqrt() * noise_agent
        )
        # Lane branch allows randomity to be regulated by texture
        next_x_lane = (
            model_mean_lane
            + nonzero_mask_lane
            * (model_log_variance_lane).exp().sqrt()
            * noise_lane
            * self.lane_sampling_temperature
        )

        return next_x_agent, next_x_lane

    @torch.no_grad()
    def p_sample_loop(
        self,
        agent_shape,
        lane_shape,
        data,
        device="cuda",
        mode="initial_scene",
        return_diffusion_chain=False,
    ):
        """Run full reverse diffusion to sample latent trajectories."""
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # Initial x_T
        x_agent = torch.randn(agent_shape, device=device)

        # Line_conventioned: lane latet fixed as condition
        if mode == "lane_conditioned":
            x_lane = data["lane"].latents[:, np.newaxis, :].to(device)
        else:
            x_lane = (
                torch.randn(lane_shape, device=device) * self.lane_sampling_temperature
            )

        # Conditionalization of some nodes in train/inputing mode
        if mode == "train":
            # Currently in progress Partition_mask is Bool: True for "before Partition"
            agent_fixed_mask = data["agent"].partition_mask.bool()
            lane_fixed_mask = data["lane"].partition_mask.bool()
            if hasattr(data["agent"], "latents"):
                x_agent[agent_fixed_mask] = data["agent"].latents[
                    agent_fixed_mask
                ].unsqueeze(1)
            if hasattr(data["lane"], "latents"):
                x_lane[lane_fixed_mask] = data["lane"].latents[
                    lane_fixed_mask
                ].unsqueeze(1)

        if mode == "inpainting":
            cond_lane_mask = data["lane"].mask.bool()
            cond_agent_mask = data["agent"].mask.bool()
            if hasattr(data["lane"], "latents"):
                x_lane[cond_lane_mask] = data["lane"].latents[
                    cond_lane_mask
                ].unsqueeze(1)
            if hasattr(data["agent"], "latents"):
                x_agent[cond_agent_mask] = data["agent"].latents[
                    cond_agent_mask
                ].unsqueeze(1)

        if return_diffusion_chain:
            diffusion_chain = [(x_agent, x_lane)]

        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            t_agent = timesteps[agent_batch]
            t_lane = timesteps[lane_batch]

            x_agent, x_lane = self.p_sample(x_agent, x_lane, data, t_agent, t_lane)

            # Numeric stability clipping
            x_agent = torch.clip(
                x_agent, -self.cfg_model.diffusion_clip, self.cfg_model.diffusion_clip
            )
            if mode == "lane_conditioned":
                x_lane = data["lane"].latents[:, np.newaxis, :].to(device)
            else:
                x_lane = torch.clip(
                    x_lane,
                    -self.cfg_model.diffusion_clip,
                    self.cfg_model.diffusion_clip,
                )

            # Reimposed conditions (inpainting /rain)
            if mode == "inpainting":
                cond_lane_mask = data["lane"].mask.bool()
                cond_agent_mask = data["agent"].mask.bool()
                if hasattr(data["lane"], "latents"):
                    x_lane[cond_lane_mask] = data["lane"].latents[
                        cond_lane_mask
                    ].unsqueeze(1)
                if hasattr(data["agent"], "latents"):
                    x_agent[cond_agent_mask] = data["agent"].latents[
                        cond_agent_mask
                    ].unsqueeze(1)

            if mode == "train":
                agent_fixed_mask = data["agent"].partition_mask.bool()
                lane_fixed_mask = data["lane"].partition_mask.bool()
                if hasattr(data["agent"], "latents"):
                    x_agent[agent_fixed_mask] = data["agent"].latents[
                        agent_fixed_mask
                    ].unsqueeze(1)
                if hasattr(data["lane"], "latents"):
                    x_lane[lane_fixed_mask] = data["lane"].latents[
                        lane_fixed_mask
                    ].unsqueeze(1)

            if return_diffusion_chain:
                diffusion_chain.append((x_agent, x_lane))

        if return_diffusion_chain:
            return x_agent[:, 0], x_lane[:, 0], diffusion_chain
        else:
            return x_agent[:, 0], x_lane[:, 0]

    @torch.no_grad()
    def forward(self, data, mode="initial_scene"):
        """Generate samples from diffusion LDM (interface used by ScenarioDreamerLDM)."""

        agent_shape = data["agent"].x[:, np.newaxis, :].shape
        lane_shape = data["lane"].x[:, np.newaxis, :].shape

        return self.p_sample_loop(
            agent_shape,
            lane_shape,
            data,
            device=data["agent"].x.device,
            mode=mode,
            return_diffusion_chain=False,
        )

    # ---------------------------------------------------------------------
    # Training losses (DDPM)
    # ---------------------------------------------------------------------
    def q_sample(self, x_start, t, noise):
        """Forward diffusion q(x_t | x_0)."""
        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_agent, x_lane, data, t_agent, t_lane):
        """Compute DDPM loss for a batch."""
        # Sample noise and generate noised sample
        agent_noise = torch.randn_like(x_agent)
        x_agent_noisy = self.q_sample(x_start=x_agent, t=t_agent, noise=agent_noise)

        lane_noise = torch.randn_like(x_lane)
        x_lane_noisy = self.q_sample(x_start=x_lane, t=t_lane, noise=lane_noise)

        # No noise on the condition section, target noise=0.
        # Partition_mask is the condition part of the tool:True.
        agent_fixed_mask = data["agent"].partition_mask.bool()
        lane_fixed_mask = data["lane"].partition_mask.bool()

        x_agent_noisy[agent_fixed_mask] = x_agent[agent_fixed_mask]
        x_lane_noisy[lane_fixed_mask] = x_lane[lane_fixed_mask]

        # Forecast Noise
        agent_noise_pred, lane_noise_pred = self.model(
            x_agent_noisy, x_lane_noisy, data, t_agent, t_lane
        )

        assert agent_noise.shape == agent_noise_pred.shape
        assert lane_noise.shape == lane_noise_pred.shape

        # Target noise of the condition part is 0
        agent_noise[agent_fixed_mask] = 0.0
        lane_noise[lane_fixed_mask] = 0.0

        agent_loss = self.agent_loss_fn(
            agent_noise_pred, agent_noise, data["agent"].batch
        )
        lane_loss = self.lane_loss_fn(
            lane_noise_pred, lane_noise, data["lane"].batch
        )

        loss = agent_loss + self.cfg.train.lane_weight * lane_loss
        return loss, agent_loss, lane_loss

    def loss(self, data):
        """High-level training loss entry for diffusion LDM."""
        x_agent = data["agent"].latents.unsqueeze(1)
        x_lane = data["lane"].latents.unsqueeze(1)

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        t = torch.randint(
            0, self.n_timesteps, (batch_size,), device=x_agent.device
        ).long()
        t_agent = t[agent_batch]
        t_lane = t[lane_batch]

        loss, agent_loss, lane_loss = self.p_losses(
            x_agent, x_lane, data, t_agent, t_lane
        )

        loss_dict = {
            "loss": loss.mean(),
            "agent_loss": agent_loss.mean().detach(),
            "lane_loss": lane_loss.mean().detach(),
        }
        return loss_dict


# -------------------------------------------------------------------------
# FlowLDM: Rectified Flow / Flow Matching on latents
# -------------------------------------------------------------------------
class FlowLDM(nn.Module):
    """Rectified Flow / Flow Matching variant of the latent generative model.

    - Training: study Victor Field v_ (x_t, t) approaching the speed field (-z).
    - Sampling: throughODESolve from t=1 ~ 0, starting strictly toN(0, I)I don't know.
    """

    def __init__(self, cfg):
        super(FlowLDM, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset

        self.model = EGRDiT(cfg)

        # The flat_sampling_temperature configuration remains in thelow model, but no more base noise;
        # If the "temperature" is to be regulated, it should be used in outer latent space (e.g. post)-hocI don't know.
        self.lane_sampling_temperature = self.cfg_model.lane_sampling_temperature

        self.n_steps = int(getattr(self.cfg_model, "flow_num_steps"))
        self.diffusion_clip = getattr(self.cfg_model, "diffusion_clip")
        self.flow_solver = getattr(self.cfg_model, "flow_solver").lower()

        loss_type = self.cfg.train.loss_type
        self.lane_loss_fn = GeometricLosses[loss_type]((1, 2))
        self.agent_loss_fn = GeometricLosses[loss_type]((1, 2))

    # ------------------------- training: flow matching ---------------------
    def loss(self, data):
        """Rectified Flow matching loss (on normalized latents)."""
        agent_latents = data["agent"].latents.unsqueeze(1)  # (Na, 1, Da)
        lane_latents = data["lane"].latents.unsqueeze(1)  # (Nl, 1, Dl)

        device = agent_latents.device
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # base noise ~ N(0, I)
        eps_agent = torch.randn_like(agent_latents)
        eps_lane = torch.randn_like(lane_latents)

        # Partition_mask is conditional: Boolean True means fixed (BEFORE_PARTITIONI'm not sure how to do it.
        agent_fixed_mask = data["agent"].partition_mask.bool()
        lane_fixed_mask = data["lane"].partition_mask.bool()

        # t ~ Uniform(0, 1) per scene
        t_scene = torch.rand(batch_size, device=device)
        t_agent = t_scene[agent_batch]  # (Na,)
        t_lane = t_scene[lane_batch]  # (Nl,)

        t_agent_b = t_agent[:, None, None]
        t_lane_b = t_lane[:, None, None]

        z_agent = agent_latents
        z_lane = lane_latents

        # Linear plug-in value x_t = (1-t) *z_agent+t *eps_agent
        x_agent_t = (1.0 - t_agent_b) * z_agent + t_agent_b * eps_agent
        x_lane_t = (1.0 - t_lane_b) * z_lane + t_lane_b * eps_lane

        # Fixed part path constant-> speed 0
        x_agent_t[agent_fixed_mask] = z_agent[agent_fixed_mask]
        x_lane_t[lane_fixed_mask] = z_lane[lane_fixed_mask]

        u_agent = eps_agent - z_agent
        u_lane = eps_lane - z_lane
        u_agent[agent_fixed_mask] = 0.0
        u_lane[lane_fixed_mask] = 0.0

        # Single forward (not during training)CFGI'm not sure how to do it.
        v_agent_pred, v_lane_pred = self.model(
            x_agent_t, x_lane_t, data, t_agent, t_lane, unconditional=False
        )

        agent_loss = self.agent_loss_fn(v_agent_pred, u_agent, agent_batch)
        lane_loss = self.lane_loss_fn(v_lane_pred, u_lane, lane_batch)
        loss = agent_loss + self.cfg.train.lane_weight * lane_loss

        return {
            "loss": loss.mean(),
            "agent_loss": agent_loss.mean().detach(),
            "lane_loss": lane_loss.mean().detach(),
        }

    # ------------------------- sampling: ODE integration -------------------
    def _cfg_vector_field(self, x_agent, x_lane, data, t_agent, t_lane):
        """Classifier-free guided vector field v_cfg(x,t)."""
        # conditional
        v_agent_cond, v_lane_cond = self.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=False
        )
        # unconditional
        v_agent_uncond, v_lane_uncond = self.model(
            x_agent, x_lane, data, t_agent, t_lane, unconditional=True
        )
        s = self.cfg.train.guidance_scale
        v_agent = v_agent_uncond + s * (v_agent_cond - v_agent_uncond)
        v_lane = v_lane_uncond + s * (v_lane_cond - v_lane_uncond)
        return v_agent, v_lane

    @torch.no_grad()
    def forward(self, data, mode: str = "initial_scene", return_chain: bool = False):
        """Generate samples using flow model; interface matches diffusion LDM.

        Parameters
        ----------
        return_chain:
            If True, return (agent_latents, lane_latents, chain is the list of length n_steps,
            Each element is (x_agent_step, x_lane_step) and is 2D length (N,D) representing the updated state of each step.
        """
        agent_shape = data["agent"].x[:, None, :].shape
        lane_shape = data["lane"].x[:, None, :].shape
        return self.sample(
            agent_shape=agent_shape,
            lane_shape=lane_shape,
            data=data,
            device=data["agent"].x.device,
            mode=mode,
            return_chain=return_chain,
        )

    @torch.no_grad()
    def sample(
        self,
        agent_shape,
        lane_shape,
        data,
        device="cuda",
        mode: str = "initial_scene",
        return_chain: bool = False,
    ):
        """ODE solver from t=1 → 0 with CFG (Euler / Heun).

        Notes
        -----
        - Default behaviour is fully consistent with the old realization;
        - When return_chain=True, return an extra item for each step to update for gradual visualization.
        """
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # Initial condition t= 1: StrictN(0, I)
        x_agent = torch.randn(agent_shape, device=device)

        if mode == "lane_conditioned":
            # lane latet is entirely conditional.
            x_lane = data["lane"].latents[:, None, :].to(device)
        else:
            x_lane = torch.randn(lane_shape, device=device)  # * self.lane_sampling_temperature

        # Condition mask
        cond_agent_mask = None
        cond_lane_mask = None

        if mode == "train":
            # Visual / debug: imposing conditions on before part of
            cond_agent_mask = data["agent"].partition_mask.bool().to(device)
            cond_lane_mask = data["lane"].partition_mask.bool().to(device)
        elif mode == "inpainting":
            cond_lane_mask = data["lane"].mask.bool().to(device)
            cond_agent_mask = data["agent"].mask.bool().to(device)
        elif mode == "lane_conditioned":
            # All lane fixed
            cond_lane_mask = torch.ones(
                x_lane.shape[0], dtype=torch.bool, device=device
            )

        # Apply initial conditions
        if cond_agent_mask is not None and hasattr(data["agent"], "latents"):
            x_agent[cond_agent_mask] = data["agent"].latents[
                cond_agent_mask
            ].unsqueeze(1).to(device)
        if cond_lane_mask is not None and hasattr(data["lane"], "latents"):
            x_lane[cond_lane_mask] = data["lane"].latents[
                cond_lane_mask
            ].unsqueeze(1).to(device)

        # Time grid t: 1 ~ 0
        t_values = torch.linspace(1.0, 0.0, steps=self.n_steps + 1, device=device)

        chain = [] if return_chain else None

        for k in range(self.n_steps):
            t_cur = t_values[k]
            t_next = t_values[k + 1]
            dt = t_cur - t_next  # > 0, actual length h = t_text -t_cur < 0

            t_scene_cur = torch.full((batch_size,), t_cur, device=device)
            t_agent_cur = t_scene_cur[agent_batch]
            t_lane_cur = t_scene_cur[lane_batch]

            v_agent_1, v_lane_1 = self._cfg_vector_field(
                x_agent, x_lane, data, t_agent_cur, t_lane_cur
            )

            if self.flow_solver == "euler":
                x_agent_next = x_agent - dt * v_agent_1
                x_lane_next = x_lane - dt * v_lane_1
            elif self.flow_solver == "heun":
                # predictor
                x_agent_tmp = x_agent - dt * v_agent_1
                x_lane_tmp = x_lane - dt * v_lane_1

                # corrector at t_next
                t_scene_next = torch.full((batch_size,), t_next, device=device)
                t_agent_next = t_scene_next[agent_batch]
                t_lane_next = t_scene_next[lane_batch]

                v_agent_2, v_lane_2 = self._cfg_vector_field(
                    x_agent_tmp, x_lane_tmp, data, t_agent_next, t_lane_next
                )

                x_agent_next = x_agent - 0.5 * dt * (v_agent_1 + v_agent_2)
                x_lane_next = x_lane - 0.5 * dt * (v_lane_1 + v_lane_2)
            else:
                raise ValueError(
                    f"Unknown flow_solver='{self.flow_solver}'. Use 'euler' or 'heun'."
                )

            x_agent, x_lane = x_agent_next, x_lane_next

            # Re-imposed conditions
            if mode == "lane_conditioned" and hasattr(data["lane"], "latents"):
                x_lane = data["lane"].latents[:, None, :].to(device)

            if cond_agent_mask is not None and hasattr(data["agent"], "latents"):
                x_agent[cond_agent_mask] = data["agent"].latents[
                    cond_agent_mask
                ].unsqueeze(1).to(device)
            if (
                cond_lane_mask is not None
                and hasattr(data["lane"], "latents")
                and mode != "lane_conditioned"
            ):
                x_lane[cond_lane_mask] = data["lane"].latents[
                    cond_lane_mask
                ].unsqueeze(1).to(device)

            # Value stable clipping
            if self.diffusion_clip is not None and self.diffusion_clip > 0:
                x_agent = torch.clamp(
                    x_agent, -self.diffusion_clip, self.diffusion_clip
                )
                if mode != "lane_conditioned":
                    x_lane = torch.clamp(
                        x_lane, -self.diffusion_clip, self.diffusion_clip
                    )

            if return_chain:
                # Record the " state updated at each step" and squeeze drop the middle dimension (N,1,D)->(N,D) to facilitate external decoding
                chain.append((x_agent[:, 0].detach().clone(), x_lane[:, 0].detach().clone()))

        if return_chain:
            return x_agent[:, 0], x_lane[:, 0], chain
        return x_agent[:, 0], x_lane[:, 0]


# -------------------------------------------------------------------------
# MeanFlowLDM: MeanFlow latent generative model
#   -Training_mode=IDityenable_jvp=True: Improved MeanFlows v-reparam + JVP
#   - trading_mode= "shortcut" or allowing_jvp=False:Shortcut-MeanFlow(formerly realized)
# -------------------------------------------------------------------------
class MeanFlowLDM(nn.Module):
    """MeanFlow latent generative model.

    Two training modules:
    1) Identity mode (recommended):
       - Use Improved Mean Blows v-reparamForm:
           Path: z_t = (1 - t) x + t e
           v_gt = e - x

           v (z_t, t) = (z_t, r=t, t) (border conditions)
           V_θ(z_t)      = u_θ(z_t, r, t) + (t-r) d/dt u_θ(z_t,r,t)
           Loss         = || V_θ(z_t) - v_gt ||^2  (scene-level GeometricL2)

           of whichd/dtI'll pass.torch.func.jvpCalculating, andJVPOutput Stop-gradientI don't know.

       - (t,r) by Logit-Normal+ meanlow_tr_ratio sampling with visible additions
         (r=0,t=1) Sample, specially covered 1-stepCase.

    2) Shortcut mode (compatible with reservations):
       - FlowMatching + Endpoint + Self-Consistency(three loss);
       - few-stepThe Euler (originally achieved) sampled from t=0 § 1.
    """

    def __init__(self, cfg):
        super(MeanFlowLDM, self).__init__()
        self.cfg = cfg
        self.cfg_model = self.cfg.model
        self.cfg_dataset = self.cfg.dataset

        # Backbone: DiT, output is interpreted as "average speed"
        self.model = EGRDiT(cfg)

        # lane_sampling_temperature: a scale used only for the initial noise of the lane during sampling
        self.lane_sampling_temperature = float(
            getattr(self.cfg_model, "lane_sampling_temperature", 1.0)
        )
        self.diffusion_clip = getattr(self.cfg_model, "diffusion_clip", None)

        # Basic Gemetric L2 (Shortcut branch & Identity branch can be reused)
        loss_type = self.cfg.train.loss_type
        self.lane_loss_fn = GeometricLosses[loss_type]((1, 2))
        self.agent_loss_fn = GeometricLosses[loss_type]((1, 2))

        # Whether or not to enable the second time channel (t and h = t) for DiT-rI'm not sure how to do it.
        self.use_two_times: bool = bool(
            getattr(self.cfg_model, "meanflow_use_two_times", False)
        )

        # Rationale step K:1 Strict one-step; >1 > new-step
        self.num_steps_eval = int(
            getattr(self.cfg_model, "meanflow_num_steps_eval", 1)
        )
        if self.num_steps_eval < 1:
            self.num_steps_eval = 1

        # Training Mode &JVPSwitch
        self.training_mode: str = str(
            getattr(self.cfg_model, "meanflow_training_mode", "identity")
        ).lower()
        self.enable_jvp: bool = bool(
            getattr(self.cfg_model, "meanflow_enable_jvp", True)
        )
        if self.training_mode not in ("identity", "shortcut"):
            raise ValueError(
                f"Unknown meanflow_training_mode='{self.training_mode}', "
                f"use 'identity' or 'shortcut'."
            )
        self.use_identity_branch: bool = (
            self.training_mode == "identity" and self.enable_jvp
        )

        # Shortcut Associated Super-Specific
        self.sc_ratio: float = float(
            getattr(self.cfg_model, "meanflow_shortcut_sc_ratio", 0.25)
        )
        self.flow_weight: float = float(
            getattr(self.cfg_model, "meanflow_shortcut_flow_weight", 1.0)
        )
        self.end_weight: float = float(
            getattr(self.cfg_model, "meanflow_shortcut_end_weight", 1.0)
        )
        self.sc_weight: float = float(
            getattr(self.cfg_model, "meanflow_pde_weight", 0.2)
        )
        self.shortcut_warmup_steps: int = int(
            getattr(self.cfg_model, "meanflow_shortcut_warmup_steps", 3000)
        )
        if self.shortcut_warmup_steps < 0:
            self.shortcut_warmup_steps = 0

        # MeanFlow Identity (scene)-levelI'm not sure how to do it.
        self.loss_p: float = float(getattr(self.cfg_model, "meanflow_loss_p", 0.0))
        self.loss_c: float = float(getattr(self.cfg_model, "meanflow_loss_c", 1e-3))

        # Training Step Counter (for Shortcut warmup)
        self.register_buffer("num_updates", torch.zeros(1, dtype=torch.long))

        # Old distillation configuration (reserved but ignored)
        self.distill_from_flow: bool = bool(
            getattr(self.cfg_model, "meanflow_distill_from_flow", False)
        )
        self.distill_weight: float = float(
            getattr(self.cfg_model, "meanflow_distill_weight", 0.0)
        )
        self.distill_teacher_ckpt = getattr(
            self.cfg_model, "meanflow_distill_teacher_ckpt", None
        )
        if self.distill_from_flow or self.distill_weight > 0.0:
            print(
                "[MeanFlowLDM] Warning: distillation-related configs are now "
                "ignored; only MeanFlow Identity / Shortcut losses are used."
            )

        print(
            f"[MeanFlowLDM] training_mode={self.training_mode}, "
            f"enable_jvp={self.enable_jvp}, "
            f"use_identity_branch={self.use_identity_branch}"
        )

    # ---------------helper: Set the number of sampling steps -------------------------------helper: Set the number of sampling steps----
    def set_num_steps_eval(self, n_steps: int):
        """Set number of MeanFlow steps to use during sampling (eval/inference)."""
        n_steps = int(n_steps)
        if n_steps < 1:
            n_steps = 1
        if n_steps > 64:
            print(
                f"[MeanFlowLDM] Warning: num_steps_eval={n_steps} is large; "
                f"this may be unnecessary for MeanFlow."
            )
        self.num_steps_eval = n_steps
        print(f"[MeanFlowLDM] num_steps_eval set to {self.num_steps_eval}")

    # ------------------------- helpers: Logit-Normal& (t,r) Sample --
    def _logit_normal_sample(self, mu: float, sigma: float, shape, device):
        """Sample from Logit-Normal: t = sigmoid(N(mu, sigma^2))."""
        eps = torch.randn(shape, device=device)
        t = torch.sigmoid(eps * sigma + mu)
        t = torch.clamp(t, 0.0, 1.0)
        return t

    def _sample_two_timesteps(self, batch_size: int, device: torch.device):
        """Sample (t, r)~Logit-Normal+ Reprocessing (Ref MeanFlow Official Time_sampler)."""
        sampler = str(getattr(self.cfg_model, "meanflow_tr_sampler", "v0")).lower()
        ratio = float(getattr(self.cfg_model, "meanflow_tr_ratio", 0.25))
        ratio = float(np.clip(ratio, 0.0, 1.0))

        base_B = batch_size
        global_ratio = 0.15  # Apparent (r=0,t=1) ratio, adjusted as required
        global_ratio = float(np.clip(global_ratio, 0.0, 0.5))
        B_global = int(global_ratio * base_B)
        B_rest = base_B - B_global

        # t/rlogit-normalParameters, allowing the distinction between t and r; no return to globalmu/sigma
        base_mu = float(getattr(self.cfg_model, "meanflow_time_mu", -0.4))
        base_sigma = float(getattr(self.cfg_model, "meanflow_time_sigma", 1.0))

        mu_t = getattr(self.cfg_model, "meanflow_time_mu_t", None)
        if mu_t is None:
            mu_t = base_mu
        sigma_t = getattr(self.cfg_model, "meanflow_time_sigma_t", None)
        if sigma_t is None:
            sigma_t = base_sigma

        mu_r = getattr(self.cfg_model, "meanflow_time_mu_r", None)
        if mu_r is None:
            mu_r = base_mu
        sigma_r = getattr(self.cfg_model, "meanflow_time_sigma_r", None)
        if sigma_r is None:
            sigma_r = base_sigma

        if B_rest > 0:
            # Step1: Independent sampling B_rest (t,r)
            t = self._logit_normal_sample(mu_t, sigma_t, (B_rest,), device)
            r = self._logit_normal_sample(mu_r, sigma_r, (B_rest,), device)

            if sampler == "v0":
                # Make sure first t> = r
                t, r = torch.maximum(t, r), torch.minimum(t, r)
                # By (1-rate) probabilities r=t
                if ratio < 1.0:
                    prob = torch.rand(B_rest, device=device)
                    mask_eq = prob < (1.0 - ratio)
                    r = torch.where(mask_eq, t, r)
            elif sampler == "v1":
                # First by (1-ratio) probability order r=t
                if ratio < 1.0:
                    prob = torch.rand(B_rest, device=device)
                    mask_eq = prob < (1.0 - ratio)
                    r = torch.where(mask_eq, t, r)
                # Make sure you don't.
                r = torch.minimum(t, r)
            else:
                raise ValueError(
                    f"Unknown meanflow_tr_sampler='{sampler}'. Use 'v0' or 'v1'."
                )
        else:
            t = torch.empty(0, device=device)
            r = torch.empty(0, device=device)

        # Render B_global samples (t=1, r=0)
        if B_global > 0:
            t_global = torch.ones(B_global, device=device)
            r_global = torch.zeros(B_global, device=device)
            t = torch.cat([t, t_global], dim=0)
            r = torch.cat([r, r_global], dim=0)

        assert t.shape[0] == base_B
        assert r.shape[0] == base_B

        return t, r

    # ------------------------- Identity: Improved MeanFlow + JVP -----------
    def _meanflow_identity_loss(self, data):
        """Improved MeanFlow Identity + JVPTraining Branch (iMF style v-reparamI'm sorry."""
        device = data["agent"].latents.device
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # ---------------- Fixed CFG label-drop mask (scene-level) ----------------
        drop_prob = float(getattr(self.cfg_model, "label_dropout", 0.0))
        scene_drop_mask = None
        if self.training and drop_prob > 0.0:
            scene_drop_mask = (torch.rand(batch_size, device=device) < drop_prob).to(torch.long)

        # Data latents(x) and a prior noise (eps)
        x_agent = data["agent"].latents.unsqueeze(1)  # (Na,1,Da)
        x_lane = data["lane"].latents.unsqueeze(1)    # (Nl,1,Dl)

        eps_agent = torch.randn_like(x_agent, device=device)
        eps_lane = torch.randn_like(x_lane, device=device)

        agent_fixed_mask = data["agent"].partition_mask.bool()
        lane_fixed_mask = data["lane"].partition_mask.bool()

        # Fixed node: Path is constant eps = x, v_gt = 0
        eps_agent[agent_fixed_mask] = x_agent[agent_fixed_mask]
        eps_lane[lane_fixed_mask] = x_lane[lane_fixed_mask]

        v_gt_agent = eps_agent - x_agent
        v_gt_lane = eps_lane - x_lane
        v_gt_agent[agent_fixed_mask] = 0.0
        v_gt_lane[lane_fixed_mask] = 0.0

        # Sampling (t,r) zirconium [0,1]
        t_scene, r_scene = self._sample_two_timesteps(batch_size, device=device)

        # node-level t, r
        t_agent = t_scene[agent_batch]
        t_lane = t_scene[lane_batch]
        r_agent = r_scene[agent_batch]
        r_lane = r_scene[lane_batch]

        t_agent_b = t_agent[:, None, None]
        t_lane_b = t_lane[:, None, None]

        # Path point z_t = (1-t) *x+t*eps
        z_agent_t = (1.0 - t_agent_b) * x_agent + t_agent_b * eps_agent
        z_lane_t = (1.0 - t_lane_b) * x_lane + t_lane_b * eps_lane

        # Make sure the fixed node is constant as x (not changed by t)
        z_agent_t[agent_fixed_mask] = x_agent[agent_fixed_mask]
        z_lane_t[lane_fixed_mask] = x_lane[lane_fixed_mask]

        # scene-level ∆t
        delta_scene = t_scene - r_scene  # (B,)
        delta_agent = delta_scene[agent_batch]  # (Na,)
        delta_lane = delta_scene[lane_batch]    # (Nl,)

        delta_agent_b = delta_agent[:, None, None]
        delta_lane_b = delta_lane[:, None, None]

        def u_func(z_agent_in, z_lane_in, t_scene_in, r_scene_in):
            t_agent_in = t_scene_in[agent_batch]
            t_lane_in = t_scene_in[lane_batch]

            if self.use_two_times:
                h_scene = t_scene_in - r_scene_in
                h_agent_in = h_scene[agent_batch]
                h_lane_in = h_scene[lane_batch]
                u_agent_out, u_lane_out = self.model(
                    z_agent_in,
                    z_lane_in,
                    data,
                    t_agent_in,
                    t_lane_in,
                    unconditional=False,
                    agent_h=h_agent_in,
                    lane_h=h_lane_in,
                    scene_drop_mask=scene_drop_mask,
                )
            else:
                u_agent_out, u_lane_out = self.model(
                    z_agent_in,
                    z_lane_in,
                    data,
                    t_agent_in,
                    t_lane_in,
                    unconditional=False,
                    scene_drop_mask=scene_drop_mask,
                )
            return u_agent_out, u_lane_out

        # 1) Average speed forecast u_ (z_t, r, t) (with gradient)
        u_agent_pred, u_lane_pred = u_func(z_agent_t, z_lane_t, t_scene, r_scene)

        # 2) Temporal speed projection v_(z_t, t) (z_t, r=t, t) (border conditions)
        v_agent_pred, v_lane_pred = u_func(z_agent_t, z_lane_t, t_scene, t_scene)

        # 3) JVPCalculated/dt u
        dtdt_scene = torch.ones_like(t_scene, device=device)
        drdt_scene = torch.zeros_like(r_scene, device=device)
        with torch.no_grad():
            (_, _), (du_agent_dt, du_lane_dt) = torch.func.jvp(
                u_func,
                (z_agent_t, z_lane_t, t_scene, r_scene),
                (v_agent_pred, v_lane_pred, dtdt_scene, drdt_scene),
            )

        # 4) Composite function V_z_t = u (t)-r) * d/dt u_θ
        V_agent_pred = u_agent_pred + delta_agent_b * du_agent_dt.detach()
        V_lane_pred = u_lane_pred + delta_lane_b * du_lane_dt.detach()

        # 5. Aggregation on scene dimensions with GemetricL2 v-loss
        agent_loss = self.agent_loss_fn(V_agent_pred, v_gt_agent, agent_batch)  # (B,)
        lane_loss = self.lane_loss_fn(V_lane_pred, v_gt_lane, lane_batch)        # (B,)

        raw_scene_loss = agent_loss + self.cfg.train.lane_weight * lane_loss     # (B,)

        # 6) scene-levelSelf-adaptation weight (optional)
        if self.loss_p != 0.0:
            w = (raw_scene_loss.detach() + self.loss_c) ** (-self.loss_p)
            scene_loss = raw_scene_loss * w
        else:
            scene_loss = raw_scene_loss

        total_loss = scene_loss.mean()

        return {
            "loss": total_loss,
            "agent_loss": agent_loss.mean().detach(),
            "lane_loss": lane_loss.mean().detach(),
            "raw_loss_flow": raw_scene_loss.mean().detach(),
            "loss_flow": scene_loss.mean().detach(),
            "loss_end": total_loss.new_tensor(0.0),
            "loss_sc": total_loss.new_tensor(0.0),
        }

    # -----------Shortcut Mode:low + Endpoint + SC --
    def _prepare_latent_pairs_shortcut(self, data):
        """Construct noise e, data z and true velocity v = z - e (use in Shortcut branch)."""
        device = data["agent"].latents.device

        z_agent = data["agent"].latents.unsqueeze(1)  # (Na,1,Da)
        z_lane = data["lane"].latents.unsqueeze(1)  # (Nl,1,Dl)

        e_agent = torch.randn_like(z_agent, device=device)
        e_lane = torch.randn_like(z_lane, device=device)

        agent_fixed_mask = data["agent"].partition_mask.bool()
        lane_fixed_mask = data["lane"].partition_mask.bool()

        # Conditional node: both starting and ending should be data z at 0
        e_agent[agent_fixed_mask] = z_agent[agent_fixed_mask]
        e_lane[lane_fixed_mask] = z_lane[lane_fixed_mask]

        v_agent = z_agent - e_agent
        v_lane = z_lane - e_lane

        v_agent[agent_fixed_mask] = 0.0
        v_lane[lane_fixed_mask] = 0.0

        return (
            e_agent,
            e_lane,
            z_agent,
            z_lane,
            v_agent,
            v_lane,
            agent_fixed_mask,
            lane_fixed_mask,
        )

    def _sample_t_scene_shortcut(self, batch_size: int, device: torch.device):
        """Shortcut branch t-sampling (Logit) forlow / SC-NormalI'm sorry."""
        mu = float(getattr(self.cfg_model, "meanflow_time_mu", -0.4))
        sigma = float(getattr(self.cfg_model, "meanflow_time_sigma", 1.0))
        rnd = torch.randn(batch_size, device=device)
        t = torch.sigmoid(rnd * sigma + mu)
        t = torch.clamp(t, 0.0, 1.0)
        return t

    def _flow_loss_shortcut(
        self,
        data,
        e_agent,
        e_lane,
        z_agent,
        z_lane,
        v_agent,
        v_lane,
        agent_fixed_mask,
        lane_fixed_mask,
    ):
        """BlowMatching part of Shortcut branch: s_s_t,t,d=0)v = z-e."""

        device = z_agent.device
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # scene-levelt Use Logit-NormalSample
        t_scene = self._sample_t_scene_shortcut(batch_size, device=device)
        t_agent = t_scene[agent_batch]
        t_lane = t_scene[lane_batch]

        t_agent_b = t_agent[:, None, None]
        t_lane_b = t_lane[:, None, None]

        # Straight path: x_t = (1 - t)*e + t*z (noise & data)
        x_agent_t = (1.0 - t_agent_b) * e_agent + t_agent_b * z_agent
        x_lane_t = (1.0 - t_lane_b) * e_lane + t_lane_b * z_lane

        x_agent_t[agent_fixed_mask] = z_agent[agent_fixed_mask]
        x_lane_t[lane_fixed_mask] = z_lane[lane_fixed_mask]

        if self.use_two_times:
            d_scene = torch.zeros_like(t_scene)
            d_agent = d_scene[agent_batch]
            d_lane = d_scene[lane_batch]
            s_agent_pred, s_lane_pred = self.model(
                x_agent_t,
                x_lane_t,
                data,
                t_agent,
                t_lane,
                unconditional=False,
                agent_h=d_agent,
                lane_h=d_lane,
            )
        else:
            s_agent_pred, s_lane_pred = self.model(
                x_agent_t,
                x_lane_t,
                data,
                t_agent,
                t_lane,
                unconditional=False,
            )

        agent_loss_vec = self.agent_loss_fn(s_agent_pred, v_agent, agent_batch)
        lane_loss_vec = self.lane_loss_fn(s_lane_pred, v_lane, lane_batch)

        agent_loss = agent_loss_vec.mean()
        lane_loss = lane_loss_vec.mean()
        loss = agent_loss + self.cfg.train.lane_weight * lane_loss

        return {
            "loss": loss,
            "agent_loss": agent_loss.detach(),
            "lane_loss": lane_loss.detach(),
        }

    def _endpoint_loss_shortcut(
        self,
        data,
        e_agent,
        e_lane,
        z_agent,
        z_lane,
        v_agent,
        v_lane,
        agent_fixed_mask,
        lane_fixed_mask,
    ):
        """Endpoint:s_s_(e, t=0, d=1)v=z-e of the Shortcut branch."""

        device = z_agent.device
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        t_scene = torch.zeros(batch_size, device=device)
        t_agent = t_scene[agent_batch]
        t_lane = t_scene[lane_batch]

        x_agent_0 = e_agent.clone()
        x_lane_0 = e_lane.clone()

        if self.use_two_times:
            d_scene = torch.ones_like(t_scene)
            d_agent = d_scene[agent_batch]
            d_lane = d_scene[lane_batch]
            s_agent_pred, s_lane_pred = self.model(
                x_agent_0,
                x_lane_0,
                data,
                t_agent,
                t_lane,
                unconditional=False,
                agent_h=d_agent,
                lane_h=d_lane,
            )
        else:
            s_agent_pred, s_lane_pred = self.model(
                x_agent_0,
                x_lane_0,
                data,
                t_agent,
                t_lane,
                unconditional=False,
            )

        agent_loss_vec = self.agent_loss_fn(s_agent_pred, v_agent, agent_batch)
        lane_loss_vec = self.lane_loss_fn(s_lane_pred, v_lane, lane_batch)

        agent_loss = agent_loss_vec.mean()
        lane_loss = lane_loss_vec.mean()
        loss = agent_loss + self.cfg.train.lane_weight * lane_loss

        return {
            "loss": loss,
            "agent_loss": agent_loss.detach(),
            "lane_loss": lane_loss.detach(),
        }

    def _self_consistency_loss_shortcut(
        self,
        data,
        e_agent,
        e_lane,
        z_agent,
        z_lane,
        v_agent,
        v_lane,
        agent_fixed_mask,
        lane_fixed_mask,
    ):
        """Self of Shortcut Branch-ConsistencyI don't know."""
        if self.sc_weight <= 0.0:
            zero = z_agent.new_tensor(0.0)
            return {"loss": zero, "agent_loss": zero, "lane_loss": zero}

        device = z_agent.device
        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # scene-level t
        t_scene = self._sample_t_scene_shortcut(batch_size, device=device)

        # To ensure t+ 2d < = 1
        u = torch.rand(batch_size, device=device)
        u = 1.0 - u ** 2
        max_d = (1.0 - t_scene) * 0.5
        d_scene = u * max_d

        if (d_scene <= 1e-6).all():
            zero = z_agent.new_tensor(0.0)
            return {"loss": zero, "agent_loss": zero, "lane_loss": zero}

        t_agent = t_scene[agent_batch]
        t_lane = t_scene[lane_batch]
        d_agent = d_scene[agent_batch]
        d_lane = d_scene[lane_batch]

        t_agent_b = t_agent[:, None, None]
        t_lane_b = t_lane[:, None, None]
        d_agent_b = d_agent[:, None, None]
        d_lane_b = d_lane[:, None, None]

        # x_t = (1 - t)*e + t*z (noise data)
        x_agent_t = (1.0 - t_agent_b) * e_agent + t_agent_b * z_agent
        x_lane_t = (1.0 - t_lane_b) * e_lane + t_lane_b * z_lane

        x_agent_t[agent_fixed_mask] = z_agent[agent_fixed_mask]
        x_lane_t[lane_fixed_mask] = z_lane[lane_fixed_mask]

        with torch.no_grad():
            if self.use_two_times:
                s_agent_1, s_lane_1 = self.model(
                    x_agent_t,
                    x_lane_t,
                    data,
                    t_agent,
                    t_lane,
                    unconditional=False,
                    agent_h=d_agent,
                    lane_h=d_lane,
                )
            else:
                s_agent_1, s_lane_1 = self.model(
                    x_agent_t,
                    x_lane_t,
                    data,
                    t_agent,
                    t_lane,
                    unconditional=False,
                )

            x_agent_mid = x_agent_t + d_agent_b * s_agent_1
            x_lane_mid = x_lane_t + d_lane_b * s_lane_1

            x_agent_mid[agent_fixed_mask] = x_agent_t[agent_fixed_mask]
            x_lane_mid[lane_fixed_mask] = x_lane_t[lane_fixed_mask]

            t_agent_next = t_agent + d_agent
            t_lane_next = t_lane + d_lane

            if self.use_two_times:
                s_agent_2, s_lane_2 = self.model(
                    x_agent_mid,
                    x_lane_mid,
                    data,
                    t_agent_next,
                    t_lane_next,
                    unconditional=False,
                    agent_h=d_agent,
                    lane_h=d_lane,
                )
            else:
                s_agent_2, s_lane_2 = self.model(
                    x_agent_mid,
                    x_lane_mid,
                    data,
                    t_agent_next,
                    t_lane_next,
                    unconditional=False,
                )

            v_agent_target = 0.5 * (s_agent_1 + s_agent_2)
            v_lane_target = 0.5 * (s_lane_1 + s_lane_2)

        if self.use_two_times:
            d2_scene = 2.0 * d_scene
            d2_agent = d2_scene[agent_batch]
            d2_lane = d2_scene[lane_batch]
            s_agent_big, s_lane_big = self.model(
                x_agent_t,
                x_lane_t,
                data,
                t_agent,
                t_lane,
                unconditional=False,
                agent_h=d2_agent,
                lane_h=d2_lane,
            )
        else:
            s_agent_big, s_lane_big = self.model(
                x_agent_t,
                x_lane_t,
                data,
                t_agent,
                t_lane,
                unconditional=False,
            )

        valid_agent = (~agent_fixed_mask) & (d_agent > 1e-6)
        valid_lane = (~lane_fixed_mask) & (d_lane > 1e-6)

        if not valid_agent.any() and not valid_lane.any():
            zero = z_agent.new_tensor(0.0)
            return {"loss": zero, "agent_loss": zero, "lane_loss": zero}

        agent_loss = z_agent.new_tensor(0.0)
        lane_loss = z_agent.new_tensor(0.0)

        if valid_agent.any():
            diff_a = s_agent_big[valid_agent] - v_agent_target[valid_agent].detach()
            agent_loss = 0.5 * (diff_a ** 2).mean()

        if valid_lane.any():
            diff_l = s_lane_big[valid_lane] - v_lane_target[valid_lane].detach()
            lane_loss = 0.5 * (diff_l ** 2).mean()

        loss = agent_loss + self.cfg.train.lane_weight * lane_loss

        return {
            "loss": loss,
            "agent_loss": agent_loss.detach(),
            "lane_loss": lane_loss.detach(),
        }

    # --------------Shortcut Total loss------------------
    def _loss_shortcut(self, data):
        """Shortcut-MeanFlow+ Endpoint + Self-ConsistencyI'm sorry."""
        self.num_updates += 1
        cur_step = int(self.num_updates.item())

        (
            e_agent,
            e_lane,
            z_agent,
            z_lane,
            v_agent,
            v_lane,
            agent_fixed_mask,
            lane_fixed_mask,
        ) = self._prepare_latent_pairs_shortcut(data)

        # Base step: Flow + Endpoint
        if (cur_step < self.shortcut_warmup_steps) or (
            torch.rand(1, device=z_agent.device).item() >= self.sc_ratio
        ):
            flow_dict = self._flow_loss_shortcut(
                data,
                e_agent,
                e_lane,
                z_agent,
                z_lane,
                v_agent,
                v_lane,
                agent_fixed_mask,
                lane_fixed_mask,
            )
            end_dict = self._endpoint_loss_shortcut(
                data,
                e_agent,
                e_lane,
                z_agent,
                z_lane,
                v_agent,
                v_lane,
                agent_fixed_mask,
                lane_fixed_mask,
            )
            zero = z_agent.new_tensor(0.0)
            sc_dict = {"loss": zero, "agent_loss": zero, "lane_loss": zero}
        else:
            zero = z_agent.new_tensor(0.0)
            flow_dict = {"loss": zero, "agent_loss": zero, "lane_loss": zero}
            end_dict = {"loss": zero, "agent_loss": zero, "lane_loss": zero}
            sc_dict = self._self_consistency_loss_shortcut(
                data,
                e_agent,
                e_lane,
                z_agent,
                z_lane,
                v_agent,
                v_lane,
                agent_fixed_mask,
                lane_fixed_mask,
            )

        warmup_end = max(self.shortcut_warmup_steps, 1000)
        alpha = min(1.0, cur_step / float(warmup_end))

        w_flow = self.flow_weight
        w_end = self.end_weight * alpha
        w_sc = self.sc_weight

        total_loss = (
            w_flow * flow_dict["loss"]
            + w_end * end_dict["loss"]
            + w_sc * sc_dict["loss"]
        )
        total_agent = (
            w_flow * flow_dict["agent_loss"]
            + w_end * end_dict["agent_loss"]
            + w_sc * sc_dict["agent_loss"]
        )
        total_lane = (
            w_flow * flow_dict["lane_loss"]
            + w_end * end_dict["lane_loss"]
            + w_sc * sc_dict["lane_loss"]
        )

        loss_dict = {
            "loss": total_loss,
            "agent_loss": total_agent,
            "lane_loss": total_lane,
            "loss_flow": flow_dict["loss"].detach(),
            "loss_end": end_dict["loss"].detach(),
            "loss_sc": sc_dict["loss"].detach(),
        }
        return loss_dict

    # ----------------------------------------------
    def loss(self, data):
        """General training interface."""
        if self.use_identity_branch:
            return self._meanflow_identity_loss(data)
        else:
            return self._loss_shortcut(data)

    # ------------------------- CFGAverage speed field - - - - - - -
    def _cfg_velocity(
        self,
        x_agent,
        x_lane,
        data,
        t_agent,
        t_lane,
        h_agent=None,
        h_lane=None,
    ):
        """Classifier-freeguided average speed fieldu_cfg(x,t,r)I don't know."""
        if self.use_two_times and (h_agent is not None) and (h_lane is not None):
            v_agent_cond, v_lane_cond = self.model(
                x_agent,
                x_lane,
                data,
                t_agent,
                t_lane,
                unconditional=False,
                agent_h=h_agent,
                lane_h=h_lane,
            )
            v_agent_uncond, v_lane_uncond = self.model(
                x_agent,
                x_lane,
                data,
                t_agent,
                t_lane,
                unconditional=True,
                agent_h=h_agent,
                lane_h=h_lane,
            )
        else:
            v_agent_cond, v_lane_cond = self.model(
                x_agent, x_lane, data, t_agent, t_lane, unconditional=False
            )
            v_agent_uncond, v_lane_uncond = self.model(
                x_agent, x_lane, data, t_agent, t_lane, unconditional=True
            )

        s = self.cfg.train.guidance_scale
        v_agent = v_agent_uncond + s * (v_agent_cond - v_agent_uncond)
        v_lane = v_lane_uncond + s * (v_lane_cond - v_lane_uncond)
        return v_agent, v_lane

    # -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    @torch.no_grad()
    def _sample_identity(
        self,
        agent_shape,
        lane_shape,
        data,
        device: str = "cuda",
        mode: str = "initial_scene",
        return_chain: bool = False,
    ):
        """Identity mode: use MeanFlow formulaz_r = z_t - (t)-r) u(z_t,r,t)Sampling from t=10."""

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        # t=1 initial: N(0, I)
        x_agent = torch.randn(agent_shape, device=device)

        if mode == "lane_conditioned":
            x_lane = data["lane"].latents[:, None, :].to(device)
        else:
            # do NOT temperature-scale for flow/meanflow
            x_lane = torch.randn(lane_shape, device=device)

        cond_agent_mask = None
        cond_lane_mask = None
        if mode == "train":
            cond_agent_mask = data["agent"].partition_mask.bool().to(device)
            cond_lane_mask = data["lane"].partition_mask.bool().to(device)
        elif mode == "inpainting":
            cond_lane_mask = data["lane"].mask.bool().to(device)
            cond_agent_mask = data["agent"].mask.bool().to(device)
        elif mode == "lane_conditioned":
            cond_lane_mask = torch.ones(x_lane.shape[0], dtype=torch.bool, device=device)

        # apply initial conditioning
        if cond_agent_mask is not None and hasattr(data["agent"], "latents"):
            x_agent[cond_agent_mask] = data["agent"].latents[cond_agent_mask].unsqueeze(1).to(device)
        if cond_lane_mask is not None and hasattr(data["lane"], "latents"):
            x_lane[cond_lane_mask] = data["lane"].latents[cond_lane_mask].unsqueeze(1).to(device)

        K = max(int(self.num_steps_eval), 1)
        t_values = torch.linspace(1.0, 0.0, steps=K + 1, device=device)

        chain = [] if return_chain else None

        for k in range(K):
            t_cur = t_values[k]
            t_next = t_values[k + 1]
            dt = t_cur - t_next  # >0

            t_scene_cur = torch.full((batch_size,), t_cur, device=device)
            t_scene_next = torch.full((batch_size,), t_next, device=device)

            t_agent_cur = t_scene_cur[agent_batch]
            t_lane_cur = t_scene_cur[lane_batch]

            if self.use_two_times:
                r_scene = t_scene_next
                h_scene = t_scene_cur - r_scene
                h_agent = h_scene[agent_batch]
                h_lane = h_scene[lane_batch]
                v_agent, v_lane = self._cfg_velocity(
                    x_agent, x_lane, data, t_agent_cur, t_lane_cur, h_agent=h_agent, h_lane=h_lane
                )
            else:
                v_agent, v_lane = self._cfg_velocity(
                    x_agent, x_lane, data, t_agent_cur, t_lane_cur, h_agent=None, h_lane=None
                )

            x_agent = x_agent - dt * v_agent
            x_lane = x_lane - dt * v_lane

            # re-apply conditioning
            if mode == "lane_conditioned" and hasattr(data["lane"], "latents"):
                x_lane = data["lane"].latents[:, None, :].to(device)

            if cond_agent_mask is not None and hasattr(data["agent"], "latents"):
                x_agent[cond_agent_mask] = data["agent"].latents[cond_agent_mask].unsqueeze(1).to(device)
            if cond_lane_mask is not None and hasattr(data["lane"], "latents") and mode != "lane_conditioned":
                x_lane[cond_lane_mask] = data["lane"].latents[cond_lane_mask].unsqueeze(1).to(device)

            if self.diffusion_clip is not None and self.diffusion_clip > 0:
                x_agent = torch.clamp(x_agent, -self.diffusion_clip, self.diffusion_clip)
                if mode != "lane_conditioned":
                    x_lane = torch.clamp(x_lane, -self.diffusion_clip, self.diffusion_clip)

            if return_chain:
                chain.append((x_agent[:, 0].detach().clone(), x_lane[:, 0].detach().clone()))

        if return_chain:
            return x_agent[:, 0], x_lane[:, 0], chain
        return x_agent[:, 0], x_lane[:, 0]

    # ---------------Shortcut model sampling----------------------------------------------------------------------Shortcut model sampling-----------------------------------------
    @torch.no_grad()
    def _sample_shortcut(
        self,
        agent_shape,
        lane_shape,
        data,
        device: str = "cuda",
        mode: str = "initial_scene",
        return_chain: bool = False,
    ):
        """Shortcut-MeanFlow few-stepsampler from t=0 § 1 (retention achieved)."""

        agent_batch = data["agent"].batch
        lane_batch = data["lane"].batch
        batch_size = data.batch_size

        x_agent = torch.randn(agent_shape, device=device)

        if mode == "lane_conditioned":
            x_lane = data["lane"].latents[:, None, :].to(device)
        else:
            # do NOT temperature-scale for flow/meanflow
            x_lane = torch.randn(lane_shape, device=device)

        cond_agent_mask = None
        cond_lane_mask = None

        if mode == "train":
            cond_agent_mask = data["agent"].partition_mask.bool().to(device)
            cond_lane_mask = data["lane"].partition_mask.bool().to(device)
        elif mode == "inpainting":
            cond_lane_mask = data["lane"].mask.bool().to(device)
            cond_agent_mask = data["agent"].mask.bool().to(device)
        elif mode == "lane_conditioned":
            cond_lane_mask = torch.ones(x_lane.shape[0], dtype=torch.bool, device=device)

        if cond_agent_mask is not None and hasattr(data["agent"], "latents"):
            x_agent[cond_agent_mask] = data["agent"].latents[cond_agent_mask].unsqueeze(1).to(device)
        if cond_lane_mask is not None and hasattr(data["lane"], "latents"):
            x_lane[cond_lane_mask] = data["lane"].latents[cond_lane_mask].unsqueeze(1).to(device)

        K = max(int(self.num_steps_eval), 1)
        t_values = torch.linspace(0.0, 1.0, steps=K + 1, device=device)

        chain = [] if return_chain else None

        for k in range(K):
            t_cur = t_values[k]
            t_next = t_values[k + 1]
            d = t_next - t_cur

            t_scene_cur = torch.full((batch_size,), t_cur, device=device)
            t_agent_cur = t_scene_cur[agent_batch]
            t_lane_cur = t_scene_cur[lane_batch]

            if self.use_two_times:
                d_scene = torch.full((batch_size,), d, device=device)
                d_agent = d_scene[agent_batch]
                d_lane = d_scene[lane_batch]
                v_agent, v_lane = self._cfg_velocity(
                    x_agent, x_lane, data, t_agent_cur, t_lane_cur, h_agent=d_agent, h_lane=d_lane
                )
            else:
                v_agent, v_lane = self._cfg_velocity(
                    x_agent, x_lane, data, t_agent_cur, t_lane_cur, h_agent=None, h_lane=None
                )

            x_agent = x_agent + d * v_agent
            x_lane = x_lane + d * v_lane

            if mode == "lane_conditioned" and hasattr(data["lane"], "latents"):
                x_lane = data["lane"].latents[:, None, :].to(device)

            if cond_agent_mask is not None and hasattr(data["agent"], "latents"):
                x_agent[cond_agent_mask] = data["agent"].latents[cond_agent_mask].unsqueeze(1).to(device)
            if cond_lane_mask is not None and hasattr(data["lane"], "latents") and mode != "lane_conditioned":
                x_lane[cond_lane_mask] = data["lane"].latents[cond_lane_mask].unsqueeze(1).to(device)

            if self.diffusion_clip is not None and self.diffusion_clip > 0:
                x_agent = torch.clamp(x_agent, -self.diffusion_clip, self.diffusion_clip)
                if mode != "lane_conditioned":
                    x_lane = torch.clamp(x_lane, -self.diffusion_clip, self.diffusion_clip)

            if return_chain:
                chain.append((x_agent[:, 0].detach().clone(), x_lane[:, 0].detach().clone()))

        if return_chain:
            return x_agent[:, 0], x_lane[:, 0], chain
        return x_agent[:, 0], x_lane[:, 0]

    # ----------------------------------------------------------------------------- a uniform sampling interface----------------------------------------------
    @torch.no_grad()
    def forward(self, data, mode: str = "initial_scene", return_chain: bool = False):
        """Generate samples with MeanFlow; interface matches LDM/FlowLDM."""
        agent_shape = data["agent"].x[:, None, :].shape
        lane_shape = data["lane"].x[:, None, :].shape
        return self.sample(
            agent_shape=agent_shape,
            lane_shape=lane_shape,
            data=data,
            device=data["agent"].x.device,
            mode=mode,
            return_chain=return_chain,
        )

    @torch.no_grad()
    def sample(
        self,
        agent_shape,
        lane_shape,
        data,
        device: str = "cuda",
        mode: str = "initial_scene",
        return_chain: bool = False,
    ):
        """Total sample interface: Select Identity or Shortcut branch based on trading_mode & energy_jvp.

        Returns the extra chain (long K) when return_chai=True, each element being (x_agent_step, x_lane_step) 2D length (N,D).
        """
        if self.use_identity_branch:
            return self._sample_identity(
                agent_shape=agent_shape,
                lane_shape=lane_shape,
                data=data,
                device=device,
                mode=mode,
                return_chain=return_chain,
            )
        else:
            return self._sample_shortcut(
                agent_shape=agent_shape,
                lane_shape=lane_shape,
                data=data,
                device=device,
                mode=mode,
                return_chain=return_chain,
            )