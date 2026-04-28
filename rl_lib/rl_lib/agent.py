from collections import OrderedDict
from collections.abc import Callable
from enum import Enum
import itertools
from typing import Tuple, Optional, Dict

import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
import torch.nn.functional as F

from rl_lib.utils import to_numpy, batch_to_torch


class ActorType(Enum):
    SINGLE_POLICY = 0
    MIX_OF_EXPERTS = 1
    POLICY_PER_SKILL = 2


def load_actor_from_file(
    file_path: str,
    device: str,
    in_dim: int,
    out_dim: int,
    hidden_dims: Tuple[int],
    activation: Callable,
):

    actor = Actor(in_dim, out_dim, hidden_dims, activation, device)
    with open(file_path, "rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return actor


def load_depth_from_file(
    file_path: str,
    device: str,
    actor_type: ActorType,
    actor_type_kwargs: Dict,
    in_dim: int,
    out_dim: int,
    hidden_dims: Tuple[int],
    activation: Callable,
    depth_output_size: int,
    n_proprio: int,
    past_obs_for_depth_encoder: int,
    encoding_tricks: bool = True,
    priv_encoder_output_dim: int = None,
    n_priv_explicit: int = None,
    estimator_hidden_dim: int = None,
    history_len: int = None,
    dropout: bool = False,
    dropout_prob: float = 0.2,
):

    if actor_type == ActorType.MIX_OF_EXPERTS:
        scan_idx = actor_type_kwargs["scan_idx"]
        scan_len = actor_type_kwargs["scan_len"]
        moe_num_experts = actor_type_kwargs["moe_n_experts"]
        moe_top_k = actor_type_kwargs["moe_top_k"]
        gate_noise_with_x = actor_type_kwargs["gate_noise_with_x"]
        moe_layer_idx = actor_type_kwargs["moe_layer_idx"]

        depth_actor = MoEActor(
            in_dim,
            out_dim,
            hidden_dims,
            scan_idx,
            scan_len,
            activation,
            device,
            moe_num_experts,
            moe_top_k,
            gate_noise_with_x=gate_noise_with_x,
            moe_layer_idx=moe_layer_idx,
            dropout=dropout,
            dropout_prob=dropout_prob,
        )
    elif actor_type == ActorType.SINGLE_POLICY:
        depth_actor = Actor(
            in_dim,
            out_dim,
            hidden_dims,
            activation,
            device,
            use_many_params_std=actor_type_kwargs["use_many_params_std"],
            dropout=False,
            dropout_prob=dropout_prob,
        )
    elif actor_type == ActorType.POLICY_PER_SKILL:
        env_keys = actor_type_kwargs["env_keys"]
        depth_actor = MultiSkillActor(
            in_dim,
            out_dim,
            hidden_dims,
            activation,
            device,
            env_keys,
            dropout=dropout,
            dropout_prob=dropout_prob,
        )
    else:
        raise NotImplementedError

    with open(file_path, "rb") as f:
        payload = torch.load(f, map_location=device, weights_only=False)

    depth_backbone = DepthOnlyFCBackbone58x87(n_proprio, depth_output_size).to(device)
    depth_encoder = RecurrentDepthBackbone(
        depth_backbone, n_proprio + n_proprio * past_obs_for_depth_encoder
    ).to(device)

    depth_actor.load_state_dict(payload["depth_actor_state_dict"])
    depth_backbone.load_state_dict(payload["depth_backbone"])
    depth_encoder.load_state_dict(payload["depth_encoder"])

    depth_actor.eval()
    depth_backbone.eval()
    depth_encoder.eval()
    if not encoding_tricks:
        return depth_actor, depth_backbone, depth_encoder, ()

    # now we need state estimator and hist estimator

    estimator = get_sequential_model(
        n_proprio, n_priv_explicit, estimator_hidden_dim, activation, False
    )
    estimator = estimator.to(device)

    history_encoder = StateHistoryEncoder(
        activation, n_proprio, history_len, priv_encoder_output_dim
    )
    history_encoder = history_encoder.to(device)

    history_encoder.load_state_dict(payload["history_encoder_state_dict"])
    estimator.load_state_dict(payload["estimator_state_dict"])
    estimator.eval()
    history_encoder.eval()

    return depth_actor, depth_backbone, depth_encoder, (estimator, history_encoder)


def get_sequential_model(
    in_dim: int,
    out_dim: int,
    hidden_layer_dims: Tuple[int],
    activation: Callable,
    activate_final_layer: bool,
    dropout: bool = False,
    dropout_prob: float = 0.2,
):

    layer_dict = OrderedDict()
    layer_ct = 0
    layer_dict[f"linear_{layer_ct}"] = nn.Linear(in_dim, hidden_layer_dims[0])
    # layer_dict[f"activation_{layer_ct}"] = activation()
    layer_ct += 1

    for i in range(len(hidden_layer_dims)):

        is_final = i == len(hidden_layer_dims) - 1
        add_dropout = False
        if is_final:
            starting_dim = hidden_layer_dims[i]
            ending_dim = out_dim
        else:
            starting_dim = hidden_layer_dims[i]
            ending_dim = hidden_layer_dims[i + 1]
            add_dropout = dropout

        layer_prefix = "hidden_" if not is_final else "final_"
        layer_dict[f"{layer_prefix}linear_{layer_ct}"] = nn.Linear(
            starting_dim, ending_dim
        )
        if not is_final or activate_final_layer:
            layer_dict[f"{layer_prefix}_activation_{layer_ct}"] = activation()
        if add_dropout:
            layer_dict[f"{layer_prefix}_dropout_{layer_ct}"] = nn.Dropout(
                p=dropout_prob
            )
        layer_ct += 1

    policy = nn.Sequential(layer_dict)

    return policy


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


class StateHistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size):
        super(StateHistoryEncoder, self).__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 10

        self.encoder = nn.Sequential(
            nn.Linear(input_size, 3 * channel_size),
            self.activation_fn(),
        )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=8,
                    stride=4,
                ),
                self.activation_fn(),
                nn.Conv1d(
                    in_channels=2 * channel_size,
                    out_channels=channel_size,
                    kernel_size=5,
                    stride=1,
                ),
                self.activation_fn(),
                nn.Conv1d(
                    in_channels=channel_size,
                    out_channels=channel_size,
                    kernel_size=5,
                    stride=1,
                ),
                self.activation_fn,
                nn.Flatten(),
            )
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=4,
                    stride=2,
                ),
                self.activation_fn(),
                nn.Conv1d(
                    in_channels=2 * channel_size,
                    out_channels=channel_size,
                    kernel_size=2,
                    stride=1,
                ),
                self.activation_fn(),
                nn.Flatten(),
            )
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(
                    in_channels=3 * channel_size,
                    out_channels=2 * channel_size,
                    kernel_size=6,
                    stride=2,
                ),
                self.activation_fn(),
                nn.Conv1d(
                    in_channels=2 * channel_size,
                    out_channels=channel_size,
                    kernel_size=4,
                    stride=2,
                ),
                self.activation_fn(),
                nn.Flatten(),
            )
        else:
            raise (ValueError("tsteps must be 10, 20 or 50"))

        self.linear_output = nn.Sequential(
            nn.Linear(channel_size * 3, output_size), self.activation_fn()
        )

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1]))
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class RecurrentDepthBackbone(nn.Module):
    def __init__(self, base_backbone, n_proprio: int) -> None:
        super().__init__()
        activation = nn.ELU()
        last_activation = nn.Tanh()
        self.base_backbone = base_backbone
        self.combination_mlp = nn.Sequential(
            nn.Linear(32 + n_proprio, 128), activation, nn.Linear(128, 32)
        )
        self.rnn = nn.GRU(input_size=32, hidden_size=512, batch_first=True)
        self.output_mlp = nn.Sequential(
            # scan output dim, plus priv target yaw obs
            nn.Linear(512, 32 + 2),
            last_activation,
        )
        self.hidden_states = None

    def forward(self, depth_image, proprioception):
        depth_image = self.base_backbone(depth_image)
        depth_latent = self.combination_mlp(
            torch.cat((depth_image, proprioception), dim=-1)
        )
        # depth_latent = self.base_backbone(depth_image)
        depth_latent, self.hidden_states = self.rnn(
            depth_latent[:, None, :], self.hidden_states
        )
        depth_latent_and_yaw = self.output_mlp(depth_latent.squeeze(1))

        return depth_latent_and_yaw[:, :-2], 1.5 * depth_latent_and_yaw[:, -2:]

    def detach_hidden_states(self):
        self.hidden_states = self.hidden_states.detach().clone()


class DepthOnlyFCBackbone58x87(nn.Module):
    def __init__(
        self, prop_dim, scandots_output_dim, output_activation=None, num_frames=1
    ):
        super().__init__()

        self.num_frames = num_frames
        activation = nn.ELU()
        self.image_compression = nn.Sequential(
            # [1, 58, 87]
            nn.Conv2d(in_channels=self.num_frames, out_channels=32, kernel_size=5),
            # [32, 54, 83]
            nn.MaxPool2d(kernel_size=2, stride=2),
            # [32, 27, 41]
            activation,
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            activation,
            nn.Flatten(),
            # [32, 25, 39]
            nn.Linear(64 * 25 * 39, 128),
            activation,
            nn.Linear(128, scandots_output_dim),
        )

        if output_activation == "tanh":
            self.output_activation = nn.Tanh()
        else:
            self.output_activation = activation

    def forward(self, images: torch.Tensor):
        images_compressed = self.image_compression(images.unsqueeze(1))
        latent = self.output_activation(images_compressed)

        return latent


class Actor(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int],
        activation: Callable,
        device: str,
        logstd_init: float = 0.0,
        use_many_params_std: bool = True,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = hidden_dims
        self.activation = activation
        self.logstd_init = logstd_init
        self.use_many_params_std = use_many_params_std
        self.dropout = dropout
        self.dropout_prob = dropout_prob

        self.device = device

        self.n_std = out_dim if use_many_params_std else 1
        self.param_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(self.n_std, dtype=torch.float32, device=self.device)
                )
            ]
        )

        self.logstd = self.param_list[0]
        with torch.no_grad():
            self.logstd.data.fill_(logstd_init)

        self.policy = get_sequential_model(
            in_dim,
            out_dim,
            hidden_dims,
            activation,
            False,
            dropout=dropout,
            dropout_prob=dropout_prob,
        )
        self.policy.to(self.device)
        # self.apply(weight_init)

    def forward(self, obs):
        mu = self.policy(obs)
        std = torch.exp(self.logstd)
        normal_distributions = Normal(mu, std)
        return (
            normal_distributions,
            normal_distributions.mean,
            normal_distributions.stddev,
        )


class Critic(nn.Module):
    def __init__(
        self,
        n_obs: int,
        hidden_dims: Tuple[int],
        activation: Callable,
        device: str,
        num_samples: int = 2,
    ):
        super().__init__()

        self.device = device
        # self.num_critics = num_critics
        self.num_samples = num_samples

        # critics = []
        # for i in range(num_critics):
        #     seq = get_sequential_model(n_obs, 1, hidden_dims, activation, False)
        #     critics.append(seq)

        # self.critics = nn.ModuleList(critics)
        # self.critics.to(self.device)

        self.critic = get_sequential_model(n_obs, 1, hidden_dims, activation, False)
        self.critic.to(self.device)
        # self.apply(weight_init)

    # def forward(self, obs):
    #     return [critic(obs) for critic in self.critics]

    # def get_value(self, obs):

    #     critic_vals = self.forward(obs)
    #     num_critics = len(critic_vals)

    #     critic_indices = torch.randint(high=num_critics, size=(self.num_samples,)).to(self.device)

    #     prev_min = critic_vals[critic_indices[0]]
    #     for i in range(1, self.num_samples):
    #         prev_min = torch.minimum(prev_min, critic_vals[critic_indices[i]])
    #     return prev_min

    def forward(self, obs):
        return self.critic(obs)

    def get_value(self, obs):
        return self.critic(obs)


TARGET_EXPERT = 0


class MoEActor(nn.Module):
    def __init__(
        self,
        in_dim_experts: int,
        out_dim: int,
        hidden_dims: Tuple[int],
        scan_idx: int,
        scan_len: int,
        activation: Callable,
        device: str,
        num_experts: int,
        top_k: int,
        logstd_init: float = 0.0,
        noise_mat_init: float = 1.0,
        use_many_params_std: bool = False,
        gate_noise_with_x: bool = False,
        moe_layer_idx: int = 1,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ):
        super().__init__()

        self.device = device
        self.out_dim = out_dim

        self.num_experts = num_experts
        self.top_k = top_k

        self.scan_idx = scan_idx
        self.scan_len = scan_len

        self.moe_layer_idx = moe_layer_idx
        self.hidden_dims = hidden_dims
        self.gate_noise_with_x = gate_noise_with_x
        self.in_dim_experts = in_dim_experts
        self.activation_callable = activation
        self.do_dropout = dropout
        self.dropout_prob = dropout_prob

        self.use_many_params_std = use_many_params_std

        in_dim_gating = self.hidden_dims[self.moe_layer_idx]
        # self.gate_network = get_sequential_model(
        #     in_dim_gating, num_experts, moe_hidden_dims, activation, False
        # ).to(self.device)

        # first is gate_mat, second is noise_mat
        # for some reason nn.Parameter doesn't register whereas nn.ParameterList does...

        self.size_noise_mat = (
            1 if not self.gate_noise_with_x else (in_dim_gating, self.num_experts)
        )

        self.param_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(
                        in_dim_gating,
                        self.num_experts,
                        device=self.device,
                        dtype=torch.float32,
                    )
                ),
                nn.Parameter(
                    torch.zeros(
                        self.size_noise_mat,
                        device=self.device,
                        dtype=torch.float32,
                    )
                ),
            ]
        )
        self.gate_mat = self.param_list[0]
        self.noise_mat = self.param_list[1]

        with torch.no_grad():
            self.param_list[1].data.fill_(noise_mat_init)

        self.soft_plus = nn.Softplus()
        self.soft_max = nn.Softmax(dim=1)

        self.net_to_moe_layer = get_sequential_model(
            in_dim_experts,
            hidden_dims[self.moe_layer_idx],
            hidden_dims[: self.moe_layer_idx],
            activation,
            True,
            dropout=dropout,
            dropout_prob=dropout_prob,
        ).to(self.device)

        self.moe_layers = nn.ModuleList(
            [
                nn.Linear(
                    hidden_dims[self.moe_layer_idx], hidden_dims[self.moe_layer_idx + 1]
                ).to(self.device)
                for _ in range(self.num_experts)
            ]
        )

        self.moe_activation = activation()
        self.moe_dropout = dropout
        self.moe_dropout_layer = nn.Dropout(p=dropout_prob)

        self.final_layers = get_sequential_model(
            hidden_dims[self.moe_layer_idx + 1],
            out_dim,
            hidden_dims[self.moe_layer_idx + 1 :],
            activation,
            False,
            dropout=dropout,
            dropout_prob=dropout_prob,
        ).to(self.device)

        self.n_std = out_dim if use_many_params_std else 1

        self.logstd_params_list = nn.ParameterList(
            [
                nn.Parameter(
                    torch.zeros(self.n_std, dtype=torch.float32, device=self.device)
                )
            ]
        )

        self.logstd = self.logstd_params_list[0]
        with torch.no_grad():
            self.logstd.data.fill_(logstd_init)

        self.weights = None

    def forward(self, obs):

        train = self.training

        # Shazeer, Noam, et al. "Outrageously large neural networks: The sparsely-gated mixture-of-experts layer." arXiv preprint arXiv:1701.06538 (2017).
        noise_dims = (obs.shape[0], self.num_experts)

        int_out = self.net_to_moe_layer(obs)

        h_x = int_out @ self.gate_mat
        # if train we add some noise to gating network to encourage exploration

        if train:
            sample = torch.randn(
                (h_x.shape[0], self.num_experts),
                dtype=torch.float32,
                device=self.device,
            )
            if self.gate_noise_with_x:
                mult_bit = self.soft_plus(int_out @ self.noise_mat)
            else:
                mult_bit = self.soft_plus(
                    torch.tile(self.noise_mat, dims=(h_x.shape[0], self.num_experts))
                )

            h_x += sample * mult_bit

        # argmax which topk uses is not differentiable
        # we seek to learn better gating so we need the gradients to flow for experts if topk == 1
        if self.top_k == 1:

            moe_outs = [self.moe_layers[i](int_out) for i in range(self.num_experts)]

            gates = F.gumbel_softmax(h_x, tau=0.1, hard=True)
            combined_moe_outs = torch.zeros(
                (
                    obs.shape[0],
                    self.hidden_dims[self.moe_layer_idx + 1],
                ),
                dtype=torch.float32,
                device=self.device,
            )

            for i in range(len(moe_outs)):
                expert_weight = gates[:, i]
                # 6000x128, * 6000x5
                combined_moe_outs += moe_outs[i] * expert_weight[:, None]

        else:
            # find top k
            ret = torch.topk(h_x, self.top_k, dim=1)

            top_k_gates = self.soft_max(ret.values)
            zeros = torch.zeros_like(h_x, requires_grad=True)
            gates = zeros.scatter(1, ret.indices, top_k_gates)

            # https://github.com/YeonwooSung/Pytorch_mixture-of-experts/blob/main/moe.py

            # sort experts
            sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)

            # drop indices
            _, expert_index = sorted_experts.split(1, dim=1)
            # get according batch index for each expert
            batch_index = sorted_experts[index_sorted_experts[:, 1], 0]

            # calculate num samples that each expert gets
            part_sizes = list((gates > 0).cpu().detach().sum(0).numpy())

            # expand gates to match with batch_index
            gates_exp = gates[batch_index.flatten()]
            nonzero_gates = torch.gather(gates_exp, 1, expert_index)

            # expand according to batch index so we can just split by _part_sizes
            inp_exp = int_out[batch_index].squeeze(1)
            inputs = torch.split(inp_exp, part_sizes, dim=0)

            moe_outs = [self.moe_layers[i](inputs[i]) for i in range(self.num_experts)]

            stitched_moe_outs = torch.cat(moe_outs, 0)

            stitched_moe_outs = stitched_moe_outs.mul(nonzero_gates)

            zeros = torch.zeros(
                gates.size(0),
                moe_outs[-1].size(1),
                requires_grad=True,
                device=self.device,
            )
            # combine samples that have been processed by the same k experts
            combined_moe_outs = zeros.index_add(
                0, batch_index.to(self.device), stitched_moe_outs.float()
            )

        combined_moe_outs = self.moe_activation(combined_moe_outs)
        if self.moe_dropout:
            combined_moe_outs = self.moe_dropout_layer(combined_moe_outs)

        combined_mus = self.final_layers(combined_moe_outs)

        std = torch.exp(self.logstd)

        self.weights = gates

        normal_distributions = Normal(combined_mus, std)
        return (
            normal_distributions,
            normal_distributions.mean,
            normal_distributions.stddev,
        )

    def forward_fast(self, obs):

        assert obs.shape[0] == 1

        int_out = self.net_to_moe_layer(obs)

        h_x = int_out @ self.gate_mat

        if self.use_many_params_std:
            raise NotImplementedError

        # find top k
        ret = torch.topk(h_x, self.top_k, dim=1)

        top_k_gates = self.soft_max(ret.values)

        combined_moe_outs = torch.zeros(
            (1, self.hidden_dims[self.moe_layer_idx + 1]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        exp_fwds = torch.zeros(
            (self.top_k, self.hidden_dims[self.moe_layer_idx + 1]),
            dtype=torch.float32,
            device=self.device,
            requires_grad=False,
        )

        for i in range(self.top_k):
            exp_id = ret.indices[0][i]
            exp_fwds[i, :] = self.moe_layers[exp_id](int_out)

        for i in range(self.top_k):
            fwd = exp_fwds[i]
            combined_moe_outs += top_k_gates[0][i] * fwd

        combined_moe_outs = self.moe_activation(combined_moe_outs)

        combined_mus = self.final_layers(combined_moe_outs)
        std = torch.exp(self.logstd)
        normal_distributions = Normal(combined_mus, std)
        return (
            normal_distributions,
            combined_mus,
            std,
        )


class ScanEncoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int],
        activation: Callable,
        device: str,
    ):
        super().__init__()

        self.device = device

        self.encoder = get_sequential_model(
            in_dim, out_dim, hidden_dims, activation, False
        )
        self.encoder.to(self.device)
        # self.apply(weight_init)

    def forward(self, obs):
        return self.encoder(obs)


class MultiSkillActor(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dims: Tuple[int],
        activation: Callable,
        device: str,
        env_keys: Tuple,
        logstd_init: float = 0.0,
        use_many_params_std: bool = True,
        dropout: bool = False,
        dropout_prob: bool = 0.2,
    ):
        super().__init__()

        self.device = device

        self.out_dim = out_dim
        self.use_many_params_std = use_many_params_std

        self.env_keys = env_keys
        self.env_key_to_idx = {
            k: i for i, k in zip(range(len(self.env_keys)), self.env_keys)
        }

        actors = []

        for i in range(len(self.env_keys)):
            actors.append(
                Actor(
                    in_dim,
                    out_dim,
                    hidden_dims,
                    activation,
                    device,
                    logstd_init=logstd_init,
                    use_many_params_std=use_many_params_std,
                    dropout=dropout,
                    dropout_prob=dropout_prob,
                )
            )

        self.actor_list = nn.ModuleList(actors)

    def forward(self, obs, env_types):
        assert (
            obs.shape[0] == env_types.shape[0]
        ), f"obs: {obs.shape}, env_types {env_types.shape}"
        assert len(env_types.shape) == 1

        means = torch.zeros((obs.shape[0], self.out_dim), device=self.device)
        stddevs = torch.zeros((obs.shape[0], self.out_dim), device=self.device)

        for key in self.env_keys:
            key_idx = self.env_key_to_idx[key]

            actor = self.actor_list[key_idx]

            _, means_a, stddevs_a = actor(obs[env_types == key, :])

            means[env_types == key, :] = means_a
            stddevs[env_types == key, :] = stddevs_a

        normal_distributions = Normal(means, stddevs)

        actions = normal_distributions.sample()
        log_probs = normal_distributions.log_prob(actions)

        return actions, log_probs, means, stddevs, normal_distributions


class ACAgent:
    def __init__(
        self,
        n_obs,
        n_action,
        n_critic_obs,
        device,
        hidden_layers_actor,
        hidden_layers_critic,
        activation,
        actor_type: ActorType,
        moe_num_experts: int = 6,
        moe_top_k: int = 3,
        scan_idx: int = 48,
        scan_len: int = 32,
        moe_noise_mat_init: float = 1.0,
        env_keys: Tuple[int] = (0),
        logstd_init: float = 1.0,
        use_many_params_std: bool = True,
        gate_noise_with_x: bool = False,
        moe_layer_idx: int = 1,
        dropout: bool = False,
        dropout_prob: float = 0.2,
    ):
        self.device = device
        self.actor_type = actor_type
        self.n_action = n_action
        self.use_many_params_std = use_many_params_std
        # models

        if actor_type == ActorType.MIX_OF_EXPERTS:
            self.actor = MoEActor(
                n_obs,
                n_action,
                hidden_layers_actor,
                scan_idx,
                scan_len,
                activation,
                device,
                moe_num_experts,
                moe_top_k,
                logstd_init=logstd_init,
                noise_mat_init=moe_noise_mat_init,
                use_many_params_std=use_many_params_std,
                gate_noise_with_x=gate_noise_with_x,
                moe_layer_idx=moe_layer_idx,
                dropout=dropout,
                dropout_prob=dropout_prob,
            )
        elif actor_type == ActorType.SINGLE_POLICY:
            self.actor = Actor(
                n_obs,
                n_action,
                hidden_layers_actor,
                activation,
                device,
                logstd_init=logstd_init,
                use_many_params_std=use_many_params_std,
                dropout=dropout,
                dropout_prob=dropout_prob,
            )
        elif actor_type == ActorType.POLICY_PER_SKILL:
            self.actor = MultiSkillActor(
                n_obs,
                n_action,
                hidden_layers_actor,
                activation,
                device,
                env_keys,
                logstd_init=logstd_init,
                use_many_params_std=use_many_params_std,
                dropout=dropout,
                dropout_prob=dropout_prob,
            )
        else:
            raise NotImplementedError

        self.critic = Critic(n_critic_obs, hidden_layers_critic, activation, device)

        self.training = False

    def set_train(self, training: bool = True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, env_types: Optional = None):
        if self.actor_type == ActorType.POLICY_PER_SKILL:
            actions, log_probs, means, stddevs, dists = self.actor(obs, env_types)
        elif self.actor_type in [ActorType.SINGLE_POLICY, ActorType.MIX_OF_EXPERTS]:
            dists, means, stddevs = self.actor(obs)
            actions = dists.sample()
            log_probs = dists.log_prob(actions)
        else:
            raise NotImplementedError
        # , ActorType.MIX_OF_EXPERTS log probs from normal dist
        if self.actor_type in [
            ActorType.SINGLE_POLICY,
            ActorType.POLICY_PER_SKILL,
            ActorType.MIX_OF_EXPERTS,
        ]:
            log_probs = log_probs.sum(dim=-1)
        return actions, log_probs, means, stddevs, dists

    def get_value(self, critic_obs):
        return self.critic.get_value(critic_obs)

    def calc_advantages(self, last_values, dones, values, rewards, gamma, lam):

        num_steps = dones.shape[0]
        returns = torch.zeros(rewards.shape, device=self.device)

        advantage = 0.0
        for step in reversed(range(num_steps)):
            if step == num_steps - 1:
                next_values = last_values
            else:
                next_values = values[step + 1]

            next_is_not_terminal = 1.0 - dones[step].float()
            delta = (
                rewards[step]
                + next_is_not_terminal * gamma * next_values
                - values[step]
            )
            advantage = delta + next_is_not_terminal * gamma * lam * advantage
            returns[step] = advantage + values[step]

        # Compute and normalize the advantages
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages


if __name__ == "__main__":

    actor = MoEActor(
        32, 4, [64, 32], 0, 4, nn.ELU, "cuda:0", 5, 2, 0.0, 1.0, False
    )
    actor.train()
    print(actor.parameters)
    print(actor.gate_mat)
    print(id(actor.gate_mat))

    (
        normal_distributions,
        mean,
        stddev,
    ) = actor(torch.randn(5, 32, device="cuda:0"))

    loss = torch.sum(mean - 4)
    loss.backward()
    print(actor.noise_mat.grad)
