# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings

import torch
import torch.distributed
from codetiming import Timer
from omegaconf import DictConfig, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_tokenizer
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from .rubric_generator import generate_progress_rubric

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def create_device_mesh(world_size, fsdp_size):
    if fsdp_size < 0 or fsdp_size >= world_size:
        device_mesh = init_device_mesh('cuda', mesh_shape=(world_size,), mesh_dim_names=['fsdp'])
    else:
        raise ValueError(
            'HSDP is not supported yet because it produces incorrect results for now. Please set fsdp_size=-1')
        assert world_size % fsdp_size == 0
        device_mesh = init_device_mesh('cuda',
                                       mesh_shape=(world_size // fsdp_size, fsdp_size),
                                       mesh_dim_names=['ddp', 'fsdp'])
    return device_mesh


def get_sharding_strategy(device_mesh):
    from torch.distributed.fsdp import ShardingStrategy
    if device_mesh.ndim == 1:
        sharding_strategy = ShardingStrategy.FULL_SHARD
    elif device_mesh.ndim == 2:
        sharding_strategy = ShardingStrategy.HYBRID_SHARD
    else:
        raise NotImplementedError(f"Get device mesh ndim={device_mesh.ndim}, but only support 1 or 2")
    return sharding_strategy


class ActorRolloutRefWorker(Worker):
    """
    This worker can be instantiated as a standalone actor or a standalone rollout or a standalone reference policy
    or a hybrid engine based on the config.rollout
    """

    def __init__(self, config: DictConfig, role: str):
        super().__init__()
        self.config = config
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")

        # build device mesh for FSDP
        world_size = torch.distributed.get_world_size()
        # TODO(sgm): support FSDP hybrid shard for larger model
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=self.config.actor.fsdp_config.fsdp_size)

        # build device mesh for Ulysses Sequence Parallel
        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.actor.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.role = role
        assert self.role in ['actor', 'rollout', 'ref', 'actor_rollout', 'actor_rollout_ref']

        self._is_actor = self.role in ['actor', 'actor_rollout', 'actor_rollout_ref']
        self._is_rollout = self.role in ['rollout', 'actor_rollout', 'actor_rollout_ref']
        self._is_ref = self.role in ['ref', 'actor_rollout_ref']

        self._is_offload_param = False
        self._is_offload_optimizer = False
        if self._is_actor:
            self._is_offload_param = self.config.actor.fsdp_config.get('param_offload', False)
            self._is_offload_optimizer = self.config.actor.fsdp_config.get('optimizer_offload', False)
        elif self._is_ref:
            # TODO: it seems that manual offload is slowly than FSDP offload
            self._is_offload_param = self.config.ref.fsdp_config.get('param_offload', False)

        # normalize config
        if self._is_actor:
            self.config.actor.ppo_mini_batch_size *= self.config.rollout.n
            self.config.actor.ppo_mini_batch_size //= (self.device_mesh.shape[0] // self.ulysses_sequence_parallel_size)
            # micro bsz
            if self.config.actor.ppo_micro_batch_size is not None:
                self.config.actor.ppo_micro_batch_size //= (self.device_mesh.shape[0] //
                                                            self.ulysses_sequence_parallel_size)
                self.config.actor.ppo_micro_batch_size_per_gpu = self.config.actor.ppo_micro_batch_size
                assert self.config.actor.ppo_mini_batch_size % self.config.actor.ppo_micro_batch_size_per_gpu == 0, \
                    f'normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}'
                assert self.config.actor.ppo_mini_batch_size // self.config.actor.ppo_micro_batch_size_per_gpu > 0, \
                    f'normalized ppo_mini_batch_size {self.config.actor.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.actor.ppo_micro_batch_size_per_gpu}'

        # normalize rollout config
        if self._is_rollout and self.config.rollout.log_prob_micro_batch_size is not None:
            self.config.rollout.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                               self.ulysses_sequence_parallel_size)
            self.config.rollout.log_prob_micro_batch_size_per_gpu = self.config.rollout.log_prob_micro_batch_size
        # normalize ref config
        if self._is_ref and self.config.ref.log_prob_micro_batch_size is not None:
            self.config.ref.log_prob_micro_batch_size //= (self.device_mesh.shape[0] //
                                                           self.ulysses_sequence_parallel_size)
            self.config.ref.log_prob_micro_batch_size_per_gpu = self.config.ref.log_prob_micro_batch_size

    def _build_model_optimizer(self,
                               model_path,
                               fsdp_config,
                               optim_config,
                               override_model_config,
                               use_remove_padding=False,
                               enable_gradient_checkpointing=False,
                               trust_remote_code=False,
                               use_liger=False,
                               role='actor'):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
        from transformers import AutoConfig, AutoModelForCausalLM

        from verl.utils.model import (
            get_generation_config,
            print_model_size,
            update_model_config,
        )
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ['actor', 'ref']

        log_gpu_memory_usage('Before init from HF AutoModel', logger=logger)
        local_path = copy_to_local(model_path)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get('model_dtype', None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)

        self.generation_config = get_generation_config(local_path, trust_remote_code=trust_remote_code)

        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(actor_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(actor_model_config, verbose=True)

        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_model_config)
        update_model_config(actor_model_config, override_config_kwargs=override_config_kwargs)
        if self.rank == 0:
            print(f'Model config after override: {actor_model_config}')

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(use_meta_tensor=not actor_model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            actor_module = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                torch_dtype=torch_dtype,
                                                                config=actor_model_config,
                                                                attn_implementation='flash_attention_2',
                                                                trust_remote_code=trust_remote_code)
            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )
                _apply_liger_kernel_to_instance(model=actor_module)

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            if enable_gradient_checkpointing:
                actor_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage('After init from HF AutoModel', logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=actor_module, config=fsdp_config.get('wrap_policy', None))

        if self._is_rollout and self.config.rollout.name == 'hf':
            # TODO(zhangchi.usc1992, shengguangming) fix me. Current, auto_wrap_policy causes HFRollout to hang in Gemma
            auto_wrap_policy = None

        # print(f'wrap_policy: {auto_wrap_policy}')

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # TODO: add transformer policy
        # We force reference policy to use CPUOffload to save memory.
        # We force turn off CPUOffload for actor because it causes incorrect results when using grad accumulation
        cpu_offload = None if role == 'actor' else CPUOffload(offload_params=True)
        actor_module_fsdp = FSDP(
            actor_module,
            cpu_offload=cpu_offload,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            mixed_precision=mixed_precision,
            sync_module_states=True,
            device_mesh=self.device_mesh,
            forward_prefetch=False)

        log_gpu_memory_usage('After Actor FSDP init', logger=logger)

        # TODO: add more optimizer args into config
        if role == 'actor':
            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            actor_optimizer = optim.AdamW(actor_module_fsdp.parameters(),
                                          lr=optim_config.lr,
                                          betas=optim_config.get('betas', (0.9, 0.999)),
                                          weight_decay=optim_config.get('weight_decay', 1e-2))

            total_steps = optim_config.get('total_training_steps', 0)
            num_warmup_steps_ratio = optim_config.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            actor_lr_scheduler = get_constant_schedule_with_warmup(optimizer=actor_optimizer,
                                                                   num_warmup_steps=num_warmup_steps)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        log_gpu_memory_usage('After actor optimizer init', logger=logger)

        return actor_module_fsdp, actor_optimizer, actor_lr_scheduler, actor_model_config

    def _build_rollout(self):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert self.world_size % infer_tp == 0, f'rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}'
        rollout_device_mesh = init_device_mesh('cuda', mesh_shape=(dp, infer_tp), mesh_dim_names=['dp', 'infer_tp'])

        if self.config.rollout.name == 'hf':
            from verl.workers.rollout import HFRollout
            from verl.workers.sharding_manager import BaseShardingManager
            rollout = HFRollout(module=self.actor_module_fsdp, config=self.config.rollout)
            rollout_sharding_manager = BaseShardingManager()
            # TODO: a sharding manager that do nothing?
        elif self.config.rollout.name == 'vllm':
            if self.config.rollout.use_fire_sampling:
                from verl.workers.rollout.vllm_rollout import (
                    FIREvLLMRollout as vLLMRollout,
                )
                from verl.workers.rollout.vllm_rollout import vllm_mode
            else:
                from verl.workers.rollout.vllm_rollout import vLLMRollout, vllm_mode
            from verl.workers.sharding_manager import FSDPVLLMShardingManager
            log_gpu_memory_usage('Before building vllm rollout', logger=None)
            local_path = copy_to_local(self.config.model.path)
            if vllm_mode == 'customized':
                rollout = vLLMRollout(actor_module=self.actor_module_fsdp,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config)
            elif vllm_mode == 'spmd':
                rollout = vLLMRollout(model_path=local_path,
                                      config=self.config.rollout,
                                      tokenizer=self.tokenizer,
                                      model_hf_config=self.actor_model_config,
                                      device_mesh=rollout_device_mesh)
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")
            log_gpu_memory_usage('After building vllm rollout', logger=None)
            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = 'dummy_hf'
            rollout_sharding_manager = FSDPVLLMShardingManager(module=self.actor_module_fsdp,
                                                               inference_engine=rollout.inference_engine,
                                                               model_config=self.actor_model_config,
                                                               full_params='hf' in self.config.rollout.load_format,
                                                               device_mesh=rollout_device_mesh)
            log_gpu_memory_usage('After building sharding manager', logger=None)

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        from verl.workers.actor import DataParallelPPOActor

        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from omegaconf import OmegaConf
        override_model_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))

        use_remove_padding = self.config.model.get('use_remove_padding', False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()
            self.actor_module_fsdp, self.actor_optimizer, self.actor_lr_scheduler, self.actor_model_config = self._build_model_optimizer(
                model_path=self.config.model.path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                enable_gradient_checkpointing=self.config.model.get('enable_gradient_checkpointing', False),
                trust_remote_code=self.config.model.get('trust_remote_code', False),
                use_liger=self.config.model.get('use_liger', False),
                role='actor')

            # get the original unwrapped module
            self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage('After offload actor optimizer during init', logger=logger)
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
            self.actor = DataParallelPPOActor(config=self.config.actor,
                                              actor_module=self.actor_module_fsdp,
                                              actor_optimizer=self.actor_optimizer)

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout()

        if self._is_ref:
            self.ref_module_fsdp = self._build_model_optimizer(model_path=self.config.model.path,
                                                               fsdp_config=self.config.ref.fsdp_config,
                                                               optim_config=None,
                                                               override_model_config=override_model_config,
                                                               use_remove_padding=use_remove_padding,
                                                               trust_remote_code=self.config.model.get(
                                                                   'trust_remote_code', False),
                                                               use_liger=self.config.model.get('use_liger', False),
                                                               role='ref')[0]
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
            self.ref_policy = DataParallelPPOActor(config=self.config.ref, actor_module=self.ref_module_fsdp)

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(model=self.actor_module_fsdp,
                                                            optimizer=self.actor.actor_optimizer,
                                                            lr_scheduler=self.actor_lr_scheduler,
                                                            tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        data = data.to('cuda')

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.actor_optimizer, device_id=torch.cuda.current_device())

        data.batch = data.batch.cuda()

        log_gpu_memory_usage('Before update policy', logger=logger)

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            with Timer(name='update_policy', logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/actor'] = estimated_flops * self.config.actor.ppo_epochs / promised_flops / self.world_size

            self.actor_lr_scheduler.step()
            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics['actor/lr'] = lr

            log_gpu_memory_usage('After update policy', logger=logger)

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={'metrics': metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to('cpu')

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        prompts = prompts.to('cuda')

        assert self._is_rollout
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        prompts.batch = prompts.batch.cuda()
        meta_info = {
            'eos_token_id':
                self.generation_config.eos_token_id
                if self.generation_config is not None else self.tokenizer.eos_token_id,
            'pad_token_id':
                self.generation_config.pad_token_id
                if self.generation_config is not None else self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)
        with self.rollout_sharding_manager:

            # after parameters sync with rollout, offload actor model to CPU
            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)

            log_gpu_memory_usage('After entering rollout sharding manager', logger=logger)

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)
            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage('After rollout generation', logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After recompute log prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        data = data.to('cuda')
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info['micro_batch_size'] = self.config.rollout.log_prob_micro_batch_size_per_gpu
        data.meta_info['max_token_len'] = self.config.rollout.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info['temperature'] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.actor.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'old_log_probs': output},
                                         meta_info={'temperature': self.config.rollout.temperature})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        # clear kv cache
        torch.cuda.empty_cache()
        log_gpu_memory_usage('After compute_log_prob', logger=logger)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        assert self._is_ref

        data = data.to('cuda')

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['temperature'] = self.config.rollout.temperature
        data.meta_info['max_token_len'] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            output = self.ref_policy.compute_log_prob(data=data)
            output = DataProto.from_dict(tensors={'ref_log_prob': output})
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to('cpu')

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.ref_policy.actor_module._handle.reshard(True)

        torch.cuda.empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        # only support save and load ckpt for actor
        assert self._is_actor
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=False):
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.actor_optimizer)


class CriticWorker(Worker):

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        # set FSDP offload params
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        # normalize config
        self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)
        if self.config.ppo_micro_batch_size is not None:
            self.config.ppo_micro_batch_size //= (torch.distributed.get_world_size() //
                                                  self.ulysses_sequence_parallel_size)
            self.config.forward_micro_batch_size //= (torch.distributed.get_world_size() //
                                                      self.ulysses_sequence_parallel_size)
            self.config.ppo_micro_batch_size_per_gpu = self.config.ppo_micro_batch_size
            self.config.forward_micro_batch_size_per_gpu = self.config.forward_micro_batch_size
            assert self.config.ppo_mini_batch_size % self.config.ppo_micro_batch_size_per_gpu == 0, \
                f'normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be divisible by ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}'
            assert self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu > 0, \
                f'normalized ppo_mini_batch_size {self.config.ppo_mini_batch_size} should be larger than ppo_micro_batch_size_per_gpu {self.config.ppo_micro_batch_size_per_gpu}'

    def _build_critic_model_optimizer(self, config):
        # the following line is necessary
        from torch import optim
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

        from verl.utils.model import LambdaLayer, print_model_size, squeeze
        from verl.utils.torch_dtypes import PrecisionType

        local_path = copy_to_local(config.model.path)
        # note that the tokenizer between actor and critic may be different. So override tokenizer info with actor info
        # using random initialized model from any architecture. May not be the same as Actor.

        tokenizer_path = copy_to_local(config.model.tokenizer_path)
        self.tokenizer = hf_tokenizer(tokenizer_path, trust_remote_code=config.model.get('trust_remote_code', False))

        from omegaconf import OmegaConf
        override_config = OmegaConf.to_container(self.config.model.get('override_config', OmegaConf.create()))
        override_config_kwargs = {
            'bos_token_id': self.tokenizer.bos_token_id,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
        }
        override_config_kwargs.update(override_config)
        if self.rank == 0:
            print(f'Critic overriding config {override_config_kwargs}')

        torch_dtype = self.config.model.fsdp_config.get('model_dtype', 'fp32')
        torch_dtype = PrecisionType.to_dtype(torch_dtype)

        from torch import nn
        from transformers import AutoConfig, AutoModelForTokenClassification

        trust_remote_code = False
        critic_model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        critic_model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(critic_model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(critic_model_config, verbose=True)

        init_context = get_init_weight_context_manager()
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(critic_model_config, 'classifier_dropout', 0.)
            setattr(critic_model_config, 'hidden_dropout', '0')
            critic_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            torch_dtype=torch_dtype,
                                                                            config=critic_model_config,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)

            # some parameters may not in torch_dtype
            critic_module.to(torch_dtype)

            if config.model.get('enable_gradient_checkpointing', False):
                critic_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if self.rank == 0:
            print_model_size(critic_module)

        self.critic_model_config = critic_model_config

        fsdp_config = self.config.model.fsdp_config
        mixed_precision_config = fsdp_config.get('mixed_precision', None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(mixed_precision_config.get('param_dtype', 'bf16'))
            reduce_dtype = PrecisionType.to_dtype(mixed_precision_config.get('reduce_dtype', 'fp32'))
            buffer_dtype = PrecisionType.to_dtype(mixed_precision_config.get('buffer_dtype', 'fp32'))
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(param_dtype=param_dtype, reduce_dtype=reduce_dtype, buffer_dtype=buffer_dtype)

        auto_wrap_policy = get_fsdp_wrap_policy(module=critic_module, config=fsdp_config.get('wrap_policy', None))

        log_gpu_memory_usage('Before critic FSDP', logger=None)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        # Note: We force turn off CPUOffload for critic because it causes incorrect results when using grad accumulation
        critic_module = FSDP(critic_module,
                             param_init_fn=init_fn,
                             use_orig_params=False,
                             auto_wrap_policy=auto_wrap_policy,
                             device_id=torch.cuda.current_device(),
                             sharding_strategy=sharding_strategy,
                             mixed_precision=mixed_precision,
                             sync_module_states=True,
                             forward_prefetch=False,
                             device_mesh=self.device_mesh,
                             cpu_offload=None)

        log_gpu_memory_usage('After critic FSDP', logger=None)

        critic_optimizer = optim.AdamW(critic_module.parameters(),
                                       lr=config.optim.lr,
                                       betas=config.optim.get('betas', (0.9, 0.999)),
                                       weight_decay=config.optim.get('weight_decay', 1e-2))

        total_steps = config.optim.get('total_training_steps', 0)
        num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
        num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

        print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

        from verl.utils.torch_functional import get_constant_schedule_with_warmup
        critic_lr_scheduler = get_constant_schedule_with_warmup(optimizer=critic_optimizer,
                                                                num_warmup_steps=num_warmup_steps)

        return critic_module, critic_optimizer, critic_lr_scheduler

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))

        from verl.workers.critic import DataParallelPPOCritic
        self.critic_module, self.critic_optimizer, self.critic_lr_scheduler = self._build_critic_model_optimizer(
            self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)

        self.critic = DataParallelPPOCritic(config=self.config,
                                            critic_module=self.critic_module,
                                            critic_optimizer=self.critic_optimizer)

        self.flops_counter = FlopsCounter(self.critic_model_config)
        self.checkpoint_manager = FSDPCheckpointManager(model=self.critic_module,
                                                        optimizer=self.critic_optimizer,
                                                        lr_scheduler=self.critic_lr_scheduler,
                                                        tokenizer=self.tokenizer)

        torch.cuda.empty_cache()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_values(self, data: DataProto):
        data = data.to('cuda')

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        micro_batch_size = self.config.forward_micro_batch_size_per_gpu
        data.meta_info['micro_batch_size'] = micro_batch_size
        data.meta_info['max_token_len'] = self.config.forward_max_token_len_per_gpu
        data.meta_info['use_dynamic_bsz'] = self.config.use_dynamic_bsz
        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            values = self.critic.compute_values(data=data)
            output = DataProto.from_dict(tensors={'values': values})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        output = output.to('cpu')
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_critic(self, data: DataProto):
        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(optimizer=self.critic_optimizer, device_id=torch.cuda.current_device())

        # perform forward computation
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            with Timer(name='update_critic', logger=None) as timer:
                metrics = self.critic.update_critic(data=data)
            delta_time = timer.last

            global_num_tokens = data.meta_info['global_token_num']
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(global_num_tokens, delta_time)
            metrics['mfu/critic'] = estimated_flops * self.config.ppo_epochs / promised_flops / self.world_size

            self.critic_lr_scheduler.step()
            lr = self.critic_lr_scheduler.get_last_lr()[0]
            metrics['critic/lr'] = lr

            output = DataProto(batch=None, meta_info={'metrics': metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.critic_optimizer)
        torch.cuda.empty_cache()
        output = output.to('cpu')
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, remove_previous_ckpt=False):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.save_checkpoint(local_path=local_path,
                                                hdfs_path=hdfs_path,
                                                global_step=global_step,
                                                remove_previous_ckpt=remove_previous_ckpt)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def load_checkpoint(self, path, del_local_after_load=True):
        import torch
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.critic_module)

        self.checkpoint_manager.load_checkpoint(path=path, del_local_after_load=del_local_after_load)

        torch.distributed.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.critic_module)

        if self._is_offload_optimizer:
            offload_fsdp_optimizer(self.critic_optimizer)


# TODO(sgm): we may need to extract it to dp_reward_model.py
class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)

        # normalize config
        if self.config.micro_batch_size is not None:
            self.config.micro_batch_size //= torch.distributed.get_world_size()
            self.config.micro_batch_size_per_gpu = self.config.micro_batch_size

    def _build_model(self, config):
        # the following line is necessary
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from transformers import AutoConfig, AutoModelForTokenClassification

        # download the checkpoint from hdfs
        local_path = copy_to_local(config.model.path)

        if self.config.model.input_tokenizer is None:
            self._do_switch_chat_template = False
        else:
            self._do_switch_chat_template = True
            input_tokenizer_local_path = copy_to_local(config.model.input_tokenizer)
            self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=config.model.get('trust_remote_code', False))
            self.tokenizer = hf_tokenizer(local_path, trust_remote_code=config.model.get('trust_remote_code', False))

        trust_remote_code = config.model.get('trust_remote_code', False)
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 1

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            reward_module = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=local_path,
                                                                            config=model_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code)
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh)

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import (
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )

        from verl.utils.ulysses import (
            gather_outpus_and_unpad,
            ulysses_pad_and_slice_inputs,
        )

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.reward_module(input_ids=input_ids_rmpad,
                                            attention_mask=None,
                                            position_ids=position_ids_rmpad,
                                            use_cache=False)  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch_size, seqlen=seqlen).squeeze(-1)
            else:
                output = self.reward_module(input_ids=input_ids,
                                            attention_mask=attention_mask,
                                            position_ids=position_ids)
                rm_score = output.logits  # (batch_size, seq_len, 1)
                rm_score = rm_score.squeeze(-1)

            # extract the result of the last valid token
            eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
            rm_score = rm_score[torch.arange(batch_size), eos_mask_idx]
            return rm_score

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        src_max_length = data.batch['attention_mask'].shape[-1]

        src_tokenizer = self.input_tokenizer
        target_tokenizer = self.tokenizer

        rm_input_ids = []
        rm_attention_mask = []

        for i in range(data.batch.batch_size[0]):
            # extract raw prompt
            chat: list = data.non_tensor_batch['raw_prompt'][i].tolist()

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = src_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(src_tokenizer.eos_token, '')

            chat.append({'role': 'assistant', 'content': response})

            prompt_with_chat_template = target_tokenizer.apply_chat_template(chat,
                                                                             add_generation_prompt=False,
                                                                             tokenize=False)
            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {prompt_with_chat_template}')

            # the maximum length is actually determined by the reward model itself
            max_length = self.config.get('max_length', src_max_length)
            if max_length is None:
                max_length = src_max_length
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=target_tokenizer,
                max_length=max_length,
                pad_token_id=target_tokenizer.pad_token_id,
                left_pad=False,  # right padding
                truncation=self.config.get('truncation', 'right'))  # truncate from the right

            rm_input_ids.append(input_ids)
            rm_attention_mask.append(attention_mask)

        rm_input_ids = torch.cat(rm_input_ids, dim=0)
        rm_attention_mask = torch.cat(rm_attention_mask, dim=0)

        rm_position_ids = compute_position_id_with_mask(rm_attention_mask)

        rm_inputs = {'input_ids': rm_input_ids, 'attention_mask': rm_attention_mask, 'position_ids': rm_position_ids}

        return DataProto.from_dict(rm_inputs)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
        data = data.to('cuda')
        if self._do_switch_chat_template:
            rm_data = self._switch_chat_template(data)

        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output


class ProcessRewardModelWorker(Worker):
    """
    Note that we only implement the process reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.fsdp_size
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)

        self.use_remove_padding = self.config.model.get('use_remove_padding', False)
        self._is_offload_param = self.config.model.fsdp_config.param_offload
        self._is_offload_optimizer = self.config.model.fsdp_config.optimizer_offload

        credit_assignment = self.config.get('credit_assignment', 0.1)
        if credit_assignment in ['gamma-decay', 'strict min-form']:
            self.disable_approx_min_form_credit_assignment = True
        else:
            self.disable_approx_min_form_credit_assignment = False
            self.temperature = credit_assignment

        # TODO: online training of PRM
        assert not self.config.training, "Not support yet."
        # normalize config
        # self.config.ppo_mini_batch_size //= (torch.distributed.get_world_size() // self.ulysses_sequence_parallel_size)

    def _build_prm_optimizer(self, config):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        from transformers import AutoConfig, AutoModelForTokenClassification

        from verl.utils.model import print_model_size

        trust_remote_code = config.model.get('trust_remote_code', False)
        local_path = copy_to_local(config.model.path)
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = torch.bfloat16
        model_config = AutoConfig.from_pretrained(local_path, trust_remote_code=trust_remote_code)
        model_config.num_labels = 2

        use_remove_padding = config.model.get('use_remove_padding', False)
        if use_remove_padding:
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(model_config.model_type)

        if use_remove_padding and self.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(model_config, verbose=True)

        init_context = get_init_weight_context_manager(use_meta_tensor=not model_config.tie_word_embeddings)
        
        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            setattr(model_config, 'classifier_dropout', 0.)
            process_reward_module = AutoModelForTokenClassification.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=model_config,
                attn_implementation='flash_attention_2',
                trust_remote_code=trust_remote_code,
            )

            # some parameters may not in torch_dtype
            process_reward_module.to(torch_dtype)

            # if config.model.get('enable_gradient_checkpointing', False):
            #     process_reward_module.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        
        if self.rank == 0:
            print_model_size(process_reward_module)

        fsdp_config = self.config.model.fsdp_config
        auto_wrap_policy = get_fsdp_wrap_policy(
            module=process_reward_module, 
            config=fsdp_config.get('wrap_policy', None),
        )

        sharding_strategy = get_sharding_strategy(self.device_mesh)

        process_reward_module = FSDP(
            process_reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,
            sync_module_states=True,
            forward_prefetch=False,
            device_mesh=self.device_mesh,
            cpu_offload=CPUOffload(offload_params=True),
        )

        log_gpu_memory_usage('After PRM FSDP', logger=None)

        if self.config.training:
            prm_optimizer = optim.AdamW(
                process_reward_module.parameters(),
                lr=config.optim.lr,
                betas=config.optim.get('betas', (0.9, 0.999)),
                weight_decay=config.optim.get('weight_decay', 1e-2),
            )

            total_steps = config.optim.get('total_training_steps', 0)
            num_warmup_steps_ratio = config.optim.get('lr_warmup_steps_ratio', 0.)
            num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f'Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}')

            from verl.utils.torch_functional import get_constant_schedule_with_warmup
            prm_lr_scheduler = get_constant_schedule_with_warmup(
                optimizer=prm_optimizer,
                num_warmup_steps=num_warmup_steps,
            )

            return process_reward_module, prm_optimizer, prm_lr_scheduler
        return process_reward_module

    def _init_separator(self, config):
        # split response into steps based on what character
        split_step_char = config.get('split_step_char', '\n\n')
        self.split_step_tokens = []
        # all tokens which end with "\n\n"
        for i in range(len(self.tokenizer)):
            if self.tokenizer.decode(i).endswith(split_step_char):
                self.split_step_tokens.append(i)
        self.split_step_tokens = torch.LongTensor(
            self.split_step_tokens, 
        ).to(device=torch.cuda.current_device())

        # token for reward prediction
        step_separator = config.get('step_separator', '\n')
        self.step_separator_token = self.tokenizer.encode(
            step_separator, 
            return_tensors='pt',
            add_special_tokens=False,
        ).squeeze(0).to(device=torch.cuda.current_device())
    
    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        import_external_libs(self.config.model.get('external_lib', None))

        results = self._build_prm_optimizer(self.config)
        if self.config.training:
            self.process_reward_module, self.prm_optimizer, self.prm_lr_scheduler = results
        else:
            self.process_reward_module = results
        
        self._init_separator(self.config)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.process_reward_module)
        if self.config.training and self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.prm_optimizer)

        if self.config.training:
            # initialize:
            # DataParallel: forward, loss, backward, optim.step
            # CheckpointManager: save, load ckpt
            raise NotImplementedError

        torch.cuda.empty_cache()

    def _split_steps(self, data):
        bs, problem_length = data.batch['prompts'].size()
        action_mask = data.batch['attention_mask'][:, problem_length:]
        num_actions = action_mask.size(1)
        solution_tokens = data.batch['responses']

        # find step separator, typically '\n\n'
        row_ids, column_ids = torch.where(
            torch.isin(solution_tokens, self.split_step_tokens)
        )
        # +1 for the last step with eos instead of step separator
        max_num_steps = max(
            [column_ids[row_ids==i].numel() for i in range(bs)]) + 1
        # end index of each step, shape: (B, max_num_steps), type: long
        score_ids = torch.full(
            (bs, max_num_steps), -1, dtype=torch.long, 
            device=torch.cuda.current_device(),
        )
        # whether end of step, shape: (B, max_response_tokens), type: bool
        reward_mask = torch.zeros_like(solution_tokens, dtype=torch.bool)
        eos_indices = num_actions - 1 - action_mask.long().fliplr().argmax(1)
        for j in range(bs):
            step_separators_per_data = column_ids[row_ids==j]
            num_intermediate_steps = step_separators_per_data.numel()
            # intermediate steps
            score_ids[j, :num_intermediate_steps] = step_separators_per_data
            reward_mask[j, step_separators_per_data] = True
            # last step
            score_ids[j, num_intermediate_steps] = eos_indices[j]
            reward_mask[j, eos_indices[j]] = True
        
        score_mask = score_ids != -1
        # score_ids, score_mask, reward_mask for data.batch['responses'],
        # not for data.batch['input_ids']
        output = dict(
            score_ids=score_ids,
            score_mask=score_mask,
            reward_mask=reward_mask,
            num_steps=score_mask.float().sum(dim=-1),
        )
        return DataProto.from_dict(tensors=output)
    
    def _build_inputs_for_prm(self, data):
        from torch.nn.utils.rnn import pad_sequence
        from torch.nn import functional as F

        # fetch var
        problem_ids = data.batch['prompts']
        attention_mask = data.batch['attention_mask']
        solution_tokens = data.batch['responses']
        score_ids = data.batch['score_ids']
        score_mask = data.batch['score_mask']
        bs, problem_length = problem_ids.shape
        total_length = data.batch['input_ids'].size(-1)
        problem_attn_mask = attention_mask[:, :problem_length]
        solution_attn_mask = attention_mask[:, problem_length:]
        device = problem_ids.device

        # build input_ids, attn_mask, and position_ids for PRM
        # (optional) remove '\n\n' at the end of each step, 
        # then add '\n' for each step to predict process reward
        input_ids = []
        attn_mask = []
        for i in range(bs):
            input_ids_per_data = problem_ids[i]
            attn_mask_per_data = problem_attn_mask[i]
            # split tokens of each step
            for idx, j in enumerate(score_ids[i][score_mask[i]]):
                # j -> '\n\n'
                if idx == 0:
                    start_idx = 0
                else:
                    start_idx = score_ids[i, idx - 1] + 1
                # slicer [..., :j] means drop the last '\n\n' of each step
                step_tokens = solution_tokens[i, start_idx:j]
                step_attn_mask = solution_attn_mask[i, start_idx:j]
                # add '\n' after each step to predict process reward
                input_ids_per_data = torch.cat(
                    (input_ids_per_data, step_tokens, self.step_separator_token)
                )
                attn_mask_per_data = torch.cat(
                    (attn_mask_per_data, step_attn_mask, torch.ones(
                        1, device=device, dtype=attn_mask_per_data.dtype
                    ))
                )
            input_ids.append(input_ids_per_data)
            attn_mask.append(attn_mask_per_data)
        # gather into batch
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        attn_mask = pad_sequence(attn_mask, batch_first=True, padding_value=0)
        # pad to total_length at dim=1
        input_ids = F.pad(input_ids, (0, total_length - input_ids.size(-1)), value=self.tokenizer.pad_token_id)
        attn_mask = F.pad(attn_mask, (0, total_length - attn_mask.size(-1)), value=0)
        position_ids = compute_position_id_with_mask(attn_mask)

        # for forward of PRM
        output = dict(
            input_ids=input_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
        )
        # for adv baseline, rather than forward of PRM
        output = DataProto.from_dict(tensors=output)
        return output

    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import (
            index_first_axis,
            pad_input,
            rearrange,
            unpad_input,
        )

        from verl.utils.ulysses import (
            gather_outpus_and_unpad,
            ulysses_pad_and_slice_inputs,
        )

        import torch.distributed as dist
        import torch

        # === Debug 开关：只在 rank 0 打印 ===
        debug_print = bool(self.config.get('debug_print', False))
        if debug_print and dist.is_initialized() and dist.get_rank() != 0:
            debug_print = False
        # ================================

        response_length = micro_batch['responses'].size(-1)

        assert 'score_ids' in micro_batch, "Error: score_ids missing from micro_batch. Did you update compute_rm_score?"

        # ================== 1. PRM 模型推理 ==================
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            reward_mask = micro_batch['reward_mask']

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                      indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad,
                        position_ids_rmpad,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.process_reward_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    use_cache=False,
                )  # prevent model thinks we are generating
                reward_rmpad = output.logits
                reward_rmpad = reward_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    reward_rmpad = gather_outpus_and_unpad(reward_rmpad,
                                                           gather_dim=0,
                                                           unpad_dim=0,
                                                           padding_size=pad_size)

                # pad it back
                rm_score = pad_input(reward_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
            else:
                output = self.process_reward_module(input_ids=input_ids,
                                                    attention_mask=attention_mask,
                                                    position_ids=position_ids)
                rm_score = output.logits  # (batch_size, seq_len, 2)
        
        # ================== 2. 计算基础 Token Reward ==================

        rm_score = rm_score[:, -response_length:]
        rm_score = rm_score.softmax(dim=-1)
        # PURE 基础打分: (P_correct - P_incorrect)
        rm_score = (rm_score[..., 1] - rm_score[..., 0]) * reward_mask  # (batch_size, seq_len)

        # [Debug] 保存一份聚合前的原始分数用于对比
        raw_rm_score_debug = rm_score.clone() if debug_print else None

        # ================== 3. Progress Step 聚合 / 切分 ==================
        use_progress_aggregation = bool(self.config.get('use_progress_aggregation', False))
        progress_agg_method = self.config.get('progress_agg_method', 'min')
        use_progress_rubric = self.config.get('use_progress_rubric', False)

        score_ids = micro_batch['score_ids']
        progress_indices = None
        debug_score_ids = None
        debug_generated_indices = None

        if use_progress_aggregation:
            debug_score_ids = score_ids

            # ---------- 分支 1：micro_batch 自带 progress_step_indices ----------
            if 'progress_step_indices' in micro_batch and micro_batch['progress_step_indices'] is not None:
                progress_indices = micro_batch['progress_step_indices']

            # ---------- 分支 2：没有自带，但 use_progress_rubric = True ----------
            elif use_progress_rubric:
                # 这里会根据 rubric 动态构造 progress_indices
                progress_indices = self._build_progress_indices_from_rubric(micro_batch)

                # rubric 失败（全部非法 / 没有边界） -> 退回随机划分
                if progress_indices is None:
                    progress_indices, debug_generated_indices = self._build_random_progress_indices_from_score_ids(
                        score_ids=score_ids,
                    )

            # ---------- 分支 3：纯随机划分 ----------
            else:
                progress_indices, debug_generated_indices = self._build_random_progress_indices_from_score_ids(
                    score_ids=score_ids,
                )

            # 执行聚合：根据 progress_indices 把原子步骤聚合成 Progress 级别 reward
            rm_score, reward_mask = self._aggregate_progress_rewards(
                rm_score, 
                reward_mask, 
                score_ids, 
                progress_indices, 
                method=progress_agg_method
            )

        # ================== 4. Approximate Min-Form Credit Assignment ==================
        debug_weight = None
        if not self.disable_approx_min_form_credit_assignment:
            # 注意：此时 reward_mask 已经被 _aggregate_progress_rewards 稀疏化了
            # 只有在 Progress Step 的边界处为 True，因此 Softmax 只会在这些节点间分配权重
            weight = torch.softmax(
                -rm_score.masked_fill(
                    ~reward_mask, float('inf')
                ) / self.temperature,
                dim=-1,
            )
            debug_weight = weight if debug_print and use_progress_aggregation else None
            rm_score *= weight

        # ================== [DEBUG PRINT BLOCK] ==================
        if debug_print and use_progress_aggregation:
            print("\n" + "="*30 + " PURE Progress Debug " + "="*30)
            # 只查看 Batch 中的第一个样本
            b = 0 
            
            # 1. 打印切分详情
            if debug_score_ids is not None:
                # 获取原子步骤的物理位置索引
                valid_locs = debug_score_ids[b]
                valid_locs = valid_locs[valid_locs != -1]
                total_steps = len(valid_locs)
                print(f"[Sample 0] Total Atomic Steps: {total_steps}")
                
                if debug_generated_indices:
                    partitions = debug_generated_indices[b]
                    print(f"[Sample 0] Random Partitions (Logical Indices, 1-based cumulative): {partitions}")
                    lengths = []
                    prev = 0
                    for p in partitions:
                        lengths.append(p - prev)
                        prev = p
                    print(f"[Sample 0] Segment Lengths: {lengths}")

            if raw_rm_score_debug is not None and debug_score_ids is not None:
                import numpy as np

                valid_locs = debug_score_ids[b]
                valid_locs = valid_locs[valid_locs != -1]
                raw_step_scores = raw_rm_score_debug[b, valid_locs].detach().float().cpu().numpy()
                print(f"[Sample 0] Raw Step Scores ({len(raw_step_scores)}):")
                print(f"   {['{:.4f}'.format(x) for x in raw_step_scores]}")
                
                # 提取聚合后的 Progress 分数
                # 我们通过 reward_mask 找到聚合后的非零位置
                agg_mask = reward_mask[b].bool()
                agg_locs = torch.nonzero(agg_mask).squeeze(-1)
                agg_scores = rm_score[b, agg_locs].detach().float().cpu().numpy() # 注意：这里是已经乘过 weight 后的最终分数
                
                # 为了看清楚聚合效果，我们需要还原未乘 weight 的聚合分数
                # 如果 weight 接近 0，还原可能会不稳定，所以这里我们重新去 rm_score (在乘 weight 之前的值很难获取，
                # 除非我们在上面存临时变量。这里为了简单，我们打印 weight 和 最终 score)
                
                print(f"[Sample 0] Progress Step Locs (Physical): {agg_locs.cpu().tolist()}")
                
                if debug_weight is not None:
                    progress_weights = debug_weight[b, agg_locs].detach().float().cpu().numpy()
                    print(f"[Sample 0] Softmax Weights (Approx Min):")
                    print(f"   {['{:.4f}'.format(x) for x in progress_weights]}")
                    
                    # 反推聚合后的原始分数 (Approximation)
                    # Final = Agg * Weight => Agg = Final / Weight (仅供参考，Weight可能极小)
                    # 更好的方式是在乘 weight 之前就 print，但为了不破坏代码结构，这里主要看 Weight 分布
                    
                print(f"[Sample 0] Final Weighted Rewards:")
                print(f"   {['{:.4f}'.format(x) for x in agg_scores]}")
                
            print("="*80 + "\n")
            # =========================================================

        return rm_score

    def _decode_student_steps_for_one_sample(self,
                                         response_tokens,  # shape: [resp_len]
                                         score_ids_1d,     # shape: [max_num_steps]
                                         tokenizer):
        """
        严格按照 _split_steps 的定义，用 score_ids 来切 step。
        返回: List[str] student_steps，长度 = 有效步数 = (score_ids_1d != -1).sum()
        """
        import torch

        # 有效 step 的结束位置（token 下标）
        valid_mask = (score_ids_1d != -1)
        if not torch.any(valid_mask):
            return []

        step_ends = score_ids_1d[valid_mask]          # e.g. tensor([3, 7, 12])
        step_ends, _ = torch.sort(step_ends)          # 理论上本来就有序，保险起见再 sort 一下

        steps = []
        start = 0
        for end in step_ends:
            end = int(end.item())
            if end < start:
                # 万一 score_ids 异常，直接跳过，避免负切片
                continue
            tokens = response_tokens[start:end + 1]   # 包含 end
            text = tokenizer.decode(
                tokens.detach().cpu().tolist(),
                skip_special_tokens=True,
            ).strip()
            steps.append(text)
            start = end + 1

        return steps

    def _map_rubric_to_step_boundaries(self,
                                       progress_items,
                                       student_steps):
        """
        输入:
        progress_items: generate_progress_rubric 返回的 JSON array (list[dict])
                        其中 "Included Steps" 是从 1 开始的 step 序号 (int 或可转成 int 的字符串)
        student_steps:  List[str]，来自 _decode_student_steps_for_one_sample

        输出:
        boundaries: List[int]，1-based 累积 step index，例如 [3, 7, 10]

        说明:
        - 完全在分支处做，不写进 generate_progress_rubric。
        - 一个 step 最多属于一个 progress，用全局指针防止复用。
        """
        boundaries = []
        num_steps = len(student_steps)
        if num_steps == 0:
            return boundaries

        # 1-based：已经分配到的最大 step index
        global_used_upto = 0

        for item in progress_items:
            if not isinstance(item, dict):
                continue
            included = item.get("Included Steps", [])
            if not isinstance(included, list):
                continue

            max_idx_for_progress = 0

            for step_idx in included:
                # 支持 int 或 "3" 这种字符串
                if isinstance(step_idx, int):
                    idx = step_idx
                elif isinstance(step_idx, str):
                    s = step_idx.strip()
                    if not s.isdigit():
                        continue
                    idx = int(s)
                else:
                    continue

                # 保证合法范围，且不复用已经分配过的 step
                if 1 <= idx <= num_steps and idx > global_used_upto:
                    if idx > max_idx_for_progress:
                        max_idx_for_progress = idx
                    global_used_upto = idx

            if max_idx_for_progress > 0:
                # 已经是 1-based
                boundaries.append(max_idx_for_progress)

        # 去重 + 排序，确保合法
        boundaries = sorted(
            set(b for b in boundaries if 1 <= b <= num_steps)
        )
        return boundaries

    def _build_progress_indices_from_rubric(self, micro_batch):
        """
        rubric 分支：完全在这里做所有额外逻辑：

        1. 用 score_ids 切每个样本的 student_steps
        2. 调 generate_progress_rubric(prompt_text, student_steps) 拿 JSON array
        3. 用 _map_rubric_to_step_boundaries 得到 boundaries
        4. 失败则退回随机划分
        5. 拼成 progress_step_indices tensor 返回（可包装成 DataProto）
        """

        if not getattr(self, "use_progress_rubric", False):
            return None

        tokenizer = self.tokenizer

        required_keys = ["prompts", "responses", "attention_mask", "score_ids"]
        if not all(k in micro_batch for k in required_keys):
            print("[Rubric] micro_batch missing required keys, skip rubric.")
            return None

        prompts = micro_batch["prompts"]          # [B, prompt_len]
        responses = micro_batch["responses"]      # [B, resp_len]
        attn_mask = micro_batch["attention_mask"] # [B, prompt_len + resp_len] or similar
        score_ids = micro_batch["score_ids"]      # [B, max_num_steps]

        batch_size = prompts.size(0)
        device = prompts.device

        min_step_size = self.config.get("min_step_size", 2)
        max_progress = self.config.get("max_progress", 5)

        progress_idx_list = []
        max_len = 0

        # prompt_text 这边简单 decode 整个 prompts（也可以用 mask 精细点）
        for i in range(batch_size):
            prompt_ids = prompts[i]
            prompt_text = tokenizer.decode(
                prompt_ids.detach().cpu().tolist(),
                skip_special_tokens=True,
            )

            resp_tokens = responses[i]
            score_ids_i = score_ids[i]

            # 1) 用 score_ids_i 切 steps
            student_steps = self._decode_student_steps_for_one_sample(
                resp_tokens,
                score_ids_i,
                tokenizer,
            )

            valid_steps_count = len(student_steps)
            if valid_steps_count == 0:
                progress_idx_list.append([])
                continue

            # 2) 调 LLM，拿 JSON array
            progress_items = generate_progress_rubric(
                prompt_text=prompt_text,
                student_steps=student_steps,
            )

            # 3) rubric → boundaries
            boundaries = []
            if progress_items:
                boundaries = self._map_rubric_to_step_boundaries(
                    progress_items,
                    student_steps,
                )

            # 4) 如果 rubric 不靠谱，就退回随机划分
            if (not boundaries) and valid_steps_count > 0:
                boundaries = self._sample_random_progress_boundaries(
                    valid_steps_count=valid_steps_count,
                    min_step_size=min_step_size,
                    max_progress=max_progress,
                )

            progress_idx_list.append(boundaries)
            max_len = max(max_len, len(boundaries))

        if max_len == 0:
            return None

        progress_indices = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        for i, boundaries in enumerate(progress_idx_list):
            if boundaries:
                progress_indices[i, :len(boundaries)] = torch.tensor(
                    boundaries,
                    dtype=torch.long,
                    device=device,
                )

        # 看你习惯：要么直接返回 tensor，要么包一层 DataProto
        # 你之前是 DataProto.from_dict(tensors={'progress_step_indices': ...})
        return DataProto.from_dict(tensors={"progress_step_indices": progress_indices})

    # ---------------------------------------------------------------------------

    def _sample_random_progress_boundaries(self, valid_steps_count: int,
                                        min_step_size: int, max_progress: int) -> list[int]:
        """
        对单个样本，根据原子步数量随机生成 Progress 边界（1-based 累积步数）。
        例如 valid_steps_count=10, 返回 [3, 6, 10]。
        """
        import torch

        if valid_steps_count <= 0:
            return []

        # 步数太少，直接作为一个整体
        if valid_steps_count <= min_step_size:
            return [valid_steps_count]

        # 最大可以切成多少段
        max_partitions = min(max_progress, valid_steps_count // min_step_size)
        if max_partitions <= 1:
            return [valid_steps_count]

        # 随机选择实际段数 k
        k = int(torch.randint(1, max_partitions + 1, (1,)).item())
        if k == 1:
            return [valid_steps_count]

        boundaries: list[int] = []
        current_step = 0
        steps_remaining = valid_steps_count

        for i in range(k - 1):
            # 剩余空间必须保证每个剩余段至少 min_step_size
            max_step_size = steps_remaining - (k - 1 - i) * min_step_size
            step_size = int(torch.randint(min_step_size, max_step_size + 1, (1,)).item())

            current_step += step_size
            boundaries.append(current_step)
            steps_remaining -= step_size

        boundaries.append(valid_steps_count)
        return boundaries


    def _build_random_progress_indices_from_score_ids(self, score_ids):
        """
        batch 级别的随机切分：
        输入: score_ids [B, max_steps]，-1 表示无效。
        输出:
            progress_indices [B, L] (1-based 累积步数, 0 为 padding),
            generated_indices_list: Python 列表，用于 Debug。
        """
        import torch

        min_step_size = self.config.get('min_step_size', 2)
        max_progress = self.config.get('max_progress', 5)

        batch_size = score_ids.size(0)
        device = score_ids.device

        generated_indices_list = []
        max_len = 0

        for b in range(batch_size):
            valid_steps_count = int((score_ids[b] != -1).sum().item())
            boundaries = self._sample_random_progress_boundaries(
                valid_steps_count=valid_steps_count,
                min_step_size=min_step_size,
                max_progress=max_progress,
            )
            generated_indices_list.append(boundaries)
            max_len = max(max_len, len(boundaries))

        if max_len == 0:
            progress_indices = torch.zeros(
                (batch_size, 0),
                dtype=torch.long,
                device=device,
            )
            return progress_indices, generated_indices_list

        progress_indices = torch.zeros(
            (batch_size, max_len),
            dtype=torch.long,
            device=device,
        )

        for b, boundaries in enumerate(generated_indices_list):
            if boundaries:
                progress_indices[b, :len(boundaries)] = torch.tensor(
                    boundaries,
                    dtype=torch.long,
                    device=device,
                )

        return progress_indices, generated_indices_list

    # ---------------------------------------------------------------------------
    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools

        from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

        data = data.to('cuda')
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.process_reward_module)

        # 1. 拆分出 score_ids 等原子步信息
        data.union(self._split_steps(data))

        # 2. 构建 PRM 输入
        prm_data = self._build_inputs_for_prm(data)
        prm_data = prm_data.to('cuda')

        # 把 reward_mask / responses / score_ids / prompts 也并入 PRM batch
        prm_data.union(
            data.select(batch_keys=['reward_mask', 'responses', 'score_ids', 'prompts'])
        )

        with self.ulysses_sharding_manager:
            prm_data = self.ulysses_sharding_manager.preprocess_data(data=prm_data)

            self.process_reward_module.eval()
            batch = prm_data.batch

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
            else:
                micro_batches = batch.split(self.config.micro_batch_size_per_gpu)

            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(rm_score)
            token_level_scores = torch.cat(output, dim=0)

            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == token_level_scores.size(0), f"{len(indices)} vs. {token_level_scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                token_level_scores = token_level_scores[revert_indices]

            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1:
            self.process_reward_module._handle.reshard(True)
        
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.process_reward_module)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output

    def _aggregate_progress_rewards(self, rm_score, reward_mask, score_ids, progress_indices, method='mean'):
        """
        rm_score: [batch, seq_len] - Token level scores
        reward_mask: [batch, seq_len] - True at step boundaries
        score_ids: [batch, max_steps] - Physical token indices of step boundaries
        progress_indices: [batch, max_progress_steps] - Logical step indices (1-based) ending each progress step
        method: 'mean', 'min', 'sum', 'product'
        """
        batch_size = rm_score.size(0)
        
        # 创建新的稀疏 Reward 和 Mask
        new_rm_score = torch.zeros_like(rm_score)
        new_reward_mask = torch.zeros_like(reward_mask)
        
        # 遍历 Batch (由于逻辑比较复杂，这里使用循环，性能影响在Micro-batch下可控)
        for b in range(batch_size):
            # 1. 获取该样本所有 Step 的物理 Token 位置
            # score_ids 中 -1 是 Padding，我们只取有效的
            valid_step_locs = score_ids[b]
            valid_step_locs = valid_step_locs[valid_step_locs != -1] # shape: [total_steps]
            
            if len(valid_step_locs) == 0:
                continue

            # 获取该样本所有 Step 的原始分数
            # 注意：rm_score 在非 step 位置是 0，我们直接按索引取值
            step_scores = rm_score[b, valid_step_locs] # shape: [total_steps]
            
            # 2. 获取 Progress 分组边界
            # progress_indices 也是 padded 的 (假设用 0 或 -1 pad，这里假设有效值为 > 0)
            p_indices = progress_indices[b]
            p_indices = p_indices[p_indices > 0] # e.g., [3, 8, 10]
            
            start_step_idx = 0
            
            for end_step_idx in p_indices:
                end_step_idx = int(end_step_idx.item())
                
                # 安全检查：如果索引越界（比如生成截断导致步数不够），则停止
                if end_step_idx > len(step_scores):
                    break
                
                # 3. 截取当前 Progress Step 内的所有原子 Step 分数
                # 这里的切片是逻辑 Step 的切片
                current_group_scores = step_scores[start_step_idx : end_step_idx]
                
                if len(current_group_scores) == 0:
                    start_step_idx = end_step_idx
                    continue

                # 4. 聚合计算
                if method == 'mean':
                    agg_score = current_group_scores.mean()
                elif method == 'sum':
                    agg_score = current_group_scores.sum()
                elif method == 'min':
                    agg_score = current_group_scores.min()
                elif method == 'product':
                    # 注意：Reward 是 [-1, 1] 的 centered score。
                    # Product 对负数可能产生不直观的结果（如负负得正），请根据业务确认是否需要先转概率
                    agg_score = current_group_scores.prod()
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")
                
                # 5. 将聚合分数写回张量
                # 我们将其放置在当前 Progress Step 的**最后一个原子 Step** 的物理位置
                # 这样保持了时序的因果性
                last_step_physical_idx = valid_step_locs[end_step_idx - 1]
                
                new_rm_score[b, last_step_physical_idx] = agg_score
                new_reward_mask[b, last_step_physical_idx] = True
                
                # 更新起点，准备下一个 Progress Step
                start_step_idx = end_step_idx
                
        return new_rm_score, new_reward_mask