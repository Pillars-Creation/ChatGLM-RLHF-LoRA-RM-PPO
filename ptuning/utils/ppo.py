import os
import math
import torch
from tqdm import tqdm
from typing import Callable, Dict, List, Literal, Optional, Tuple

from transformers import Seq2SeqTrainingArguments, TrainerState
from transformers.modeling_utils import PreTrainedModel

from trl import PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from trl.trainer.ppo_trainer import PPODecorators, logprobs_from_logits

from .peft_trainer import PeftTrainer, LogCallback

from .config import FinetuningArguments

from .other import (
    AverageMeter,
    get_logger,
    get_logits_processor
)


logger = get_logger(__name__)

#这个函数的作用是替换模型的头部，以便于在训练过程中计算奖励。
def replace_model(model: AutoModelForCausalLMWithValueHead, target: Literal["default", "reward"]) -> None:
    if target == "reward": # 如果目标是reward，则先保存原始头部权重和偏置
        valuehead_state_dict = model.v_head.state_dict()

        setattr(model, "origin_head_weight", valuehead_state_dict["summary.weight"])
        setattr(model, "origin_head_bias", valuehead_state_dict["summary.bias"])

    model.pretrained_model.set_adapter(target) # 设置LoRA适配器为活动状态
    model.v_head.load_state_dict({
        "summary.weight": getattr(model, "{}_head_weight".format(target)), # 加载指定头部权重
        "summary.bias": getattr(model, "{}_head_bias".format(target)) # 加载指定头部偏置
    })


def cast_layernorm_dtype(
        model: AutoModelForCausalLMWithValueHead,
        layer_norm_names: List[str] = ["layernorm"], # for chatglm setting
        layer_norm_params: Optional[Dict[str, torch.Tensor]] = None
) -> Tuple[AutoModelForCausalLMWithValueHead, Dict[str, torch.Tensor]]:

    layer_norm_state_dict = {}

    for name, param in model.named_parameters():
        if param.ndim == 1 and any(layer_norm_name in name for layer_norm_name in layer_norm_names):
            if layer_norm_params is not None:
                param.data = layer_norm_params[name] # restore float32 weights
            else:
                layer_norm_state_dict[name] = param.data.detach().clone() # store float32 weights for stability
                param.data = param.data.to(torch.float16)

    return model, layer_norm_state_dict


class PPOTrainerForChatGLM(PPOTrainer, PeftTrainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
            self,
            training_args: Seq2SeqTrainingArguments,
            finetuning_args: FinetuningArguments,
            callbacks: List[LogCallback],
            **kwargs
    ):
        PPOTrainer.__init__(self, **kwargs)
        self.args = training_args
        self.finetuning_args = finetuning_args
        self.log_callback = callbacks[0]
        self.state = TrainerState()
        self.data_collator = self.accelerator.prepare(kwargs["data_collator"])

    def ppo_train(self, max_target_length: int) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        total_train_batch_size = self.config.batch_size * self.config.gradient_accumulation_steps * self.args.world_size
        len_dataloader = len(self.dataloader)
        num_steps_per_epoch = max(len_dataloader // self.config.gradient_accumulation_steps, 1)
        num_examples = len(self.dataset)
        num_train_epochs = self.args.num_train_epochs
        max_steps = math.ceil(num_train_epochs * num_steps_per_epoch)

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info(f"  Num examples = {num_examples}")
            logger.info(f"  Num Epochs = {num_train_epochs}")
            logger.info(f"  Instantaneous batch size per device = {self.config.batch_size}")
            logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
            logger.info(f"  Gradient Accumulation steps = {self.config.gradient_accumulation_steps}")
            logger.info(f"  Total optimization steps = {max_steps}")
            logger.info(f"  Number of trainable parameters = {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

        # Keyword arguments for `model.generate`
        gen_kwargs = {
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "logits_processor": get_logits_processor()
        }
        output_length_sampler = LengthSampler(max_target_length // 2, max_target_length)
        unwrapped_model: PreTrainedModel = self.accelerator.unwrap_model(self.model)

        dataiter = iter(self.dataloader)
        steps_trained = 0
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()

        for step in tqdm(range(max_steps), disable=not self.is_world_process_zero()):

            for _ in range(self.config.gradient_accumulation_steps):

                batch = next(dataiter)  # 从数据迭代器中获取一个batch的数据
                steps_trained += 1  # 累加已训练的步数

                unwrapped_model.gradient_checkpointing_disable()  # 禁用梯度检查点
                unwrapped_model.config.use_cache = True  # 启用缓存

                # Get response from ChatGLM
                query_tensors: torch.Tensor = batch["input_ids"]  # 从batch中获取查询的张量
                response_tensors = self.generate(batch, length_sampler=output_length_sampler, return_prompt=False,
                                                 **gen_kwargs)  # 使用ChatGLM模型生成回复的张量

                queries: List[torch.Tensor] = []  # 查询列表
                responses: List[torch.Tensor] = []  # 回复列表
                for i in range(len(query_tensors)):  # 遍历每个查询和回复的张量
                    query_length = (query_tensors[i] != self.tokenizer.pad_token_id).nonzero()[0]  # 计算查询的长度
                    response_length = (response_tensors[i] != self.tokenizer.pad_token_id).nonzero()[-1] + 1  # 计算回复的长度
                    queries.append(query_tensors[i, query_length:])  # 将查询添加到查询列表中，去掉左侧的填充
                    if response_length < 2:  # 如果回复的长度小于2
                        responses.append(
                            response_tensors.new_empty(2).fill_(self.tokenizer.eos_token_id))  # 将回复填充为EOS标记，确保回复至少有两个标记
                    else:
                        responses.append(response_tensors[i, :response_length])  # 将回复添加到回复列表中，去掉右侧的填充

                # Compute rewards
                replace_model(unwrapped_model, target="reward")  # 将模型切换到奖励模式
                _, _, values = self.model(**self.prepare_model_inputs(queries, responses))  # 使用查询和回复的张量计算奖励值
                rewards = [reward for reward in values[-1]]  # 将奖励值存储在rewards列表中
                replace_model(unwrapped_model, target="default")  # 确保模型在最后是默认模式

                # Run PPO step
                unwrapped_model.gradient_checkpointing_enable()  # 启用梯度检查点
                unwrapped_model.config.use_cache = False  # 禁用缓存

                stats = self.step(queries, responses, rewards)  # 使用PPO算法更新模型参数，并记录损失和奖励的平均值

                loss_meter.update(stats["ppo/loss/total"], n=len(rewards))  # 记录损失的平均值
                reward_meter.update(torch.stack(rewards).mean().item(), n=len(rewards))  # 记录奖励的平均值

                if steps_trained == len_dataloader:  # 如果已经训练了一个epoch
                    dataiter = iter(self.dataloader)  # 重置数据迭代器
                    steps_trained = 0  # 重置已训练的步数

            if self.is_world_process_zero() and (step + 1) % self.args.logging_steps == 0:  # 如果是主进程且到达了logging_steps的倍数
                logs = {
                    "loss": round(loss_meter.avg, 4),  # 记录损失的平均值
                    "reward": round(reward_meter.avg, 4),  # 记录奖励的平均值
                    "learning_rate": stats["ppo/learning_rate"],  # 记录学习率
                    "epoch": round(step / num_steps_per_epoch, 2)  # 记录当前epoch
                }
                print(logs)  # 打印日志
                logs["step"] = step  # 记录当前步数
                self.state.log_history.append(logs)  # 将日志添加到历史记录中
                self.log_callback.on_log(self.args, self.state, None)  # 调用回调函数记录日志
                loss_meter.reset()  # 重置损失的平均值
                reward_meter.reset()  # 重置奖励的平均值

            if (step + 1) % self.args.save_steps == 0:  # 如果到达了save_steps的倍数
                self.save_model(os.path.join(self.args.output_dir, f"checkpoint-{step + 1}"))  # 保存模型的checkpoint

    @torch.no_grad()
    def generate(
            self,
            inputs: Dict[str, torch.Tensor],
            length_sampler: Optional[Callable] = None,
            return_prompt: Optional[bool] = True,
            **generation_kwargs,
    ) -> torch.Tensor:
        r"""
        Generates model's responses given queries.

        Subclass and override to inject custom behavior.
        """
        self.model, layer_norm_params = cast_layernorm_dtype(self.model)

        if length_sampler is not None:
            generation_kwargs["max_new_tokens"] = length_sampler()

        unwrapped_model = self.accelerator.unwrap_model(self.model)

        response = unwrapped_model.generate(**inputs, **generation_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # Inspired by: https://github.com/huggingface/transformers/blob/v4.28.1/src/transformers/trainer_seq2seq.py#L273
        if unwrapped_model.pretrained_model.generation_config._from_model_config:
            unwrapped_model.pretrained_model.generation_config._from_model_config = False

        self.model, _ = cast_layernorm_dtype(self.model, layer_norm_params)

        if not return_prompt and not self.is_encoder_decoder:
            return response[:, inputs["input_ids"].size(1):]
        return response

    #这个函数的作用是将查询和回复的张量拼接起来，并使用数据处理器处理拼接后的张量。处理后的张量被移动到当前设备，并返回。在处理过程中，标签被移除，避免计算语言模型的损失。
    def prepare_model_inputs(self, queries: List[torch.Tensor], responses: List[torch.Tensor]) -> Dict[
        str, torch.Tensor]:
        input_ids = [torch.cat([q, r]) for q, r in zip(queries, responses)]  # 将查询和回复的张量拼接起来
        input_data = self.data_collator([{"input_ids": ids} for ids in input_ids])  # 使用数据处理器处理拼接后的张量
        input_data = {k: v.to(self.current_device) for k, v in input_data.items() if v is not None}  # 将张量移动到当前设备
        input_data.pop("labels", None)  # we don't want to compute LM losses # 移除标签，避免计算语言模型的损失
        return input_data  # 返回处理后的张量

    @PPODecorators.empty_cuda_cache()
    def batched_forward_pass(
        self,
        model: AutoModelForCausalLMWithValueHead,
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(model_inputs["input_ids"])
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(int(bs / fbs)):
            torch.cuda.empty_cache()
            input_kwargs = {k: v[i * fbs : (i + 1) * fbs] for k, v in model_inputs.items()}
            input_ids: torch.Tensor = input_kwargs["input_ids"] # left-padded sequences
            if self.is_distributed: # re-generate them to adapt padded inputs
                input_kwargs["attention_mask"] = self.data_collator.get_attention_masks(input_ids, device=self.current_device)
                input_kwargs["position_ids"] = self.data_collator.get_position_ids(input_ids, device=self.current_device)
            logits, _, values = model(**input_kwargs)
            logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])

            values = values.transpose(0, 1)
            masks = torch.zeros_like(input_ids)

            for j in range(fbs):
                start = (input_ids[j] == self.tokenizer.bos_token_id).nonzero()[0].item()
                masks[j][start:] = 1
                if len(masks[j][start:]) < 2:
                    raise ValueError("Responses are too short. Make sure they are at least 4 tokens long.")

            all_logits.append(logits)
            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1],
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            self._save(output_dir)
