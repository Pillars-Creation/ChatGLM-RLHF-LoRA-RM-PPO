#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
import json

import numpy as np
from datasets import load_dataset
import jieba 
from rouge_chinese import Rouge
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import argparse
import transformers
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed,
)
from trainer_seq2seq import Seq2SeqTrainer

from arguments import ModelArguments, DataTrainingArguments

logger = logging.getLogger(__name__)

PRE_SEQ_LEN = 64
LR = 2e-2
device = torch.device("cpu")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--do_train", action="store_true",default="true")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--validation_file", type=str, default="dev.json")
    parser.add_argument("--prompt_column", type=str, default="content")
    parser.add_argument("--response_column", type=str, default="summary")
    parser.add_argument("--overwrite_cache", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="../../chatglm-6B")
    parser.add_argument("--output_dir", type=str, default=f"output/adgen-chatglm-6b-pt-{PRE_SEQ_LEN}-{LR}")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--max_source_length", type=int, default=64)
    parser.add_argument("--max_target_length", type=int, default=64)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--predict_with_generate", action="store_true")
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=LR)
    parser.add_argument("--pre_seq_len", type=int, default=PRE_SEQ_LEN)
    # parser.add_argument("--quantization_bit", type=int, default=8)

    # 解析参数
    args = parser.parse_args()

    # 将参数赋值给 sys.argv
    sys.argv = [sys.argv[0]] + [f"--{k}={v}" for k, v in vars(args).items()]

    # 解析命令行参数
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果命令行参数只有一个，并且是一个 JSON 文件的路径，则解析该文件以获取参数
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则，解析命令行参数
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 设置日志记录的格式和级别
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # 如果需要记录日志，则设置日志级别为 info
        transformers.utils.logging.set_verbosity_info()
        log_level = training_args.get_process_log_level()
        logger.setLevel(log_level)
        # 获取日志级别，并设置 logger 的日志级别

        transformers.utils.logging.set_verbosity(log_level)
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
        # 设置 transformers 的日志级别，并启用默认处理程序和显式格式

        # Log on each process the small summary:
        logger.warning(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )
        logger.info(f"Training/evaluation parameters {training_args}")
        # 打印进程、设备、GPU 数量、分布式训练和 16 位训练等信息，并打印训练/评估参数

        # Set seed before initializing model.
        set_seed(training_args.seed)
        # 设置随机种子

        # Load dataset
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        # 加载数据集文件

        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        # 加载数据集，并设置缓存目录和身份验证令牌

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.pre_seq_len = model_args.pre_seq_len
    config.prefix_projection = model_args.prefix_projection
    # 从预训练模型中加载配置文件，并设置 pre_seq_len 和 prefix_projection

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    # 从预训练模型中加载 tokenizer

    if model_args.ptuning_checkpoint is not None:
        # 如果设置了 ptuning_checkpoint

        # Evaluation
        # Loading extra state dict of prefix encoder
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        # 从预训练模型中加载模型

        prefix_state_dict = torch.load(os.path.join(model_args.ptuning_checkpoint, "pytorch_model.bin"))
        new_prefix_state_dict = {}
        for k, v in prefix_state_dict.items():
            if k.startswith("transformer.prefix_encoder."):
                new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
        model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
        # 加载额外的 prefix encoder 的状态字典

    else:
        # Finetune
        model = AutoModel.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True)
        # 从预训练模型中加载模型

    if model_args.quantization_bit is not None:
        print(f"Quantized to {model_args.quantization_bit} bit")
        model = model.quantize(model_args.quantization_bit).to(device)
        # 如果设置了 quantization_bit，将模型量化为指定的位数

    if model_args.pre_seq_len is not None:
        # 如果设置了 pre_seq_len

        # P-tuning v2
        model = model.half()
        model.transformer.prefix_encoder.float()
        # 将模型和 prefix encoder 转换为半精度浮点数

    else:
        # Finetune
        model = model.float()
        # 将模型转换为浮点数

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    # 如果设置了 source_prefix，将其赋值给 prefix，否则将其设置

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Get the column names for input/target.
    prompt_column = data_args.prompt_column
    response_column = data_args.response_column
    history_column = data_args.history_column
    
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length

    def preprocess_function_eval(examples):
        # 定义预处理函数，输入为 examples，输出为 model_inputs

        inputs, targets = [], []
        # 定义 inputs 和 targets 列表

        for i in range(len(examples[prompt_column])):
            # 遍历 examples 中的每个样本

            if examples[prompt_column][i] and examples[response_column][i]:
                # 如果 prompt 和 response 都不为空

                query = examples[prompt_column][i]
                # 将 prompt 赋值给 query

                if history_column is None or len(examples[history_column][i]) == 0:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                # 如果有 history_column，将历史对话和当前的 prompt 拼接起来，否则直接使用 prompt

                inputs.append(prompt)
                targets.append(examples[response_column][i])
                # 将 prompt 和 response 分别添加到 inputs 和 targets 中

        inputs = [prefix + inp for inp in inputs]
        # 将 prefix 和 inputs 中的每个元素拼接起来

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, truncation=True, padding=True)
        # 使用 tokenizer 对 inputs 进行编码，并进行截断和填充

        labels = tokenizer(text_target=targets, max_length=max_target_length, truncation=True)
        # 使用 tokenizer 对 targets 进行编码，并进行截断

        if data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        # 如果 ignore_pad_token_for_loss 为 True，将 labels 中的 pad_token_id 替换为 -100

        model_inputs["labels"] = labels["input_ids"]
        # 将 labels 中的 input_ids 赋值给 model_inputs 中的 labels

        return model_inputs
        # 返回 model_inputs

    def preprocess_function_train(examples):
        # 定义预处理函数，输入为 examples，输出为 model_inputs

        max_seq_length = data_args.max_source_length + data_args.max_target_length
        # 计算最大序列长度，即 prompt 和 response 的最大长度之和

        model_inputs = {
            "input_ids": [],
            "labels": [],
        }
        # 定义 model_inputs，包含 input_ids 和 labels 两个字段

        for i in range(len(examples[prompt_column])):
            # 遍历 examples 中的每个样本

            if examples[prompt_column][i] and examples[response_column][i]:
                # 如果 prompt 和 response 都不为空

                query, answer = examples[prompt_column][i], examples[response_column][i]
                # 将 prompt 和 response 分别赋值给 query 和 answer

                if history_column is None:
                    prompt = query
                else:
                    prompt = ""
                    history = examples[history_column][i]
                    for turn_idx, (old_query, response) in enumerate(history):
                        prompt += "[Round {}]\n问：{}\n答：{}\n".format(turn_idx, old_query, response)
                    prompt += "[Round {}]\n问：{}\n答：".format(len(history), query)
                # 如果有 history_column，将历史对话和当前的 prompt 拼接起来，否则直接使用 prompt

                prompt = prefix + prompt
                # 将 prefix 和 prompt 拼接起来

                a_ids = tokenizer.encode(text=prompt, add_special_tokens=False)
                b_ids = tokenizer.encode(text=answer, add_special_tokens=False)
                # 使用 tokenizer 对 prompt 和 response 进行编码

                if len(a_ids) > data_args.max_source_length - 1:
                    a_ids = a_ids[: data_args.max_source_length - 1]
                # 如果 prompt 的长度超过了最大长度减一，截断 prompt

                if len(b_ids) > data_args.max_target_length - 2:
                    b_ids = b_ids[: data_args.max_target_length - 2]
                # 如果 response 的长度超过了最大长度减二，截断 response

                input_ids = tokenizer.build_inputs_with_special_tokens(a_ids, b_ids)
                # 将 prompt 和 response 拼接起来，并加上特殊的 token

                context_length = input_ids.index(tokenizer.bos_token_id)
                mask_position = context_length - 1
                labels = [-100] * context_length + input_ids[mask_position + 1:]
                # 构造 labels，其中 context_length 是 bos_token_id 的位置，mask_position 是下一个 token 的位置

                pad_len = max_seq_length - len(input_ids)
                input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
                labels = labels + [tokenizer.pad_token_id] * pad_len
                # 将 input_ids 和 labels 补齐到最大长度

                if data_args.ignore_pad_token_for_loss:
                    labels = [(l if l != tokenizer.pad_token_id else -100) for l in labels]
                # 如果 ignore_pad_token_for_loss 为 True，将 labels 中的 pad_token_id 替换为 -100

                model_inputs["input_ids"].append(input_ids)
                model_inputs["labels"].append(labels)
                # 将 input_ids 和 labels 添加到 model_inputs 中

        return model_inputs
        # 返回 model_inputs
    
    def print_dataset_example(example):
        print("input_ids",example["input_ids"])
        print("inputs", tokenizer.decode(example["input_ids"]))
        print("label_ids", example["labels"])
        print("labels", tokenizer.decode(example["labels"]))

    if training_args.do_train:  # 如果需要进行训练
        if "train" not in raw_datasets:  # 如果原始数据集中没有训练集
            raise ValueError("--do_train requires a train dataset")  # 抛出异常，提示需要提供训练集
        train_dataset = raw_datasets["train"]  # 获取训练集
        if data_args.max_train_samples is not None:  # 如果设置了最大训练样本数
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)  # 计算最大训练样本数
            train_dataset = train_dataset.select(range(max_train_samples))  # 选择最大训练样本数的子集
        with training_args.main_process_first(desc="train dataset map pre-processing"):  # 在主进程中进行训练集预处理
            train_dataset = train_dataset.map(  # 对训练集进行映射操作
                preprocess_function_train,  # 预处理函数
                batched=True,  # 是否对数据进行批处理
                num_proc=data_args.preprocessing_num_workers,  # 预处理使用的进程数
                remove_columns=column_names,  # 需要移除的列名
                load_from_cache_file=not data_args.overwrite_cache,  # 是否从缓存文件中加载数据
                desc="Running tokenizer on train dataset",  # 显示的描述信息
            )
        print_dataset_example(train_dataset[0])  # 打印训练集的第一个样本

    if training_args.do_eval:  # 如果需要进行评估
        max_target_length = data_args.val_max_target_length  # 获取最大目标长度
        if "validation" not in raw_datasets:  # 如果原始数据集中没有验证集
            raise ValueError("--do_eval requires a validation dataset")  # 抛出异常，提示需要提供验证集
        eval_dataset = raw_datasets["validation"]  # 获取验证集
        if data_args.max_eval_samples is not None:  # 如果设置了最大评估样本数
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)  # 计算最大评估样本数
            eval_dataset = eval_dataset.select(range(max_eval_samples))  # 选择最大评估样本数的子集
        with training_args.main_process_first(desc="validation dataset map pre-processing"):  # 在主进程中进行验证集预处理
            eval_dataset = eval_dataset.map(  # 对验证集进行映射操作
                preprocess_function_eval,  # 预处理函数
                batched=True,  # 是否对数据进行批处理
                num_proc=data_args.preprocessing_num_workers,  # 预处理使用的进程数
                remove_columns=column_names,  # 需要移除的列名
                load_from_cache_file=not data_args.overwrite_cache,  # 是否从缓存文件中加载数据
                desc="Running tokenizer on validation dataset",  # 显示的描述信息
            )
        print_dataset_example(eval_dataset[0])  # 打印验证集的第一个样本

    if training_args.do_predict:
        # 如果设置了 do_predict 标志

        max_target_length = data_args.val_max_target_length
        # 设置最大目标长度为 val_max_target_length

        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        # 如果 raw_datasets 中没有 test 数据集，抛出 ValueError 异常

        predict_dataset = raw_datasets["test"]
        # 将 test 数据集赋值给 predict_dataset

        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        # 如果设置了 max_predict_samples，将 predict_dataset 截取到指定长度

        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function_eval,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )
        # 使用 preprocess_function_eval 对 predict_dataset 进行预处理

        print_dataset_example(predict_dataset[0])
        # 打印 predict_dataset 的第一个样本

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    # 如果 ignore_pad_token_for_loss 为 True，将 label_pad_token_id 设置为 -100，否则设置为 tokenizer.pad_token_id

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=None,
        padding=False
    )

    # 定义 data_collator，用于将数据转换为模型所需的格式

    # Metric
    def compute_metrics(eval_preds):
        # 定义评估指标函数，输入为 eval_preds，输出为 score_dict

        preds, labels = eval_preds
        # 将 preds 和 labels 赋值给 preds 和 labels

        if isinstance(preds, tuple):
            preds = preds[0]
        # 如果 preds 是元组类型，将其转换为列表类型

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        # 使用 tokenizer 对 preds 进行解码

        if data_args.ignore_pad_token_for_loss:
            # 如果 ignore_pad_token_for_loss 为 True，将 labels 中的 -100 替换为 pad_token_id
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        # 使用 tokenizer 对 labels 进行解码

        score_dict = {
            "rouge-1": [],
            "rouge-2": [],
            "rouge-l": [],
            "bleu-4": []
        }
        # 定义 score_dict，包含 rouge-1、rouge-2、rouge-l 和 bleu-4 四个指标

        for pred, label in zip(decoded_preds, decoded_labels):
            # 遍历解码后的 preds 和 labels

            hypothesis = list(jieba.cut(pred))
            reference = list(jieba.cut(label))
            # 使用 jieba 对预测值和真实值进行分词

            rouge = Rouge()
            scores = rouge.get_scores(' '.join(hypothesis), ' '.join(reference))
            result = scores[0]
            # 使用 Rouge 计算 ROUGE 指标，ROUGE（Recall-Oriented Understudy for Gisting Evaluation）
            # 是一种用于自动评估文本摘要和机器翻译的指标。它通过比较生成的摘要或翻译与参考摘要或翻译之间的重叠来计算得分。
            # ROUGE 指标包括 ROUGE-1、ROUGE-2 和 ROUGE-L 等，
            # 其中 ROUGE-1 表示单个词的重叠，ROUGE-2 表示两个词的重叠，ROUGE-L 表示最长公共子序列的重叠。
            # ROUGE 指标的取值范围为 0 到 1，值越大表示生成的摘要或翻译与参考摘要或翻译之间的重叠越多，即越好。
            # 在使用 Rouge 计算 ROUGE 指标时，rouge.get_scores() 方法返回一个包含多个指标的列表，每个指标都是一个字典，
            # 包含 precision、recall 和 f-measure 三个值。因此，scores[0] 表示第一个指标的字典，
            # 其中包含 precision、recall 和 f-measure 三个值。在这里，我们默认使用 ROUGE-1 指标，
            # 因此 scores[0] 表示 ROUGE-1 指标的字典。

            for k, v in result.items():
                score_dict[k].append(round(v["f"] * 100, 4))
            # 将 ROUGE 指标添加到 score_dict 中

            bleu_score = sentence_bleu([list(label)], list(pred), smoothing_function=SmoothingFunction().method3)
            score_dict["bleu-4"].append(round(bleu_score * 100, 4))
            # 计算 BLEU 指标，并将其添加到 score_dict 中

        for k, v in score_dict.items():
            score_dict[k] = float(np.mean(v))
        # 计算每个指标的平均值，并将其转换为浮点数类型

        return score_dict
        # 返回 score_dict

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    # 如果设置了 generation_max_length，将其赋值给 training_args.generation_max_length，否则将 val_max_target_length 赋值给其
    # 如果设置了 num_beams，将其赋值给 training_args.generation_num_beams，否则将其保持不变

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        save_prefixencoder=model_args.pre_seq_len is not None
    )
    # 初始化 Seq2SeqTrainer，包括模型、参数、训练集、验证集、tokenizer、data_collator、compute_metrics 和 save_prefixencoder

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        # elif last_checkpoint is not None:
        #     checkpoint = last_checkpoint

        # 启用梯度检查点，减少显存使用，提高训练效率
        model.gradient_checkpointing_enable()
        # 启用输入梯度，使模型在训练时计算输入的梯度，提高训练效果
        model.enable_input_require_grads()

        # 开始训练模型，resume_from_checkpoint 表示是否从检查点恢复训练
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # trainer.save_model()  # Saves the tokenizer too for easy upload

        # 将训练指标保存到 metrics 变量中
        metrics = train_result.metrics
        # 判断是否设置了 max_train_samples 参数，如果设置了，则将 train_samples 设置为 max_train_samples 和训练数据集大小的较小值，否则将其设置为训练数据集的大小
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        # 将训练指标记录到日志中
        trainer.log_metrics("train", metrics)
        # 将训练指标保存到文件中
        trainer.save_metrics("train", metrics)
        # 保存训练状态，包括模型和优化器的参数
        trainer.save_state()

    # Evaluation
    results = {}
    # 定义一个空字典 results

    max_seq_length = data_args.max_source_length + data_args.max_target_length + 1
    # 计算最大序列长度

    if training_args.do_eval:
        # 如果设置了 do_eval 为 True

        logger.info("*** Evaluate ***")
        # 打印日志信息

        metrics = trainer.evaluate(metric_key_prefix="eval", do_sample=True, top_p=0.7, max_length=max_seq_length,
                                   temperature=0.95)
        # 使用 evaluate 方法评估模型，并返回评估指标

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        # 计算评估样本数，并将其添加到 metrics 中

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        # 打印评估指标，并将其保存到文件中

    if training_args.do_predict:
        # 如果设置了 do_predict 为 True

        logger.info("*** Predict ***")
        # 打印日志信息

        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=max_seq_length,
                                          do_sample=True, top_p=0.7, temperature=0.95)
        # 使用 predict 方法预测模型，并返回预测结果

        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        # 计算预测样本数，并将其添加到 metrics 中

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        # 打印预测指标，并将其保存到文件中

        if trainer.is_world_process_zero():
            # 如果是主进程

            if training_args.predict_with_generate:
                # 如果设置了 predict_with_generate 为 True

                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                # 使用 tokenizer 对预测值进行解码，并去除特殊标记和空格

                predictions = [pred.strip() for pred in predictions]
                # 去除预测值中的空格

                labels = tokenizer.batch_decode(
                    predict_results.label_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                # 使用 tokenizer 对真实值进行解码，并去除特殊标记和空格

                labels = [label.strip() for label in labels]
                # 去除真实值中的空格

                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    for p, l in zip(predictions, labels):
                        res = json.dumps({"labels": l, "predict": p}, ensure_ascii=False)
                        writer.write(f"{res}\n")
                # 将预测值和真实值写入文件中

    return results
    # 返回结果字典 results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
