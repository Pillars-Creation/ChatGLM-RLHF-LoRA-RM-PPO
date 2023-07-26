import sys
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments
)

from utils import (
    prepare_args,
    prepare_data,
    load_pretrained,
    preprocess_data,
    PairwiseDataCollatorForChatGLM,
    PairwiseTrainerForChatGLM,
    LogCallback,
    plot_loss
)

def main():

    # prepare pretrained model and dataset
    model_args, data_args, training_args, finetuning_args = prepare_args(stage="rm")
    dataset = prepare_data(model_args, data_args)
    model, tokenizer = load_pretrained(model_args, training_args, finetuning_args, training_args.do_train, stage="rm")
    dataset = preprocess_data(dataset, tokenizer, data_args, training_args, stage="rm")
    data_collator = PairwiseDataCollatorForChatGLM(tokenizer, model.pretrained_model)

    training_args.remove_unused_columns = False # Important for pairwise dataset

    # Split the dataset
    if training_args.do_train:
        if data_args.dev_ratio > 1e-6:
            dataset = dataset.train_test_split(test_size=data_args.dev_ratio)
            trainer_kwargs = {"train_dataset": dataset["train"], "eval_dataset": dataset["test"]}
        else:
            trainer_kwargs = {"train_dataset": dataset}
    else: # do_eval or do_predict
        trainer_kwargs = {"eval_dataset": dataset}

    # Initialize our Trainer
    trainer = PairwiseTrainerForChatGLM(
        finetuning_args=finetuning_args,
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[LogCallback()],
        **trainer_kwargs
    )

    # Training
    if training_args.do_train:  # 如果设置了do_train为True，则进行训练
        train_result = trainer.train()  # 进行训练，并返回训练结果
        trainer.log_metrics("train", train_result.metrics)  # 记录训练指标
        trainer.save_metrics("train", train_result.metrics)  # 保存训练指标
        trainer.save_state()  # 保存训练状态
        trainer.save_model()  # 保存模型
        if trainer.is_world_process_zero() and model_args.plot_loss:  # 如果是全局进程中的第一个进程，并且设置了plot_loss为True，则绘制损失图
            plot_loss(training_args, keys=["loss", "eval_loss"])

    # Evaluation
    if training_args.do_eval:  # 如果设置了do_eval为True，则进行评估
        metrics = trainer.evaluate(metric_key_prefix="eval")  # 进行评估，并返回评估指标
        trainer.log_metrics("eval", metrics)  # 记录评估指标
        trainer.save_metrics("eval", metrics)  # 保存评估指标


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

CUDA_VISIBLE_DEVICES=1

if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].endswith(".sh"):
        sys.argv.extend([
            "--do_train",
            "--dataset", "comparison_gpt4_zh",
            "--dataset_dir", "./data",
            "--finetuning_type", "lora",
            "--output_dir", "path_to_rm_checkpoint",
            "--overwrite_cache",
            "--per_device_train_batch_size", "2",
            "--gradient_accumulation_steps", "8",
            "--lr_scheduler_type", "cosine",
            "--logging_steps", "10",
            "--save_steps", "100",
            "--learning_rate", "1e-5",
            "--num_train_epochs", "1.0",
            "--plot_loss",
            "--fp16"
        ])
    main()
