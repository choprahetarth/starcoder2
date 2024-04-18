# Code adapted from https://github.com/huggingface/trl/blob/main/examples/research_projects/stack_llama/scripts/supervised_finetuning.py
# and https://huggingface.co/blog/gemma-peft
import argparse
import multiprocessing
import os

import torch
import transformers
from accelerate import PartialState
from datasets import load_dataset, DatasetDict
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    logging,
    set_seed,
)
# from trl import SFTTrainer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default="bigcode/starcoder2-3b")
    parser.add_argument("--dataset_name", type=str, default="/u/choprahetarth/all_files/data/train_ftdata-new-small.json")
    parser.add_argument("--dataset_text_field", type=str, default="content")

    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--micro_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--bf16", type=bool, default=True)

    parser.add_argument("--attention_dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="finetune_starcoder2")
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--push_to_hub", type=bool, default=True)
    return parser.parse_args()


# def print_trainable_parameters(model):
#     """
#     Prints the number of trainable parameters in the model.
#     """
#     trainable_params = 0
#     all_param = 0
#     for _, param in model.named_parameters():
#         all_param += param.numel()
#         if param.requires_grad:
#             trainable_params += param.numel()
#     print(
#         f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
#     )


def main(args):
    # # config
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16,
    # )
    # lora_config = LoraConfig(
    #     r=8,
    #     target_modules=[
    #         "q_proj",
    #         "o_proj",
    #         "k_proj",
    #         "v_proj",
    #         "gate_proj",
    #         "up_proj",
    #         "down_proj",
    #     ],
    #     task_type="CAUSAL_LM",
    # )

    # load model and dataset
    token = os.environ.get("HF_TOKEN", None)
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_id,
    #     quantization_config=bnb_config,
    #     device_map={"": PartialState().process_index},
    #     attention_dropout=args.attention_dropout,
    # )
    # print_trainable_parameters(model)

    def create_instruction(sample):
        return {
            "prompt": "Given this Ansible Name Field, please generate the ansible task. " + sample["input"],
            "completion": sample["output"]
        }

    dataset = load_dataset('json', data_files=args.dataset_name, num_proc=multiprocessing.cpu_count())
    columns_to_remove = ['license', 'repo_name', 'path', 'download_link', 'download_count', 'org_name', 'output', 'input']
    dataset = dataset['train'].map(create_instruction, remove_columns=columns_to_remove, batched=False)
    train_val_split = dataset.train_test_split(test_size=0.1, seed=args.seed)  
    data = DatasetDict({
        'train': train_val_split['train'],
        'validation': train_val_split['test'],
    })
    print("Sample from the training dataset: ", data['train'][0])

#     # setup the trainer
#     trainer = SFTTrainer(
#         model=model,
#         # train_dataset=data,
#         train_dataset=data['train'],
#         eval_dataset=data['validation'],
#         max_seq_length=args.max_seq_length,
#         args=transformers.TrainingArguments(
#             per_device_train_batch_size=args.micro_batch_size,
#             gradient_accumulation_steps=args.gradient_accumulation_steps,
#             warmup_steps=args.warmup_steps,
#             max_steps=args.max_steps,
#             learning_rate=args.learning_rate,
#             lr_scheduler_type=args.lr_scheduler_type,
#             weight_decay=args.weight_decay,
#             bf16=args.bf16,
#             logging_strategy="steps",
#             logging_steps=10,
#             output_dir=args.output_dir,
#             optim="paged_adamw_8bit",
#             seed=args.seed,
#             run_name=f"train-{args.model_id.split('/')[-1]}",
#             report_to="wandb",
#         ),
#         peft_config=lora_config,
#         dataset_text_field=args.dataset_text_field,
#     )

#     # launch
#     print("Training...")
#     trainer.train()

#     print("Saving the last checkpoint of the model")
#     model.save_pretrained(os.path.join(args.output_dir, "final_checkpoint/"))
#     if args.push_to_hub:
#         trainer.push_to_hub("Upload model")
#     print("Training Done! 💥")


if __name__ == "__main__":
    args = get_args()
#     set_seed(args.seed)
#     os.makedirs(args.output_dir, exist_ok=True)

#     logging.set_verbosity_error()

    main(args)
