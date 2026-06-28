import os, sys, json, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import wandb

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT_DIR = "/content/drive/MyDrive/tanglish-project/checkpoints"
ADAPTER_DIR = "/content/drive/MyDrive/tanglish-project/tanglish-lora-adapter"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    wandb.login()

    train_dataset = Dataset.from_list(load_jsonl("/content/data/train_sample_formatted.jsonl"))
    dev_dataset   = Dataset.from_list(load_jsonl("/content/data/dev_formatted.jsonl"))
    print(f"Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16, lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=CHECKPOINT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        fp16=False, bf16=True,
        logging_steps=10,
        save_strategy="steps",
        save_steps=100,
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb",
        run_name="tanglish-lora-r16-lr2e4-final",
        max_steps=300,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    resume = None
    if os.path.exists(CHECKPOINT_DIR):
        ckpts = [d for d in os.listdir(CHECKPOINT_DIR) if "checkpoint" in d]
        if ckpts:
            resume = True
            print(f"Resuming from checkpoint")

    trainer.train(resume_from_checkpoint=resume)

    os.makedirs(ADAPTER_DIR, exist_ok=True)
    model.save_pretrained(ADAPTER_DIR)
    tokenizer.save_pretrained(ADAPTER_DIR)
    print(f"Adapter saved to {ADAPTER_DIR}")

if __name__ == "__main__":
    main()
