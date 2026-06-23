%%writefile src/train_lora.py
import os, sys, json, torch
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
sys.path.insert(0, os.path.dirname(__file__))

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
from datasets import Dataset
import wandb

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CHECKPOINT_DIR = "/content/drive/MyDrive/tanglish-project/checkpoints"

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def main():
    wandb.login()

    train_dataset = Dataset.from_list(load_jsonl("data/train_formatted.jsonl"))
    dev_dataset   = Dataset.from_list(load_jsonl("data/dev_formatted.jsonl"))

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
        num_train_epochs=3,
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
        eval_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",
        run_name="tanglish-lora-r16-lr2e4-full",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        args=training_args,
        processing_class=tokenizer,
    )

    trainer.train(
        resume_from_checkpoint=True if os.path.exists(CHECKPOINT_DIR) and
        any("checkpoint" in d for d in os.listdir(CHECKPOINT_DIR)) else None
    )

    # Save the final LoRA adapter
    adapter_path = "/content/drive/MyDrive/tanglish-project/tanglish-lora-adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to {adapter_path}")

if __name__ == "__main__":
    main()
