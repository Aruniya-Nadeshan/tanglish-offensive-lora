import os, sys, csv, re, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(__file__))
from load_data import load_all, LABELS

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_FILE = "zero_shot_predictions.csv"

LABEL_DESCRIPTIONS = """- Not_offensive: not offensive
- Offensive_Untargetede: offensive, but not aimed at a specific person/group
- Offensive_Targeted_Insult_Individual: offensive, aimed at a specific person
- Offensive_Targeted_Insult_Group: offensive, aimed at a specific group
- Offensive_Targeted_Insult_Other: offensive, aimed at something else (e.g. an event)
- not-Tamil: the comment is in a DIFFERENT language entirely (e.g. Hindi, French, Sinhala), with no Tamil content at all. Note: Tamil written purely in Tamil script, with no English mixed in, still counts as Tamil — NOT this category."""

FEW_SHOT_EXAMPLES = """Here are some worked examples:

Comment: செட்டியார் இன மக்கள் சார்பாக எனது சொந்த செலவில் 100 டிக்கெட் இலவசமாக வழங்குகிறேன். படக்குழுவினர்க்கு எனது மனமார்ந்த வாழ்த்துக்கள்
Category: Not_offensive

Comment: في قناتي على اليوتيوب ( كل ماتتمناه 18 ) ًاشتركو في قناتي فضلً منكم ولا يسى امرا
Category: not-Tamil

Comment: Pa Ranjith paru Da unaku sc St quote job government job quarters ellam cancel pannanum
Category: Offensive_Targeted_Insult_Individual
"""


def normalize(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def parse_label(generated_text):
    cleaned = normalize(generated_text)
    for label in LABELS:
        if normalize(label) in cleaned:
            return label
    return "PARSE_FAIL"


def load_model():
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, quantization_config=bnb_config, device_map="auto",
    )
    return tokenizer, model


def classify(text, tokenizer, model):
    prompt = f"""You are classifying social media comments written in Tamil-English \
code-mixed text ("Tanglish"), which may use Tamil script, Latin-script transliterated \
Tamil, or English, sometimes mixed in the same comment.

Classify the comment into exactly one of these categories:
{LABEL_DESCRIPTIONS}

{FEW_SHOT_EXAMPLES}
Now classify this comment:
Comment: {text}

Respond with ONLY the category name, nothing else."""
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
    ).to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=20, do_sample=False, pad_token_id=tokenizer.eos_token_id
    )
    raw = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return parse_label(raw)


def already_done(output_file):
    if not os.path.exists(output_file):
        return set()
    with open(output_file, "r", encoding="utf-8") as f:
        return {int(row["index"]) for row in csv.DictReader(f)}


def main(limit=None):
    _, _, test = load_all()
    done_indices = already_done(OUTPUT_FILE)
    print(f"Already done: {len(done_indices)} / {len(test)}")

    print("Loading model...")
    tokenizer, model = load_model()

    file_exists = os.path.exists(OUTPUT_FILE)
    count = 0
    start = time.time()

    with open(OUTPUT_FILE, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["index", "text", "true_label", "predicted_label"])

        for idx, row in test.iterrows():
            if idx in done_indices:
                continue
            if limit is not None and count >= limit:
                break

            pred = classify(row["text"], tokenizer, model)
            writer.writerow([idx, row["text"], row["label"], pred])
            f.flush()

            count += 1
            if count % 100 == 0:
                rate = (time.time() - start) / count
                print(f"{count} done this run | {rate:.2f}s/example")

    print("Batch finished.")


if __name__ == "__main__":
    main()
