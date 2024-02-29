from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from bitnet158 import BitLinear, BitLinear158, inject
import argparse

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="microsoft/phi-2")
parser.add_argument("--dataset", type=str, default="EleutherAI/wikitext_document_level")
parser.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--output_dir", type=str, default="saved_model")
parser.add_argument("--save_steps", type=int, default=100)
parser.add_argument("--inject", choices=["BitLinear158", "BitLinear", "None"], default="BitLinear")

# DeepSpeed Arguments
parser.add_argument("--train_args_file", type=str, default='--', help="")
parser.add_argument("--deepspeed", type=str, default='--', help="")
parser.add_argument('--local_rank', type=int, default=-1,
                help='local rank passed from distributed launcher')

args = parser.parse_args()

### Load Model
model_id = args.model_id

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)


### Load Dataset
ds = load_dataset(args.dataset, args.subset, split="train[:10%]+test[:1]")
def tokenize_function(examples):
    return tokenizer(examples["page"], truncation=True, max_length=256)

tokenized_datasets = ds.map(tokenize_function, batched=False, num_proc=32, remove_columns=["page"])
print(tokenized_datasets)

### Inject BitLinear layers
if args.inject == "BitLinear158":
    inject(model, copy_weights=True, module_class=BitLinear158)
elif args.inject == "BitLinear":
    inject(model, copy_weights=True, module_class=BitLinear)
else:
    pass


### Start Training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        deepspeed="ds_config.json",
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        fp16=True,
        save_total_limit=3,
        per_device_train_batch_size=2,
    ),
    train_dataset=tokenized_datasets["train"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[],
)

trainer.train()
