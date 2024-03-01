from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import load_dataset, Dataset
from bitnet158 import BitLinear, BitLinear158, inject
import argparse
from transformers import TrainerCallback

### Parse Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_id", type=str, default="microsoft/phi-2")
parser.add_argument("--dataset", type=str, default="EleutherAI/wikitext_document_level")
parser.add_argument("--subset", type=str, default="wikitext-103-raw-v1")
parser.add_argument("--output_dir", type=str, default="saved_model")
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

# print number of parameters that are trainable
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Number of parameters: ", num_params)

### Load Dataset
ds = load_dataset(args.dataset, args.subset, split="train")
val_ds = load_dataset(args.dataset, args.subset, split="test")
def tokenize_function(examples):
    return tokenizer(examples["page"], truncation=True, max_length=256)

tokenized_datasets = ds.map(tokenize_function, batched=False, num_proc=32, remove_columns=["page"])
tokenized_datasets_val = val_ds.map(tokenize_function, batched=False, num_proc=32, remove_columns=["page"])
print(tokenized_datasets)

### Inject BitLinear layers
if args.inject == "BitLinear158":
    inject(model, copy_weights=True, module_class=BitLinear158)
elif args.inject == "BitLinear":
    inject(model, copy_weights=True, module_class=BitLinear)
else:
    pass


class PrintFirstLayerGradientsCallback(TrainerCallback):
    """
    A custom callback that prints the gradient of the first layer of the model at each training step.
    """
    def on_step_end(self, args, state, control, **kwargs):
        # Assuming 'model' is your model instance and it's a PyTorch model
        # You may need to adjust the layer name depending on your model architecture
        num_params = 0
        print("First layer gradients")
        for k, v in model.named_parameters():
            if v.grad is not None:
                print(k, v.grad)
                num_params += v.numel()
            else:
                print(k, "None")
        print("Number of parameters: ", num_params)


### Start Training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        deepspeed="ds_config.json",
        output_dir=args.output_dir,
        save_steps=100,
        fp16=True,
        save_total_limit=3,
        per_device_train_batch_size=2,
        eval_steps=100,
        evaluation_strategy="steps",
        logging_steps=10,
        learning_rate=1e-3,
    ),
    train_dataset=tokenized_datasets,
    eval_dataset=tokenized_datasets_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    callbacks=[PrintFirstLayerGradientsCallback()],
)

trainer.train()
