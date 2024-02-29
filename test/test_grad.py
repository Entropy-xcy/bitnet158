import transformers
from bitnet158 import inject
from bitnet158 import BitLinear158, BitLinear
import torch

model_id = "EleutherAI/gpt-neo-125m"
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
model = transformers.AutoModelForCausalLM.from_pretrained(model_id)

inject(model, copy_weights=True, module_class=BitLinear)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
# print(labels.shape)
outputs = model(**inputs, labels=inputs["input_ids"])
loss = outputs.loss
loss.backward()

# print the grad
for name, param in model.named_parameters():
    print(name, param.grad)
