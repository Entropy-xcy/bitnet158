import transformers
from bitnet158 import inject
from bitnet158 import BitLinear158, BitLinear
import torch

model = transformers.AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
class_name = BitLinear
inject(model, copy_weights=True, module_class=class_name)

sample_input = torch.randint(2000, (1, 1024)).long()
out = model(sample_input)
print(out.logits.shape)
