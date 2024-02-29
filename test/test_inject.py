import transformers
from bitnet158 import inject
from bitnet158 import BitLinear158, BitLinear

# init LLaMa 7B model
model = transformers.AutoModel.from_pretrained("EleutherAI/gpt-neo-125m")
class_name = BitLinear
inject(model, copy_weights=True, module_class=class_name)
print(model)
