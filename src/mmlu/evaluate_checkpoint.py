from transformers import AutoTokenizer, Qwen2ForCausalLM, Qwen2Config
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.benchmarks import MMLU
from deepeval.benchmarks.mmlu.task import MMLUTask
import torch


class CustomQwen2(DeepEvalBaseLLM):
    def __init__(self, model, tokenizer, prompt_prefix=None):
        self.model = model
        self.tokenizer = tokenizer
        if prompt_prefix is not None:
            self.prompt_prefix = prompt_prefix + '\n\n'
        else:
            self.prompt_prefix = ''

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        # Format the prompt
        full_prompt = self.prompt_prefix + prompt
        
        # Tokenize the input
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        
        # Move to the same device as the model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,  # MMLU typically needs single letter answers
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode only the new tokens (excluding the input prompt)
        input_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return "Qwen/Qwen2.5-0.5B"

# Initialize your model configuration
hidden_size = 1536  # You'll need to specify this value
config = Qwen2Config(
    attention_dropout=0.0,
    bos_token_id=151643,
    eos_token_id=151643,
    hidden_act="silu",
    hidden_size=hidden_size,
    initializer_range=0.02,
    intermediate_size=4864,
    max_position_embeddings=513,
    max_window_layers=12,
    model_type="qwen2",
    num_attention_heads=16,
    num_hidden_layers=12,
    num_key_value_heads=16,
    rms_norm_eps=1e-06,
    rope_theta=1000000.0,
    sliding_window=1024,
    tie_word_embeddings=True,
    torch_dtype="bfloat16",
    use_cache=True,
    use_mrope=False,
    use_sliding_window=False,
    vocab_size=151936,
)

# Load your model
model = Qwen2ForCausalLM(config)

# Load the tokenizer (you'll need to specify the correct tokenizer)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

# If you have a checkpoint to load
model.load_state_dict(torch.load("path/to/your/checkpoint.pth"))

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# Create the custom model wrapper
custom_qwen = CustomQwen2(model, tokenizer)

# Run MMLU benchmark
benchmark = MMLU(tasks=[MMLUTask.HIGH_SCHOOL_COMPUTER_SCIENCE])
results = benchmark.evaluate(model=custom_qwen)
print("Overall Score: ", results)
