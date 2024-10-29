"""
This file contains the configuration of the models used in the project.
"""
class get_model_config:
    def __init__(self):
        pass
    def create_llama_config(self,model):

        llama_config = {
        "model_type" : model.config.model_type,
        "hidden_size" : model.config.hidden_size,
        "num_hidden_layers" : model.config.num_hidden_layers,
        "num_attention_heads" : model.config.num_attention_heads,
        "start_layer_prefex" : "model.layers.",
        "mlp_name": "mlp",
        "attention_name": "self_attn",
        "attn_proj_name": "self_attn.o_proj",

        }
        return llama_config





    def model_config(self,model_name,model):
        if model_name=="huggyllama/llama-7b" or model_name=="lmsys/vicuna-7b-v1.3" or "llama" in model_name or "GOAT" in model_name or "mistral" in model_name\
                or "gemma" in model_name:
            return self.create_llama_config(model)