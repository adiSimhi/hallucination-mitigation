"""
This file runs the model and collects the hidden states after each component at each layer.
"""
import functools
import gc
from typing import List
import psutil
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from models_config import get_model_config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class InnerStatesUsingWrapper():

    def __init__(self, MODEL_NAME):
        self.MODEL_NAME = MODEL_NAME
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(device)
        self.model.eval()

        model_creator = get_model_config()
        self.model_condig = model_creator.model_config(MODEL_NAME, self.model)

        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
            print(f"Setting pad_token to eos_token: {self.tok.pad_token}")

    def generate_interactive(self, prompt: str, paraphraze_prompt: str):
        """
        get the hidden states of the model using the prompt and the paraphraze_prompt
        """
        assert len(paraphraze_prompt) >= len(prompt), f"{len(paraphraze_prompt)=} {len(prompt)=}"
        mlp_all_layers_stats = []
        attention_all_layers_stats = []
        norm_mean, attention_norm_mean, _, _, _, _ = self.run_model([prompt], self.model)
        assert norm_mean.shape == attention_norm_mean.shape
        mlp_all_layers_stats.append((norm_mean, "without context"))
        attention_all_layers_stats.append((attention_norm_mean, "without context"))
        norm_mean, attention_norm_mean, last_token_mlp_vector, last_token_attention_vector, heads_vectores, last_token_residual_stream = self.run_model(
            [paraphraze_prompt], self.model)
        mlp_all_layers_stats.append((norm_mean, "with context"))
        attention_all_layers_stats.append((attention_norm_mean, "with context"))
        return mlp_all_layers_stats, attention_all_layers_stats, last_token_mlp_vector, last_token_attention_vector, heads_vectores, last_token_residual_stream

    def rgetattr(self, obj, attr, *args):
        def _getattr(obj, attr):
            return getattr(obj, attr, *args)

        return functools.reduce(_getattr, [obj] + attr.split('.'))

    # a safe way to set attribute of an object
    def rsetattr(self, obj, attr, val):
        pre, _, post = attr.rpartition('.')
        return setattr(self.rgetattr(obj, pre) if pre else obj, post, val)

    def wrap_model(self, model,
                   layers_to_check: List[str],
                   max_len=1000):

        hs_collector = {}
        handles = []

        for layer_idx in range(self.model_condig["num_hidden_layers"]):
            for layer_type in layers_to_check:
                list_inputs = []
                list_outputs = []

                layer_with_idx = f'{layer_idx}{layer_type}'
                inside_name = f"{self.model_condig['start_layer_prefex']}{layer_with_idx}"
                layer_pointer = self.rgetattr(model, inside_name)

                handel = layer_pointer.register_forward_hook(self.extract_hs_include_prefix(list_inputs=list_inputs,
                                                                                            list_outputs=list_outputs,
                                                                                            info=layer_with_idx,
                                                                                            ))
                handles.append(handel)

                if layer_idx not in hs_collector:
                    hs_collector[layer_idx] = {}
                layer_key = layer_type.strip('.')
                # first time seeing this layer
                if layer_key not in hs_collector[layer_idx]:
                    hs_collector[layer_idx][layer_key] = {}

                hs_collector[layer_idx][layer_key]['input'] = list_inputs
                hs_collector[layer_idx][layer_key]['output'] = list_outputs
        del list_inputs, list_outputs, layer_pointer

        return hs_collector, handles

    def model_wrap_remove_hooks(self, model, handels_to_remove: List[torch.utils.hooks.RemovableHandle]):
        """
        remove hooks from model
        :param model:
        :param handels_to_remove:
        :return:
        """
        for handel in handels_to_remove:
            handel.remove()

    def extract_hs_include_prefix(self, list_inputs, list_outputs, info=''):
        def save_list_tokens(list_tokens, outorin):
            last_tokens = outorin[0].clone().detach().squeeze().cpu()
            while len(last_tokens.shape) > 2:
                last_tokens = last_tokens[0]
            for last_token in last_tokens:
                last_token = last_token.squeeze()
                list_tokens.append(last_token)

        def hook(module, input, output):
            assert info != self.last_hook_place, f"the same hook was called twice {info=} {self.last_hook_place=}"
            self.last_hook_place = info
            if list_inputs is not None and len(input) > 0:
                save_list_tokens(list_tokens=list_inputs, outorin=input)

            if list_outputs is not None and len(output) > 0:

                save_list_tokens(list_tokens=list_outputs, outorin=output)
            else:
                print(f"output is empty for {info=}")

        return hook

    def test_model_mlp_and_attention(self, model: AutoModelForCausalLM, prompts: List[str]):
        """
        collect the hidden states after each of those layers (modules) using the prompts
        :param model:
        :param prompts:
        :return: hidden states after each of those layers (modules)
        """
        model.requires_grad_(False)
        self.last_hook_place = ""
        # collect the hidden states before and after each of those layers (modules)
        hs_collector, handles = self.wrap_model(model, layers_to_check=
        ["." + self.model_condig["mlp_name"], "." + self.model_condig["attention_name"],
         "." + self.model_condig["attn_proj_name"], ""])
        encoded_line = self.tok.encode(prompts[0].rstrip(), return_tensors='pt').to(device)
        # print(f"{encoded_line=} {encoded_line.shape=}")
        with torch.no_grad():
            model(encoded_line, output_hidden_states=True, output_attentions=True, use_cache=False)
        torch.cuda.empty_cache()
        gc.collect()
        # remove hooks
        self.model_wrap_remove_hooks(model, handles)

        return hs_collector

    def get_mlp_and_attention_vectors(self, model, hs_collector):
        """
        get the residual, mlp and attention vectors for the last token
        :param model:
        :param hs_collector: the hidden states after each of the layers and inside the mlp and attention
        :return: mlp norm of all the layers using the last token, attention norm of all the layers using the last token,
        mlp vector of the last token, attention vector of the last token, residual stream vector of the last token
        """
        all_mlp_norm_mean = []
        all_attention_norm_mean = []
        last_token_mlp_vector = []
        last_token_attention_vector = []
        last_token_residual_stream_vector = []
        # collect the hidden states after mlp
        for layer, params in hs_collector.items():
            block_out = params[self.model_condig["mlp_name"]]["output"]
            norm_mean = torch.tensor([torch.linalg.vector_norm(v) for v in block_out])[-1]
            assert norm_mean == torch.linalg.vector_norm(block_out[-1])
            all_mlp_norm_mean.append(norm_mean.item())
            last_token_mlp_vector.append(block_out[-1].numpy())
        # collect the attention output vectors
        for layer, params in hs_collector.items():
            attention_out = params[self.model_condig["attention_name"]]["output"]
            attention_norm_mean = torch.tensor([torch.linalg.vector_norm(v) for v in attention_out])[-1]
            assert attention_norm_mean == torch.linalg.vector_norm(attention_out[-1])
            all_attention_norm_mean.append(attention_norm_mean.item())
            last_token_attention_vector.append(attention_out[-1].numpy())
        # collect the residual stream vectors
        for layer, params in hs_collector.items():
            residual_stream = params[""]["output"]
            residual_stream_norm_mean = torch.tensor([torch.linalg.vector_norm(v) for v in residual_stream])[-1]
            assert residual_stream_norm_mean == torch.linalg.vector_norm(residual_stream[-1])
            last_token_residual_stream_vector.append(residual_stream[-1].numpy())
        return np.array(all_mlp_norm_mean), np.array(all_attention_norm_mean), np.array(
            last_token_mlp_vector), np.array(last_token_attention_vector), np.array(last_token_residual_stream_vector)

    def attention_heads_no_projection(self, model, hs_collector):
        """
        get the attention heads for the last token without projection the heads. The size of the output is the size of the heads before projection (smaller than the hidden state)"""

        heads_outputs_for_all_the_layers_using_the_last_token = []
        dim_head = self.model_condig["hidden_size"] // self.model_condig["num_attention_heads"]
        row_idx = -1
        for layer_idx in range(self.model_condig["num_hidden_layers"]):
            layer_heads = []
            concated_heads_wihtout_projection = hs_collector[layer_idx][self.model_condig["attn_proj_name"]]['input'][
                row_idx]
            for head_idx in range(self.model_condig["num_attention_heads"]):
                assert dim_head * self.model_condig["num_attention_heads"] == self.model_condig["hidden_size"]
                head_output = concated_heads_wihtout_projection[dim_head * head_idx:dim_head * (
                        head_idx + 1)]
                head_output_projected = head_output
                layer_heads.append(np.array(head_output_projected))
            heads_outputs_for_all_the_layers_using_the_last_token.append(layer_heads)

        return np.array(heads_outputs_for_all_the_layers_using_the_last_token)

    def run_model(self, prompts: list[str], model: AutoModelForCausalLM):
        """
        run the model and collect the hidden states after each of the components at each layer
        :param prompts: list of prompts
        :param model: the model to run
        :return: the hidden states after each of the components at each layer
        """
        hs_collector = self.test_model_mlp_and_attention(model, prompts)

        all_mlp_norm_mean, all_attention_norm_mean, last_token_mlp_vector, last_token_attention_vector, last_token_residual = self.get_mlp_and_attention_vectors(
            model, hs_collector)
        heads_outputs_for_all_the_layers_using_the_last_token = self.attention_heads_no_projection(model, hs_collector)
        # del hs_collector
        return np.array(all_mlp_norm_mean), np.array(all_attention_norm_mean), np.array(
            last_token_mlp_vector), np.array(
            last_token_attention_vector), heads_outputs_for_all_the_layers_using_the_last_token, np.array(
            last_token_residual)

