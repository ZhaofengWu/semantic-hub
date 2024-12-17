"""
Adapted from https://colab.research.google.com/drive/1X2ZfC4y8Jx8FbkR7m-bLi8Ifrq-8MPTO
"""

from functools import partial
from typing import Optional, Union, Tuple, Callable, List, Dict, Any, cast

import torch
import transformer_lens
from transformer_lens.hook_points import NamesFilter
from transformer_lens.utils import Slice, SliceInput


def prepare_prompts(prompt_add: str, prompt_sub: str, model: torch.nn.Module) -> tuple:
    def tlen(prompt):
        return model.to_tokens(prompt).shape[1]

    def pad_right(prompt, length):
        coef = 1
        if "Llama-2" in model.tokenizer.name_or_path:
            coef = 16
        return prompt + " " * (length - tlen(prompt)) * coef

    l = max(tlen(prompt_add), tlen(prompt_sub))
    return pad_right(prompt_add, l), pad_right(prompt_sub, l)


def get_resid_pre(prompt: str, layer: int, model: torch.nn.Module) -> torch.Tensor:
    name = f"blocks.{layer}.hook_resid_pre"
    cache, caching_hooks, _ = model.get_caching_hooks(lambda n: n == name)
    with model.hooks(fwd_hooks=caching_hooks):
        _ = model(prompt)
    return cache[name]


def ave_hook(resid_pre, hook, act_diff, coeff):
    if resid_pre.shape[1] == 1:
        return
    ppos, apos = resid_pre.shape[1], act_diff.shape[1]
    assert apos <= ppos, f"More mod tokens ({apos}) than prompt tokens ({ppos})!"
    resid_pre[:, :apos, :] += coeff * act_diff


def hooked_generate(
    prompt_batch: List[str], editing_hooks: list, seed: int, model: torch.nn.Module, **kwargs
) -> torch.Tensor:
    if seed is not None:
        torch.manual_seed(seed)
    with model.hooks(fwd_hooks=editing_hooks):
        tokenized = model.to_tokens(prompt_batch)
        result = model.generate(
            input=tokenized, max_new_tokens=32, do_sample=True, verbose=False, **kwargs
        )
    return result


def generate_actadd(
    model,
    prompts: List[str],
    layer: int,
    prompt_add: str,
    prompt_sub: str,
    coeff: int,
    seed: int,
    sampling_kwargs: Dict[str, Any],
) -> List[str]:
    prompt_add, prompt_sub = prepare_prompts(prompt_add, prompt_sub, model)
    act_add = get_resid_pre(prompt_add, layer, model)
    act_sub = get_resid_pre(prompt_sub, layer, model)
    act_diff = act_add - act_sub
    editing_hooks = [
        (
            f"blocks.{layer}.hook_resid_pre",
            lambda resid_pre, hook: ave_hook(resid_pre, hook, act_diff, coeff),
        )
    ]
    results_tensor = hooked_generate(prompts, editing_hooks, seed, model, **sampling_kwargs)
    return model.tokenizer.batch_decode(results_tensor, skip_special_tokens=True)


def intervened_generate(
    prompt,
    model,
    act_name,
    prompt_add,
    prompt_sub,
    coeff,
    seed,
    sampling_kwargs,
):
    while True:
        try:
            prompt_lst = [prompt]
            output = generate_actadd(
                model, prompt_lst, act_name, prompt_add, prompt_sub, coeff, seed, sampling_kwargs
            )[0]
            break
        except Exception as e:
            error_message = str(e)
            print(
                f"Generate control text: something went wrong. Error: {error_message} Output: {output}. Retrying..."
            )
            break

    return output


# we perform this fix to the function in transformer_lens for version 1.17.0: https://github.com/neelnanda-io/TransformerLens/pull/578


def get_caching_hooks(
    self,
    names_filter: NamesFilter = None,
    incl_bwd: bool = False,
    device=None,
    remove_batch_dim: bool = False,
    cache: Optional[dict] = None,
    pos_slice: Union[Slice, SliceInput] = None,
) -> Tuple[dict, list, list]:
    """Creates hooks to cache activations. Note: It does not add the hooks to the model.

    Args:
        names_filter (NamesFilter, optional): Which activations to cache. Can be a list of strings (hook names) or a filter function mapping hook names to booleans. Defaults to lambda name: True.
        incl_bwd (bool, optional): Whether to also do backwards hooks. Defaults to False.
        device (_type_, optional): The device to store on. Keeps on the same device as the layer if None.
        remove_batch_dim (bool, optional): Whether to remove the batch dimension (only works for batch_size==1). Defaults to False.
        cache (Optional[dict], optional): The cache to store activations in, a new dict is created by default. Defaults to None.

    Returns:
        cache (dict): The cache where activations will be stored.
        fwd_hooks (list): The forward hooks.
        bwd_hooks (list): The backward hooks. Empty if incl_bwd is False.
    """
    if cache is None:
        cache = {}

    if not isinstance(pos_slice, Slice):
        if isinstance(
            pos_slice, int
        ):  # slicing with an int collapses the dimension so this stops the pos dimension from collapsing
            pos_slice = [pos_slice]
        pos_slice = Slice(pos_slice)

    if names_filter is None:
        names_filter = lambda name: True
    elif isinstance(names_filter, str):
        filter_str = names_filter
        names_filter = lambda name: name == filter_str
    elif isinstance(names_filter, list):
        filter_list = names_filter
        names_filter = lambda name: name in filter_list
    self.is_caching = True

    # mypy can't seem to infer this
    names_filter = cast(Callable[[str], bool], names_filter)

    def save_hook(tensor, hook, is_backward=False):
        hook_name = hook.name
        if is_backward:
            hook_name += "_grad"
        resid_stream = tensor.detach().to(device)
        if remove_batch_dim:
            resid_stream = resid_stream[0]

        # for attention heads the pos dimension is the third from last
        if (
            hook.name.endswith("hook_q")
            or hook.name.endswith("hook_k")
            or hook.name.endswith("hook_v")
            or hook.name.endswith("hook_z")
            or hook.name.endswith("hook_result")
        ):
            pos_dim = -3
        else:
            # for all other components the pos dimension is the second from last
            # including the attn scores where the dest token is the second from last
            pos_dim = -2

        if (
            tensor.dim() >= -pos_dim
        ):  # check if the residual stream has a pos dimension before trying to slice
            resid_stream = pos_slice.apply(resid_stream, dim=pos_dim)
        cache[hook_name] = resid_stream

    fwd_hooks = []
    bwd_hooks = []
    for name, hp in self.hook_dict.items():
        if names_filter(name):
            fwd_hooks.append((name, partial(save_hook, is_backward=False)))
            if incl_bwd:
                bwd_hooks.append((name, partial(save_hook, is_backward=True)))

    return cache, fwd_hooks, bwd_hooks


# Replace the original get_caching_hooks function
transformer_lens.hook_points.HookedRootModule.get_caching_hooks = get_caching_hooks
