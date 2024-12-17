import ast
import builtins
import inspect
import os
import pickle
import sys

from datasets import load_dataset
from matplotlib import colors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.logit_lens import init_model, get_hidden_logprobs, first_token_id_for_string


def get_character_offsets(program: str, expr: ast.expr) -> tuple[int, int]:
    lines = program.split("\n")

    start = 0
    for line in lines[: expr.lineno - 1]:  # expr.lineno is 1-base-indexed
        start += len(line) + 1  # +1 for the newline character
    start += expr.col_offset

    end = 0
    for line in lines[: expr.end_lineno - 1]:  # expr.lineno is 1-base-indexed
        end += len(line) + 1  # +1 for the newline character
    end += expr.end_col_offset

    if expr.lineno == expr.end_lineno:
        assert end == start + expr.end_col_offset - expr.col_offset

    assert ast.unparse(ast.parse(program[start:end]).body[0].value) == ast.unparse(expr)
    return start, end


def get_builtin_argument_name(function_name: str, arg_idx: int) -> str:
    # all used builtins in mbpp:
    # {'abs', 'all', 'any', 'bin', 'bool', 'chr', 'complex', 'dict', 'enumerate', 'filter', 'float', 'int', 'isinstance', 'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'ord', 'pow', 'range', 'reversed', 'round', 'set', 'sorted', 'str', 'sum', 'tuple', 'type', 'zip'}

    # These functions have overloaded versions have different first-args
    blacklist_functions = {"range", "max", "min", "dict", "zip", "type"}
    if function_name in blacklist_functions:
        raise NotImplementedError

    try:
        fn = getattr(builtins, function_name)
        return list(inspect.signature(fn).parameters)[arg_idx]
    except ValueError:
        # Some builtins can't be inspected, idk why. So we hardcode them here.
        # These are manually taken from builtins.pyi
        builtin_args = {
            "bool": ["o"],
            "int": ["x", "base"],
            "float": ["x"],
            "filter": ["function", "iterable"],
            "map": ["func", "iter1", "iter2"],
            "next": ["i", "default"],
            "iter": ["object", "sentinel"],
            "set": ["iterable"],
            "str": ["object", "encoding", "errors"],
        }
        return builtin_args[function_name][arg_idx]


def get_contextual_argument_name(function_name: str, arg_idx: int, program: str) -> str:
    tree = ast.parse(program)
    function_defs = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
    function_defs = [fn for fn in function_defs if fn.name == function_name]
    assert len(function_defs) <= 1  # assumes no overloading
    if len(function_defs) == 0:
        # Sometimes there are adhoc imports from e.g. math; not sure how to handle those
        raise NotImplementedError
    function_def = function_defs[0]
    arg = function_def.args.args[arg_idx]
    return arg.arg


def load_mbpp(include_tests=False, return_instructions=False):
    dataset_dict = load_dataset("google-research-datasets/mbpp", "full")
    programs = [
        ex["code"] + (("\n" + "\n".join(ex["test_list"])) if include_tests else "")
        for dataset in dataset_dict.values()
        for ex in dataset
    ]
    instructions = [ex["text"] for dataset in dataset_dict.values() for ex in dataset]
    if return_instructions:
        return programs, instructions
    else:
        return programs


CACHE_FILE = "semantic_role_cache_llama2_mbpp_notest.pkl"


def main(model_path):
    if os.path.exists(CACHE_FILE):
        per_arg_data = pickle.load(open(CACHE_FILE, "rb"))
    else:
        tokenizer, model = init_model(model_path)
        # programs = load_humaneval()
        programs = load_mbpp(include_tests=False)

        per_arg_data = []
        for program in tqdm(programs):
            tree = ast.parse(program)

            function_calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
            function_calls = [
                call
                for call in function_calls
                if isinstance(call.func, ast.Name)
                and len(call.args) > 0  # i.e. not a.b() & has args
            ]
            if len(function_calls) == 0:
                continue

            logprobs = get_hidden_logprobs(program, tokenizer, model).cpu()
            encoded = tokenizer.encode_plus(program, return_offsets_mapping=True)
            input_ids = encoded["input_ids"]
            token_offsets = encoded["offset_mapping"]

            for call in function_calls:
                for arg_idx in range(len(call.args)):
                    # "key" refers to open paren for the first arg and comma for the rest
                    prev_element = call.args[arg_idx - 1] if arg_idx > 0 else call.func
                    _, key_char_idx = get_character_offsets(program, prev_element)
                    # Sometimes it's in the form of func((arg1), arg2), in which case the offset doesn't include the closing ). We need to manually skip it
                    while program[key_char_idx] in {" ", ")"}:
                        key_char_idx += 1
                    assert program[key_char_idx] in {"(", ","}

                    token_indices_that_end_in_key = [
                        token_idx
                        for token_idx, (s, e) in enumerate(token_offsets)
                        if key_char_idx == e - 1
                    ]
                    assert len(token_indices_that_end_in_key) <= 1
                    if len(token_indices_that_end_in_key) == 0:
                        # Sometimes no token ends in a key ("(" or ",") because they are merged with the
                        # next punctuation (or in the case of llama3, any character), e.g. "([" where "[" is a part of the arg. In this case it's
                        # impossible to extract the probability of the arg.
                        # assert program[key_char_idx + 1] in string.punctuation
                        continue
                    key_token_idx = token_indices_that_end_in_key[0]
                    assert tokenizer.convert_ids_to_tokens(input_ids[key_token_idx])[-1] in {
                        "(",
                        ",",
                    }

                    fn_name = call.func.id
                    try:
                        if fn_name in dir(builtins):
                            arg_name = get_builtin_argument_name(fn_name, arg_idx)
                        else:
                            arg_name = get_contextual_argument_name(fn_name, arg_idx, program)
                    except NotImplementedError:
                        continue
                    if arg_name == ast.unparse(call.args[arg_idx]):
                        # semantic role == surface form
                        continue
                    if ast.unparse(call.args[arg_idx])[0] == "(":
                        # This is a special case. "((" is a single token in llama2's vocab, so it's unlikely to appear as separate tokens
                        continue
                    if arg_name in {"obj", "object"}:
                        continue

                    arg_token_id = first_token_id_for_string(tokenizer, arg_name)
                    probs = logprobs[:, key_token_idx, arg_token_id]

                    surface_arg_token_id = first_token_id_for_string(
                        tokenizer, ast.unparse(call.args[arg_idx])
                    )
                    surface_probs = logprobs[:, key_token_idx, surface_arg_token_id]

                    if arg_token_id == surface_arg_token_id:
                        # semantic role == surface form
                        continue

                    per_arg_data.append(
                        [
                            fn_name,
                            arg_name,
                            ast.unparse(call.args[arg_idx]),
                            probs.tolist(),
                            surface_probs.tolist(),
                            (probs - surface_probs).tolist(),
                        ]
                    )
        pickle.dump(per_arg_data, open(CACHE_FILE, "wb"))

    # create discrete colormap
    cmap = colors.ListedColormap(["#6b8a97", "#e29dbc"])
    bounds = [-1e10, 0, 1e10]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    # fig, ax = plt.subplots()
    data = [x[-1] for x in per_arg_data]
    # transpose data
    data = list(zip(*data))
    plt.imshow(data, origin="lower", cmap=cmap, norm=norm)
    # legend outside of plot
    plt.legend(
        handles=[
            mpatches.Patch(color="#6b8a97", label="Hidden closer to surface argument"),
            mpatches.Patch(color="#e29dbc", label="Hidden closer to semantic role"),
        ],
        # ncol=2,
        bbox_to_anchor=(0.85, 3.6),
    )
    # ticks only at 1 and 32
    plt.yticks([0, 31], ["1", "32"])
    plt.ylabel("Layer")
    plt.xlabel("All Function Arguments in MBPP")
    plt.tight_layout(pad=0)

    plt.savefig("code_semantic_role.pdf")


if __name__ == "__main__":
    try:
        main(*sys.argv[1:])  # pylint: disable=no-value-for-parameter,too-many-function-args
    except Exception as e:
        import pdb
        import traceback

        if not isinstance(e, (pdb.bdb.BdbQuit, KeyboardInterrupt)):
            print("\n" + ">" * 100 + "\n")
            traceback.print_exc()
            print()
            pdb.post_mortem()
