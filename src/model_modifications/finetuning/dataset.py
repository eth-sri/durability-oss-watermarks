from datasets import load_dataset
from itertools import chain
from strenum import StrEnum
from .data_utils import (
    convert_sft_dataset,
    tokenize_dataset_with_chat,
    add_chat_template,
)


class DatasetType(StrEnum):
    learnability_adv = "learnability_adv"
    OpenMathInstruct = "OpenMathInstruct"
    Dummy = "dummy" # For testing purposes

def update_tokenizer(tokenizer, dataset_type: DatasetType):
    if dataset_type == DatasetType.OpenMathInstruct:
        tokenizer = add_chat_template(tokenizer=tokenizer)
    return tokenizer

def get_dataset(tokenizer, dataset_type: DatasetType):
    dataset, eval_dataset = None, None

    if dataset_type == DatasetType.Dummy:
        dataset = load_dataset("Skylion007/openwebtext", split="train[0:5000]", trust_remote_code=True)
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=256)

    elif dataset_type == DatasetType.learnability_adv:
        dataset = load_dataset("Skylion007/openwebtext", split="train[0:500000]")
        dataset = tokenize_dataset(dataset, tokenizer, sequence_length=min(512, tokenizer.model_max_length))

    elif dataset_type == DatasetType.OpenMathInstruct:
        dataset = load_dataset("nvidia/OpenMathInstruct-2", split="train_1M[0:500000]")

        conversion_func = lambda example: {  # noqa: E731
            "messages": [
                {"role": "user", "content": example["problem"]},
                {"role": "assistant", "content": example["generated_solution"]},
            ]
        }

        dataset = convert_sft_dataset(
            ds=dataset,
            convert_fn=conversion_func,
            min_response_length=200,
        )

        tokenizer = add_chat_template(tokenizer=tokenizer)
        dataset = tokenize_dataset_with_chat(
            dataset=dataset, tokenizer=tokenizer, max_length=min(2048, tokenizer.model_max_length)
        )

    else:
        raise ValueError("Unknown dataset type")

    return dataset, eval_dataset, tokenizer


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"])


def group_texts(examples, sequence_length):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // sequence_length) * sequence_length
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + sequence_length] for i in range(0, total_length, sequence_length)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def tokenize_dataset(dataset, tokenizer, sequence_length: int = 200):
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True,
        remove_columns="text",
    )
    lm_dataset = tokenized_dataset.map(
        lambda examples: group_texts(examples, sequence_length),
        batched=True,
    )

    return lm_dataset
