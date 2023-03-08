from datasets import load_dataset
from transformers import AutoTokenizer

def get_e2e(model_checkpoint="distilgpt2", prompt_len=1, combined_block_size=512):
    """Load e2e dataset and tokeize"""

    def tokenize_e2e_test(examples):
        """Helper function for tokenizing e2e test set"""
        input_prompt = [
            "Summarize: " + x + "\n 
            for x examples["meaning_representation"])
        ]
        label = [
            "Answer: " + y + tokenizer.eos_token 
            for y in examples["human_reference"]
        ]
        d = {
            "input_ids" : tokenizer(input_prompt)["input_ids"]
            "attention_mask" : tokenizer(input_prompt)["attention_mask"]
            "labels" : tokenizer(label)["attention_mask"]
        }

        return d

    def tokenize_e2e_seq2seq(examples):
        """Helper function for tokenizing e2e dataset"""

        prompt = [
            "Summarize: " + x + "\nAnswer: " + y + tokenizer.eos_token 
            for x, y in zip(examples["meaning_representation"], examples["human_reference"])
        ]
        #tokenizer_args = {padding:"max_length", truncation:True, max_length=512}
        tokenizer_args = {}
        d = {
            "input_ids": tokenizer(prompt, **tokenizer_args)["input_ids"],
            "attention_mask": tokenizer(prompt, **tokenizer_args)["attention_mask"],
            "labels": tokenizer(prompt, **tokenizer_args)["input_ids"],
        }

        return d

    def tokenize_e2e(examples):
        """Helper function for tokenizing e2e dataset"""

        d = {
            "input_ids": tokenizer(examples["meaning_representation"])["input_ids"],
            "attention_mask": tokenizer(examples["meaning_representation"])[
                "attention_mask"
            ],
            "labels": tokenizer(examples["human_reference"])["input_ids"],
        }

        return d

    def group_texts(examples):
        """Helper function to group short examples together for faster training. Note
        This is not necessary, and is only used to accelerate training"""

        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size

        dummy_input_prefix = [0] * prompt_len
        dummy_label_prefix = [-100] * prompt_len
        dummy_attention_mask_prefix = [1] * prompt_len

        # Split by chunks of max_len.
        input_ids = concatenated_examples["input_ids"]
        attention_masks = concatenated_examples["attention_mask"]
        result = {}

        result["input_ids"] = [
            dummy_input_prefix + input_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        result["labels"] = [
            dummy_label_prefix + input_ids[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]
        result["attention_mask"] = [
            dummy_attention_mask_prefix + attention_masks[i : i + block_size]
            for i in range(0, total_length, block_size)
        ]

        return result

    def add_prompt_dummy_tokens(examples):
        """Helper function to group short examples together for faster training. Note
        This is not necessary, and is only used to accelerate training"""

        dummy_input_prefix = [0] * prompt_len
        dummy_label_prefix = [-100] * prompt_len
        dummy_attention_mask_prefix = [1] * prompt_len

        # Split by chunks of max_len.
        input_ids = examples["input_ids"]
        attention_masks = examples["attention_mask"]
        labels = examples["labels"]
        result = {}

        result["input_ids"] = [
            dummy_input_prefix + x for x in input_ids
        ]
        result["labels"] = [
            dummy_label_prefix + x for x in labels
        ]
        result["attention_mask"] = [
            dummy_attention_mask_prefix + x for x in attention_masks
        ]

        return result

    # Load ataset
    assert prompt_len < combined_block_size
    block_size = combined_block_size - prompt_len
    datasets = load_dataset("e2e_nlg")

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    tokenizer.add_special_tokens({
        "pad_token": tokenizer.eos_token,
    })
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})



    tokenized_datasets = datasets.map(
        tokenize_e2e_seq2seq,
        num_proc=4,
        batched=True,
        remove_columns=["meaning_representation", "human_reference"],
    )
    """
    # Process dataset
    tokenized_datasets = datasets.map(
        tokenize_e2e,
        num_proc=4,
        batched=True,
        remove_columns=["meaning_representation", "human_reference"],
    )
    """
    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=4,
    )
    """
    lm_datasets = tokenized_datasets.map(
        add_prompt_dummy_tokens,
        batched=True,
        num_proc=4,
    )
    """

    for example in lm_datasets["test"]:
        assert(len(example["input_ids"]) == len(example["labels"]))
        assert(len(example["input_ids"]) == len(example["attention_mask"]))

    return lm_datasets
