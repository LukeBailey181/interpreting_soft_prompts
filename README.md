# Prefix-Tuning Replication 

Replication of experiment described in section 7.2 of "Prefix-Tuning:
Optimizing Continuous Prompts for Generation" by Xiang Lisa Li and Percy Liang.
For full details, including descriptions of what each Python file in this repo
does see the accompanying Checkpoint 2 writeup pdf. 

## Running Code 

`python train.py` will prefix-tune a GPT-2 model with the same hyperparameters
as that used by Li et al. It will save a checkpoint of the model after each
epoch of training to a directory called "test-clm". Note we advise running on a
GPU to ensure training time is not too long.

To evaluate the model, run `python evaluate.py path/to/model/checkpoint`. This
will report a BLEU score for the model on the E2E task. Note that GPUs may OOM
when running `evaluate.py`. To avoid this edit the file reducing the
`eval_step` argument of the `evaluate_gpt2_with_prefix` function.
