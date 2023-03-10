from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForCausalLM

from data import get_e2e
from modules import PromptInputEmbedding

import numpy as np
import sys

def evaluate_gpt2_with_prefix(model_checkpoint="test-clm/checkpoint-243", eval_step=100, prompt_len=1, model_type="gpt2"):
    """Evaluate a gpt2 model with prefix on the e2e dataset.

    Keyword arguments:
    model_checkpoint -- path to checkpoint or "gpt2" to use pretrained gpt2.
    eval_step -- number of example to evaluate at once. Tune if GPU is running out of memory
    prompt_len -- continuous prompt length used in the model you are evaluating. Set to 0 
        if evaluating standard GPT2
    model_type -- huggingface model type, used to instantiate appropriate tokenizer
    """

    # Load model and freeze all parameters
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    for param in model.parameters():
        param.requires_grad = False

    # load e2e dataset 
    lm_datasets = get_e2e(model_type, prompt_len)

    # Setup Trainer just to use predict interface
    trainer = Trainer(
        model,
        args=TrainingArguments("test-clm", per_device_eval_batch_size=8),
    )

    # Run evaluation in batches of training set to avoid GPU OOM
    N = len(lm_datasets["test"])
    predictions = []
    for idx, k in enumerate(range(0, N, eval_step)):
        print(f"{idx} / {N // eval_step}")
        window = range(k, min(k+eval_step, N))
        preds = trainer.predict(lm_datasets["test"].select(window), metric_key_prefix="test_bleu")
        predictions.append(preds)
 
    #Calculate average BLEU metric across sets
    average_bleu = 0
    bleu_vals = []
    for p in predictions:
        bleu = p[2]['test_bleu_loss']
        average_bleu += bleu
        bleu_vals.append(bleu)

    print(bleu_vals)
    print("Average Bleu: ", average_bleu/len(predictions))

if __name__ == "__main__":

    # Evaluate trained model. Note you may have to change the checkpoint path.
    checkpoint_path = sys.argv[1]
    evaluate_gpt2_with_prefix(checkpoint_path, 200, 10)

    # This code evaluates plain gpt2
    #evaluate_gpt2_with_prefix("gpt2", 50, 0)
