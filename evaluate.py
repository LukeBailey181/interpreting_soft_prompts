from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForCausalLM

from data import get_e2e
from modules import PromptInputEmbedding

import numpy as np

def evaluate_gpt2_with_prefix(model_checkpoint="test-clm/checkpoint-243", eval_step=100, prompt_len=1):

    # Load model and freeze all parameters
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    for param in model.parameters():
        param.requires_grad = False

    # load e2e dataset 
    lm_datasets = get_e2e("distilgpt2", prompt_len)

    # Setup Trainer just to use predict interface
    trainer = Trainer(
        model,
        args=TrainingArguments("test-clm", per_device_eval_batch_size=8),
    )

    # Run evaluation in batches of training set to avoid GPU OOM
    N = len(lm_datasets["test"])
    predictions = []
    for k in range(0, N, eval_step):
        window = range(k, min(k+eval_step, N))
        preds = trainer.predict(lm_datasets["test"].select(window), metric_key_prefix="test_bleu")
        predictions.append(preds)
 
    #Calculate average BLEU metric across sets
    average_bleu = 0
    for p in predictions:
        bleu = p[2]['test_bleu_loss']
        average_bleu += bleu
    print("Average Bleu: ", average_bleu/len(predictions))

if __name__ == "__main__":
    evaluate_gpt2_with_prefix("test-clm/checkpoint-972", 50)

#Average Bleu:  55.19827842712402