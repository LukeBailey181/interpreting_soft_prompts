from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForCausalLM,  AutoTokenizer

from data import get_e2e
from modules import PromptInputEmbedding

import numpy as np

def evaluate_gpt2_bleu(model_checkpoint="test-clm/checkpoint-243", eval_step=100, prompt_len=1, model_type="gpt2"):
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

def generate_gpt2(model_checkpoint="test-clm/checkpoint-655", dir = "test-clm",eval_step=100, prompt_len=1, model_type="gpt2"):
    datasets = get_e2e()
    tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)

    prompt = "!!!!!!!!!!Summarize: name[The Vaults], eatType[pub], priceRange[more than £30], customer rating[5 out of 5], near[Café Adriatic]\nAnswer:"
    label = "The Vaults pub near Café Adriatic has a 5 star rating.  Prices start at £30.<|endoftext|>"

    gpt2 = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    greedy_output = gpt2.generate(input_ids, max_length=100)

    print("Output:\n" + 100 * '-')
    print(tokenizer.decode(greedy_output[0], skip_special_tokens=False))

if __name__ == "__main__":
     #Prompt model, prompt len = 10
    # evaluate_gpt2_bleu("test-clm/checkpoint-243", "test-clm", 50, 10)

    # generate_gpt2("test-clm/checkpoint-243", "test-clm", 50, 10)

    #Top 2 layered model, Prompt len = 0
    evaluate_gpt2_bleu("test-top22/checkpoint-655", "test-top2", 50, 10)

    generate_gpt2("test-top22/checkpoint-655", "test-top2", 200, 10)

    # This code evaluates plain gpt2
    #evaluate_gpt2_with_prefix("gpt2", 50, 0)