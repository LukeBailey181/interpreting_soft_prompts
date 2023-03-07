from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForCausalLM

from data import get_e2e
from modules import PromptInputEmbedding

import numpy as np

def train_gpt2_with_prefix(model_checkpoint="test-clm/checkpoint-243", prompt_len=1):

    # Load model and freeze all parameters
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    # original_wte = model.transformer.wte    # Embedding params
    for param in model.parameters():
        param.requires_grad = False

    # load e2e dataset 
    lm_datasets = get_e2e("distilgpt2", prompt_len)

    # Prepare training arguments
    training_args = TrainingArguments(
        "test-clm",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-3,
        weight_decay=0.01,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
        per_device_eval_batch_size=8
    )

    callbacks = []

    patience = 10
    threshold = 0.
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=patience,
        early_stopping_threshold=threshold
    )
    callbacks.append(early_stopping)

    #Set up trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        callbacks=callbacks,
    )

    # Use model to predict splits in the test data
    predictions1 = trainer.predict(lm_datasets["test"].select(range(100)), metric_key_prefix="test_bleu")
    predictions2 = trainer.predict(lm_datasets["test"].select(range(100, 200)), metric_key_prefix="test_bleu")
    predictions3 = trainer.predict(lm_datasets["test"].select(range(200, 300)), metric_key_prefix="test_bleu")
    predictions4 = trainer.predict(lm_datasets["test"].select(range(300, 374)), metric_key_prefix="test_bleu")

    #Calculate average BLEU metric across sets
    predictions = [predictions1, predictions2, predictions3, predictions4 ]
    average_bleu = 0
    for p in predictions:
        bleu = p[2]['test_bleu_loss']
        average_bleu += bleu
    print("Average Bleu: ", average_bleu/4)

if __name__ == "__main__":
    train_gpt2_with_prefix()

#Average Bleu:  55.19827842712402