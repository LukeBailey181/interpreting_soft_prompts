from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForCausalLM

from data import get_e2e
from modules import PromptInputEmbedding

def train_gpt2_with_prefix(
    model_checkpoint="distilgpt2",
    prompt_len=1,
    lr=8e-5,
    epochs=4,
    batch_size=10,
    early_stopping=False,
    model_output_dir="test-clm"
):

    # Load model and freeze all parameters
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)
    original_wte = model.transformer.wte    # Embedding params
    for param in model.parameters():
        param.requires_grad = False

    model.transformer.wte = PromptInputEmbedding(original_wte, prompt_len, True, 512)

    # load e2e dataset 
    lm_datasets = get_e2e(model_checkpoint, prompt_len)

    # Prepare training arguments
    training_args = TrainingArguments(
        model_output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        weight_decay=0.01,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        metric_for_best_model="eval_loss",
        load_best_model_at_end=True,
    )

    callbacks = []

    if early_stopping:
        patience = 10
        threshold = 0.
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=patience,
            early_stopping_threshold=threshold
        )
        callbacks.append(early_stopping)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_datasets["train"],
        eval_dataset=lm_datasets["validation"],
        callbacks=callbacks,
    )

    # Train model 
    trainer.train()

if __name__ == "__main__":
    train_gpt2_with_prefix()

