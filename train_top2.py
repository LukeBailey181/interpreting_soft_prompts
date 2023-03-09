from transformers import Trainer, TrainingArguments, EarlyStoppingCallback
from transformers import AutoModelForCausalLM

from data import get_e2e
from modules import PromptInputEmbedding

def train_gpt2_with_prefix(
    model_checkpoint="gpt2",
    prompt_len=10,
    lr=8e-5,
    epochs=5,
    batch_size=10,
    early_stopping=False,
    model_output_dir="test-top2"
    """Keyword arguments:
    model_checkpoint -- Huggingface model checkpoint to instantiate pretrained model
    prompt_len -- numberer of tokens in continuous prompt
    lr -- learning rate for training
    epochs -- epochs for training
    batch_size -- batch size for training
    early_stopping -- if early stopping in training should be used 
    model_output_dir -- directory to save model checkpoints to whilst training. If 
        directory doesn't already exist it will be created.
    """

):
    # Load model and freeze all parameters
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    for name, param in model.named_parameters():
        if "transformer.h.11" not in name and "transformer.h.10" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

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
    train_gpt2()

