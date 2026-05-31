"""Experiment script to run PhaseQ optimizer on a language modeling task."""

import argparse
import logging
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import torch

from phaseq import PhaseQAdamW, PhaseQConfig
from phaseq.hooks import PhaseQTrainerCallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run PhaseQ Optimizer")
    parser.add_argument("--model_name", type=str, default="gpt2", help="Model name or path")
    parser.add_argument("--dataset_name", type=str, default="wikitext", help="Dataset name")
    parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1", help="Dataset config")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size per GPU")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--output_dir", type=str, default="./results_phaseq", help="Output directory")
    args = parser.parse_args()
    
    logger.info(f"Loading model {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    logger.info(f"Loading dataset {args.dataset_name} ({args.dataset_config})")
    dataset = load_dataset(args.dataset_name, args.dataset_config)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
        
    tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # 1. Initialize PhaseQ Optimizer
    logger.info("Initializing PhaseQ Optimizer")
    config = PhaseQConfig(
        lr=args.learning_rate,
        weight_decay=0.01,
        tau1=0.3,
        tau2=0.05
    )
    
    # We assign parameter names so PhaseQ can track them properly
    for name, param in model.named_parameters():
        param.name = name # Inject name for the optimizer param groups
        
    param_groups = [{"params": [p], "name": n} for n, p in model.named_parameters() if p.requires_grad]
    optimizer = PhaseQAdamW(param_groups, config=config)
    
    # Register model to detector and scheduler
    optimizer.register_model(model)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )
    
    # 2. Add PhaseQ Trainer Callback
    callback = PhaseQTrainerCallback()
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        optimizers=(optimizer, None), # Custom optimizer, default scheduler
        callbacks=[callback]
    )
    
    logger.info("Starting training with PhaseQ")
    trainer.train()

if __name__ == "__main__":
    main()
