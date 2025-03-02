import torch
from transformers import Trainer
from entropy_regularization import EntropyRegularization

class EntropyRegularizedTrainer(Trainer):
    """
    Custom Trainer that applies entropy regularization to transformer models.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Initialize the trainer with entropy regularization.
        
        Additional kwargs:
            loss_coeff (float): Coefficient for entropy regularization (default: 1e-5)
            tolerance_margin_factor (float): Margin factor (default: 0.20)
            context_size (int, optional): Size of the context window for calculating max entropy
        """
        # Extract custom parameters
        loss_coeff = kwargs.pop('loss_coeff', 1e-5)
        tolerance_margin_factor = kwargs.pop('tolerance_margin_factor', 0.20)
        context_size = kwargs.pop('context_size', None)
        
        super().__init__(*args, **kwargs)
        
        # Initialize entropy regularization
        self.entropy_regularizer = EntropyRegularization(
            loss_coeff=loss_coeff,
            tolerance_margin_factor=tolerance_margin_factor,
            context_size=context_size
        )
        self.logged_steps = set()
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute loss with entropy regularization.
        
        Args:
            model: The model to train
            inputs: The inputs to the model
            return_outputs (bool): Whether to return outputs along with loss
            
        Returns:
            torch.Tensor or tuple: Loss or (loss, outputs)
        """
        # Forward pass
        outputs = model(**inputs, output_attentions=True)
        
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
            
        # Standard CE loss
        ce_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        # Skip regularization during evaluation
        if not model.training:
            return (ce_loss, outputs) if return_outputs else ce_loss
        
        # Get model configuration
        num_layers = model.config.n_layer
        num_heads = model.config.n_head
        
        # Get attention matrices
        attention_matrices = outputs.attentions
        
        # Get regularization thresholds for each head
        # This assumes the model has a specific structure, adapt as needed
        reg_threshold_weights = []
        for layer_idx, block in enumerate(model.transformer.h):
            # Extract threshold weights from attention module
            # Assuming the structure matches your original code
            if hasattr(block.attn, 'reg_threshold_weights'):
                reg_threshold_weights = block.attn.reg_threshold_weights
                break
        
        # Compute entropy regularization loss
        entropy_reg_loss = self.entropy_regularizer.compute_loss(
            attention_matrices, 
            reg_threshold_weights,
            num_heads,
            num_layers
        )
        
        # Combine losses
        total_loss = ce_loss + entropy_reg_loss
        
        # Log metrics periodically
        if (self.state.global_step % 200 == 0 and 
            self.state.global_step not in self.logged_steps and
            "wandb" in self.args.report_to):
            
            self.logged_steps.add(self.state.global_step)
            
            # Log entropy metrics
            self.entropy_regularizer.log_entropy_metrics(
                attention_matrices,
                self.state.global_step,
                logger=self.args.report_to["wandb"]
            )
        
        return (total_loss, outputs) if return_outputs else total_loss


# Example usage:
"""
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Training arguments
training_args = TrainingArguments(
    output_dir="./output",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_steps=200,
    report_to=["wandb"],
)

# Initialize custom trainer
trainer = EntropyRegularizedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    
    # Entropy regularization parameters
    loss_coeff=1e-5,
    tolerance_margin_factor=0.20,
    context_size=1024,  # Set based on your model's context window
)

# Train model
trainer.train()
"""