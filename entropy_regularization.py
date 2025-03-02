import torch

class EntropyRegularization:
    """
    Module for applying entropy regularization to attention matrices in transformer models.
    
    This module computes entropy-based regularization losses for attention matrices,
    helping to control the entropy of attention distributions in transformers.
    """
    
    def __init__(self, loss_coeff=1e-5, tolerance_margin_factor=0.20, context_size=None):
        """
        Initialize the entropy regularization module.
        
        Args:
            loss_coeff (float): Coefficient for the entropy regularization loss when added to CE loss
            tolerance_margin_factor (float): Factor of max entropy to use as tolerance margin
            context_size (int, optional): Size of the context window for calculating max entropy.
                                         If None, will be determined from the attention matrices.
        """
        self.loss_coeff = loss_coeff
        self.tolerance_margin_factor = tolerance_margin_factor
        self.context_size = context_size
        
    def compute_loss(self, attentions, reg_threshold_weights, num_heads, num_layers):
        """
        Compute entropy regularization loss for attention matrices.
        
        Args:
            attentions (list): List of attention matrices from model
            reg_threshold_weights (list): Threshold weights for each attention head
            num_heads (int): Number of attention heads
            num_layers (int): Number of transformer layers
            
        Returns:
            torch.Tensor: Entropy regularization loss
        """
        device = attentions[0].device
        
        # Determine context size if not provided
        context_size = self.context_size
        if context_size is None:
            # Infer from attention matrix shape
            context_size = attentions[0].shape[-1]
        
        # Calculate max entropy based on context size
        max_entropy = torch.log(torch.tensor(float(context_size), device=device))
        tolerance_margin = self.tolerance_margin_factor * max_entropy
        
        entropy_reg_loss = 0
        
        for layer_idx, attn_mat in enumerate(attentions):
            # Calculate entropy for each attention matrix
            ent_val = -torch.nansum(attn_mat * torch.log(attn_mat + 1e-9), dim=-1).to(device)
            layer_entropy_reg_loss = 0
            
            for head_idx in range(num_heads):
                head_entropy = ent_val[:, head_idx, :]
                threshold = reg_threshold_weights[head_idx] * max_entropy
                
                # Calculate deviation from threshold
                deviation = head_entropy - threshold
                
                # Penalize deviations (square the deviation) beyond tolerance margin
                penalty = torch.square(
                    torch.where(
                        torch.abs(deviation) > tolerance_margin, 
                        deviation,
                        torch.zeros_like(deviation)
                    )
                )
                layer_entropy_reg_loss += penalty.sum()
            
            # Average across heads
            layer_entropy_reg_loss /= num_heads
            entropy_reg_loss += layer_entropy_reg_loss
        
        # Average across layers
        entropy_reg_loss /= num_layers
        
        # Apply loss coefficient
        return self.loss_coeff * entropy_reg_loss
    
    @staticmethod
    def compute_entropy(attention_matrix):
        """
        Compute entropy of attention matrix.
        
        Args:
            attention_matrix (torch.Tensor): Attention probability matrix
            
        Returns:
            torch.Tensor: Entropy values for the attention matrix
        """
        return -torch.nansum(
            attention_matrix * torch.log(attention_matrix + 1e-9), 
            dim=-1
        )
    
    @staticmethod
    def log_entropy_metrics(attention_matrices, step, logger=None):
        """
        Calculate and log entropy metrics for attention matrices.
        
        Args:
            attention_matrices (list): List of attention matrices from model
            step (int): Current training step
            logger: Logger to use (e.g., WandB)
            
        Returns:
            dict: Dictionary of entropy metrics
        """
        metrics = {}
        
        for i, attn_mat in enumerate(attention_matrices):
            ent = -torch.nansum(attn_mat * torch.log(attn_mat + 1e-9), dim=-1)
            
            for head_index in range(ent.shape[1]):
                head_entropy = ent[:, head_index, :].mean()
                metrics[f"attn.{i}.head.{head_index}.entropy"] = head_entropy
        
        if logger is not None:
            logger.log(metrics)
            
        return metrics