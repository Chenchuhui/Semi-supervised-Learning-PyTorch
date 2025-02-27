from copy import deepcopy

import torch


class ModelEMA(object):
    def __init__(self, args, model, decay):
        from copy import deepcopy
        self.ema = deepcopy(model)  # Copy model structure for EMA
        self.ema.to(args.device)
        self.ema.eval()
        self.decay = decay
        self.ema_has_module = hasattr(self.ema, 'module')

        # Get parameter and buffer keys
        self.param_keys = [k for k, _ in self.ema.named_parameters()]
        self.buffer_keys = [k for k, _ in self.ema.named_buffers()]
        
        # Disable gradient computation for EMA model
        for p in self.ema.parameters():
            p.requires_grad_(False)

        # Define weight decay proportional to learning rate
        self.wd = 0.02 * args.lr  # Equivalent to WeightEMA's wd

    def update(self, model):
        """
        Update the EMA model parameters using exponential moving average 
        and apply weight decay to prevent overfitting on the main model.
        """
        needs_module = hasattr(model, 'module') and not self.ema_has_module
        with torch.no_grad():
            msd = model.state_dict()  # Main model state dict
            esd = self.ema.state_dict()  # EMA model state dict
            
            for k in self.param_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                
                model_v = msd[j].detach()
                ema_v = esd[k]

                # Apply EMA update (same as before)
                esd[k].copy_(ema_v * self.decay + (1. - self.decay) * model_v)



            # Update buffers (like BatchNorm running statistics)
            for k in self.buffer_keys:
                if needs_module:
                    j = 'module.' + k
                else:
                    j = k
                esd[k].copy_(msd[j])


