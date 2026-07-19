import tensorflow as tf

class DelayedCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Delayed Cosine Learning Rate Decay Scheduler
    
    Keeps a fixed learning rate for the first N steps, then applies cosine decay.
    """
    
    def __init__(self, 
                 initial_learning_rate=5e-3,
                 decay_start_step=50000,
                 decay_steps=50000,
                 target_learning_rate=1e-4,
                 alpha=None,
                 name=None):
        """
        Initialize the delayed cosine decay scheduler.
        
        Args:
            initial_learning_rate: Initial learning rate (float)
            decay_start_step: Step at which to start decaying (int)
            decay_steps: Number of steps to apply cosine decay after decay_start_step (int)
            target_learning_rate: Final learning rate after cosine decay (float)
            alpha: Minimum learning rate as fraction of initial_learning_rate (float)
            name: Optional name for the schedule
        """
        super().__init__()
        self.initial_learning_rate = initial_learning_rate
        self.decay_start_step = decay_start_step
        self.decay_steps = decay_steps
        self.target_learning_rate = target_learning_rate
        self.name = name
        
        # Calculate alpha if not provided
        if alpha is None:
            self.alpha = target_learning_rate / initial_learning_rate
        else:
            self.alpha = alpha
        
        # Create the cosine decay schedule (only used after decay_start_step)
        self.cosine_decay = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            alpha=self.alpha,
            name=name
        )
    
    def __call__(self, step):
        """
        Get the learning rate for the given step.
        
        Args:
            step: Current training step (int)
            
        Returns:
            Learning rate for the current step (float)
        """
        step = tf.cast(step, tf.float32)
        decay_start_step = tf.cast(self.decay_start_step, tf.float32)
        decay_steps = tf.cast(self.decay_steps, tf.float32)
        
        # Fixed learning rate phase
        fixed_lr = tf.cast(self.initial_learning_rate, tf.float32)
        
        # Minimum learning rate after decay
        min_lr = tf.cast(self.initial_learning_rate * self.alpha, tf.float32)
        
        # Calculate decay step
        decay_step = step - decay_start_step
        
        # Use TensorFlow operations for conditional logic
        # If step < decay_start_step, return fixed_lr
        # Else if decay_step >= decay_steps, return min_lr
        # Else return cosine_decay(decay_step)
        
        # Check if we're in the fixed phase
        in_fixed_phase = step < decay_start_step
        
        # Check if we're past the decay phase
        past_decay_phase = decay_step >= decay_steps
        
        # Calculate cosine decay value
        cosine_lr = self.cosine_decay(decay_step)
        
        # Use tf.where to handle conditional logic
        lr = tf.where(
            in_fixed_phase,
            fixed_lr,
            tf.where(
                past_decay_phase,
                min_lr,
                cosine_lr
            )
        )
        
        return lr
    
    def get_config(self):
        """Get configuration dictionary for serialization."""
        return {
            'initial_learning_rate': self.initial_learning_rate,
            'decay_start_step': self.decay_start_step,
            'decay_steps': self.decay_steps,
            'target_learning_rate': self.target_learning_rate,
            'alpha': self.alpha,
            'name': self.name
        }
