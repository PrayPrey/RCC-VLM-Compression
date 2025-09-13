# RCC System - Detailed Logic Design Specifications
## Phase 3-2: Implementation-Ready Component Logic

---

## 1. Compression Methods

### 1.1 DARECompressor Class

#### Class: DARECompressor
**Purpose**: Implements magnitude-based unstructured pruning with gradual sparsification

**Attributes**:
- model: nn.Module - Target model to compress
- target_sparsity: float - Final sparsity level (0.0 to 1.0)
- current_sparsity: float - Current sparsity level
- pruning_schedule: str - Schedule type ('linear', 'exponential', 'cosine')
- num_iterations: int - Total pruning iterations
- current_iteration: int - Current iteration counter
- pruning_masks: Dict[str, torch.Tensor] - Binary masks for each layer
- original_weights: Dict[str, torch.Tensor] - Backup of original weights
- layer_importance: Dict[str, float] - Importance scores per layer
- device: torch.device - Computation device (cuda/cpu)

**Methods**:

#### __init__(model, target_sparsity, num_iterations, schedule='linear', device='cuda')
- Parameters:
  - model: nn.Module - Model to compress
  - target_sparsity: float (0.0-1.0) - Target pruning rate
  - num_iterations: int - Number of pruning steps
  - schedule: str - Sparsity schedule type
  - device: str - Device for computation
- Logic:
  1. Store model reference and configuration
  2. Initialize current_sparsity to 0.0
  3. Create empty pruning_masks dictionary
  4. Identify prunable layers (Conv2d, Linear) using model.named_modules()
  5. For each prunable layer:
     - Initialize mask with ones: shape = layer.weight.shape
     - Store in pruning_masks[layer_name]
  6. Clone and store original weights for potential restoration
  7. Register forward hooks for gradient tracking if needed
  8. Validate target_sparsity is achievable (not > 0.99)

#### compute_layer_importance(layer_name, weight_tensor) -> float
- Parameters:
  - layer_name: str - Name of the layer
  - weight_tensor: torch.Tensor - Layer weights [out_features, in_features]
- Returns: float - Importance score (0.0 to 1.0)
- Logic:
  1. Compute L2 norm of weight tensor: torch.norm(weight_tensor, p=2)
  2. Calculate gradient magnitude if gradients available:
     - grad_norm = torch.norm(weight_tensor.grad, p=2) if weight_tensor.grad is not None
     - else grad_norm = 0
  3. Compute variance of weights: torch.var(weight_tensor)
  4. Calculate importance = 0.4 * norm + 0.4 * grad_norm + 0.2 * variance
  5. Normalize by layer size: importance / weight_tensor.numel()
  6. Return clamped value between 0.0 and 1.0

#### calculate_sparsity_schedule(iteration) -> float
- Parameters:
  - iteration: int - Current iteration number
- Returns: float - Target sparsity for this iteration
- Logic:
  1. Calculate progress ratio: t = iteration / num_iterations
  2. If schedule == 'linear':
     - sparsity = target_sparsity * t
  3. If schedule == 'exponential':
     - sparsity = target_sparsity * (1 - exp(-5 * t))
  4. If schedule == 'cosine':
     - sparsity = target_sparsity * 0.5 * (1 - cos(pi * t))
  5. If schedule == 'polynomial':
     - sparsity = target_sparsity * (t ** 3)
  6. Clamp between 0 and target_sparsity
  7. Return sparsity value

#### apply_pruning_step() -> Dict[str, float]
- Parameters: None
- Returns: Dict[str, float] - Sparsity per layer
- Logic:
  1. Get target sparsity for current iteration:
     - target = calculate_sparsity_schedule(current_iteration)
  2. Calculate delta_sparsity = target - current_sparsity
  3. For each prunable layer in model:
     - Get weight tensor and current mask
     - Calculate number of weights to prune:
       - num_prune = int(delta_sparsity * weight.numel())
     - Compute importance scores for unpruned weights:
       - active_weights = weight[mask == 1]
       - scores = torch.abs(active_weights).flatten()
     - Find threshold using torch.kthvalue:
       - threshold = torch.kthvalue(scores, num_prune).values
     - Update mask: mask[torch.abs(weight) < threshold] = 0
     - Apply mask: weight.data.mul_(mask)
  4. Update current_sparsity = target
  5. Increment current_iteration
  6. Calculate and return per-layer sparsity statistics

#### rescale_weights(scale_factor=None) -> None
- Parameters:
  - scale_factor: float - Optional manual scaling factor
- Logic:
  1. If scale_factor is None:
     - Calculate automatic scale based on remaining weights:
     - scale_factor = 1.0 / (1.0 - current_sparsity)
  2. For each prunable layer:
     - Get weight and mask
     - Count active weights: num_active = mask.sum()
     - If num_active > 0:
       - Apply scaling: weight.data[mask == 1] *= scale_factor
     - Store scale factor for layer
  3. Log rescaling statistics

#### get_compression_stats() -> Dict
- Parameters: None
- Returns: Dict - Comprehensive compression statistics
- Logic:
  1. Initialize stats dictionary
  2. Calculate total parameters: sum of weight.numel() for all layers
  3. Calculate pruned parameters: sum of (mask == 0).sum() for all masks
  4. Compute overall sparsity: pruned / total
  5. For each layer:
     - Calculate layer sparsity: (mask == 0).sum() / mask.numel()
     - Calculate remaining parameters
     - Store in stats['layers'][layer_name]
  6. Calculate memory reduction:
     - original_size = total * 4 (float32 bytes)
     - compressed_size = (total - pruned) * 4
     - reduction_ratio = 1 - (compressed_size / original_size)
  7. Return complete statistics

#### restore_weights(layer_names=None) -> None
- Parameters:
  - layer_names: List[str] - Optional list of layers to restore
- Logic:
  1. If layer_names is None, restore all layers
  2. For each layer to restore:
     - Get original weight from backup
     - Copy to model: layer.weight.data.copy_(original_weight)
     - Reset mask to ones
  3. Reset current_sparsity to 0.0
  4. Log restoration details

### 1.2 NulluProjector Class

#### Class: NulluProjector
**Purpose**: SVD-based low-rank decomposition with null space projection

**Attributes**:
- model: nn.Module - Target model
- rank_reduction_ratio: float - Target rank reduction (0.0 to 1.0)
- energy_threshold: float - Singular value energy threshold
- decomposed_layers: Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] - U, S, V matrices
- original_ranks: Dict[str, int] - Original matrix ranks
- selected_ranks: Dict[str, int] - Selected low ranks
- null_spaces: Dict[str, torch.Tensor] - Null space bases
- device: torch.device - Computation device

**Methods**:

#### __init__(model, rank_reduction_ratio=0.5, energy_threshold=0.95, device='cuda')
- Parameters:
  - model: nn.Module - Model to compress
  - rank_reduction_ratio: float - Target rank reduction
  - energy_threshold: float - Energy preservation threshold
  - device: str - Device for computation
- Logic:
  1. Store model and configuration
  2. Initialize decomposed_layers as empty dict
  3. Identify decomposable layers (Linear, Conv2d with kernel_size=1)
  4. For each decomposable layer:
     - Record original rank: min(weight.shape[0], weight.shape[1])
     - Store in original_ranks[layer_name]
  5. Initialize null_spaces dictionary
  6. Set numerical stability epsilon: 1e-10

#### compute_svd_decomposition(weight_tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
- Parameters:
  - weight_tensor: torch.Tensor - Weight matrix [out_features, in_features]
- Returns: Tuple[U, S, V] - SVD components
- Logic:
  1. Ensure 2D tensor:
     - If Conv2d: reshape to [out_channels, in_channels * k * k]
     - Store original shape for reconstruction
  2. Perform SVD: U, S, V = torch.linalg.svd(weight_tensor, full_matrices=False)
  3. Handle numerical stability:
     - Replace near-zero singular values: S[S < epsilon] = epsilon
  4. Sort singular values descending (should already be sorted)
  5. Return U, S, V tensors

#### select_optimal_rank(singular_values, energy_threshold) -> int
- Parameters:
  - singular_values: torch.Tensor - Singular values from SVD
  - energy_threshold: float - Energy to preserve (0.0 to 1.0)
- Returns: int - Selected rank
- Logic:
  1. Calculate total energy: total = torch.sum(singular_values ** 2)
  2. Calculate cumulative energy: cumsum = torch.cumsum(singular_values ** 2, dim=0)
  3. Find minimum rank preserving threshold energy:
     - normalized_cumsum = cumsum / total
     - rank = torch.searchsorted(normalized_cumsum, energy_threshold)
  4. Apply rank reduction constraint:
     - max_rank = int(len(singular_values) * (1 - rank_reduction_ratio))
     - rank = min(rank, max_rank)
  5. Ensure minimum rank of 1
  6. Return selected rank

#### decompose_layer(layer_name, weight_tensor) -> Dict
- Parameters:
  - layer_name: str - Name of layer
  - weight_tensor: torch.Tensor - Layer weights
- Returns: Dict - Decomposition results
- Logic:
  1. Perform SVD: U, S, V = compute_svd_decomposition(weight_tensor)
  2. Select rank: r = select_optimal_rank(S, energy_threshold)
  3. Truncate matrices:
     - U_r = U[:, :r]
     - S_r = S[:r]
     - V_r = V[:r, :]
  4. Store decomposition: decomposed_layers[layer_name] = (U_r, S_r, V_r)
  5. Calculate compression ratio:
     - original_params = weight_tensor.numel()
     - compressed_params = U_r.numel() + S_r.numel() + V_r.numel()
     - ratio = 1 - (compressed_params / original_params)
  6. Store selected rank: selected_ranks[layer_name] = r
  7. Return {'rank': r, 'compression_ratio': ratio, 'energy_preserved': energy}

#### project_to_null_space(layer_name) -> torch.Tensor
- Parameters:
  - layer_name: str - Name of layer
- Returns: torch.Tensor - Null space projection
- Logic:
  1. Get decomposition: U_r, S_r, V_r = decomposed_layers[layer_name]
  2. Get original shape from stored metadata
  3. Compute null space basis:
     - If U.shape[1] > r:
       - null_basis = U[:, r:] (columns beyond rank r)
     - Else: null_basis = empty tensor
  4. Orthonormalize null basis using QR decomposition:
     - Q, R = torch.linalg.qr(null_basis)
     - null_space = Q
  5. Store in null_spaces[layer_name]
  6. Return null_space tensor

#### reconstruct_weight(layer_name) -> torch.Tensor
- Parameters:
  - layer_name: str - Name of layer
- Returns: torch.Tensor - Reconstructed weight matrix
- Logic:
  1. Get decomposition: U_r, S_r, V_r = decomposed_layers[layer_name]
  2. Reconstruct weight: W_approx = U_r @ torch.diag(S_r) @ V_r
  3. If original was Conv2d:
     - Reshape to original shape stored in metadata
  4. Apply null space projection if exists:
     - if layer_name in null_spaces:
       - null_proj = null_spaces[layer_name]
       - W_approx = W_approx - null_proj @ null_proj.T @ W_approx
  5. Return reconstructed weight

#### analyze_subspace_overlap(layer1_name, layer2_name) -> float
- Parameters:
  - layer1_name: str - First layer name
  - layer2_name: str - Second layer name
- Returns: float - Grassmann distance between subspaces
- Logic:
  1. Get null spaces: N1 = null_spaces[layer1_name], N2 = null_spaces[layer2_name]
  2. Compute Grassmann distance:
     - M = N1.T @ N2 (inner product matrix)
     - singular_values = torch.linalg.svdvals(M)
     - angles = torch.acos(torch.clamp(singular_values, -1, 1))
     - distance = torch.norm(angles, p=2)
  3. Normalize distance to [0, 1] range
  4. Return distance value

### 1.3 AlphaEditor Class

#### Class: AlphaEditor
**Purpose**: Adaptive weight scaling with learnable importance parameters

**Attributes**:
- model: nn.Module - Target model
- alpha_parameters: Dict[str, nn.Parameter] - Learnable scaling factors
- task_vectors: Dict[str, torch.Tensor] - Extracted task-specific vectors
- importance_scores: Dict[str, torch.Tensor] - Channel/neuron importance
- learning_rate: float - Alpha optimization learning rate
- optimizer: torch.optim.Optimizer - Alpha parameter optimizer
- interpolation_weights: Dict[str, float] - Task interpolation weights
- device: torch.device - Computation device

**Methods**:

#### __init__(model, learning_rate=0.001, importance_metric='gradient', device='cuda')
- Parameters:
  - model: nn.Module - Model to adapt
  - learning_rate: float - Learning rate for alpha parameters
  - importance_metric: str - Method for importance calculation
  - device: str - Device for computation
- Logic:
  1. Store model and configuration
  2. Initialize alpha_parameters dictionary
  3. For each adaptable layer (Conv2d, Linear):
     - Create alpha parameter: torch.ones(out_features, requires_grad=True)
     - Wrap as nn.Parameter
     - Store in alpha_parameters[layer_name]
  4. Create optimizer for alpha parameters:
     - optimizer = Adam(alpha_parameters.values(), lr=learning_rate)
  5. Initialize task_vectors and importance_scores as empty dicts
  6. Set importance computation method

#### compute_importance_scores(layer_name, weight_tensor, activations=None) -> torch.Tensor
- Parameters:
  - layer_name: str - Name of layer
  - weight_tensor: torch.Tensor - Layer weights
  - activations: torch.Tensor - Optional activation maps
- Returns: torch.Tensor - Importance scores [out_features]
- Logic:
  1. If importance_metric == 'gradient':
     - If weight_tensor.grad exists:
       - scores = torch.norm(weight_tensor.grad, dim=tuple(range(1, weight_tensor.dim())))
     - Else: scores = torch.norm(weight_tensor, dim=tuple(range(1, weight_tensor.dim())))
  2. If importance_metric == 'magnitude':
     - scores = torch.norm(weight_tensor, p=2, dim=tuple(range(1, weight_tensor.dim())))
  3. If importance_metric == 'taylor':
     - If activations provided and gradients exist:
       - scores = torch.abs(activations * activations.grad).mean(dim=0)
     - Else: fallback to magnitude
  4. If importance_metric == 'fisher':
     - Compute Fisher information:
       - if weight_tensor.grad exists:
         - scores = (weight_tensor.grad ** 2).sum(dim=tuple(range(1, weight_tensor.dim())))
  5. Normalize scores: scores = scores / scores.max()
  6. Store in importance_scores[layer_name]
  7. Return scores

#### extract_task_vector(layer_name, reference_weight, fine_tuned_weight) -> torch.Tensor
- Parameters:
  - layer_name: str - Name of layer
  - reference_weight: torch.Tensor - Pre-trained weights
  - fine_tuned_weight: torch.Tensor - Task-specific weights
- Returns: torch.Tensor - Task vector
- Logic:
  1. Compute difference: task_vector = fine_tuned_weight - reference_weight
  2. Apply importance weighting if available:
     - if layer_name in importance_scores:
       - importance = importance_scores[layer_name].unsqueeze(-1)
       - task_vector = task_vector * importance
  3. Normalize by magnitude:
     - norm = torch.norm(task_vector)
     - if norm > 0: task_vector = task_vector / norm
  4. Store in task_vectors[layer_name]
  5. Return task_vector

#### apply_alpha_scaling(layer_name, weight_tensor) -> torch.Tensor
- Parameters:
  - layer_name: str - Name of layer
  - weight_tensor: torch.Tensor - Original weights
- Returns: torch.Tensor - Scaled weights
- Logic:
  1. Get alpha parameters: alpha = alpha_parameters[layer_name]
  2. Apply sigmoid for bounded scaling: alpha_scaled = torch.sigmoid(alpha) * 2
  3. Reshape alpha for broadcasting:
     - If Conv2d: alpha_scaled = alpha_scaled.view(-1, 1, 1, 1)
     - If Linear: alpha_scaled = alpha_scaled.view(-1, 1)
  4. Apply scaling: scaled_weight = weight_tensor * alpha_scaled
  5. If task vector exists:
     - task_vec = task_vectors[layer_name]
     - scaled_weight = scaled_weight + 0.1 * task_vec
  6. Return scaled_weight

#### optimize_alphas(dataloader, criterion, num_epochs=10) -> Dict[str, float]
- Parameters:
  - dataloader: DataLoader - Training data
  - criterion: nn.Module - Loss function
  - num_epochs: int - Training epochs
- Returns: Dict[str, float] - Training metrics
- Logic:
  1. Initialize metrics: {'loss': [], 'accuracy': []}
  2. For each epoch:
     - Set model to train mode
     - For each batch in dataloader:
       a. Zero gradients: optimizer.zero_grad()
       b. Apply alpha scaling to all layers:
          - For layer_name in alpha_parameters:
            - layer.weight = apply_alpha_scaling(layer_name, original_weight)
       c. Forward pass: outputs = model(inputs)
       d. Compute loss: loss = criterion(outputs, targets)
       e. Add L2 regularization on alphas:
          - reg_loss = 0.001 * sum(torch.norm(alpha, 2) for alpha in alpha_parameters.values())
          - total_loss = loss + reg_loss
       f. Backward pass: total_loss.backward()
       g. Clip gradients: torch.nn.utils.clip_grad_norm_(alpha_parameters.values(), 1.0)
       h. Update alphas: optimizer.step()
       i. Clamp alphas to reasonable range: alpha.data.clamp_(-3, 3)
     - Calculate epoch metrics
     - Store in metrics dictionary
  3. Return final metrics

#### interpolate_task_vectors(task_weights) -> None
- Parameters:
  - task_weights: Dict[str, float] - Interpolation weights for tasks
- Logic:
  1. Validate weights sum to 1.0
  2. For each layer with task vectors:
     - Initialize interpolated_vector = torch.zeros_like(first_task_vector)
     - For each task and weight:
       - interpolated_vector += weight * task_vectors[task][layer_name]
     - Store final interpolated vector
  3. Update model weights with interpolation

---

## 2. Cascade Pipeline

### 2.1 RCCPipeline Class

#### Class: RCCPipeline
**Purpose**: Orchestrates cascaded compression with checkpointing and validation

**Attributes**:
- stages: List[CompressionMethod] - Ordered list of compression stages
- model: nn.Module - Model being compressed
- checkpoints: Dict[str, Dict] - Saved states after each stage
- validation_metrics: Dict[str, Dict] - Performance metrics per stage
- performance_threshold: float - Minimum acceptable performance
- rollback_enabled: bool - Whether to allow rollback on failure
- current_stage: int - Current stage index
- stage_configs: Dict[str, Dict] - Configuration per stage
- logger: Logger - Logging interface

**Methods**:

#### __init__(model, stages, performance_threshold=0.95, rollback_enabled=True)
- Parameters:
  - model: nn.Module - Model to compress
  - stages: List[Dict] - Stage configurations
  - performance_threshold: float - Minimum performance retention
  - rollback_enabled: bool - Enable rollback mechanism
- Logic:
  1. Store model and configuration
  2. Initialize stages list:
     - For each stage_config in stages:
       - Create compression method instance based on type
       - Append to stages list
  3. Initialize checkpoints dictionary
  4. Set current_stage to -1 (not started)
  5. Create checkpoint directory if not exists
  6. Initialize validation metrics storage
  7. Set up logger with appropriate verbosity

#### save_checkpoint(stage_name, model_state, metrics) -> str
- Parameters:
  - stage_name: str - Name of completed stage
  - model_state: Dict - Model state dictionary
  - metrics: Dict - Performance metrics
- Returns: str - Checkpoint path
- Logic:
  1. Create checkpoint data structure:
     - checkpoint = {
         'stage_name': stage_name,
         'stage_index': current_stage,
         'model_state': model_state,
         'metrics': metrics,
         'timestamp': datetime.now(),
         'compression_stats': stage.get_compression_stats()
       }
  2. Generate checkpoint filename:
     - filename = f"checkpoint_{stage_name}_{timestamp}.pt"
  3. Save checkpoint: torch.save(checkpoint, path)
  4. Store in checkpoints[stage_name] = checkpoint
  5. Clean old checkpoints if exceeds limit (keep last 5)
  6. Return checkpoint path

#### validate_stage(stage_name, validator, dataloader) -> Dict[str, float]
- Parameters:
  - stage_name: str - Name of stage to validate
  - validator: Evaluator - Validation evaluator
  - dataloader: DataLoader - Validation data
- Returns: Dict[str, float] - Validation metrics
- Logic:
  1. Set model to eval mode: model.eval()
  2. Disable gradients: with torch.no_grad()
  3. Initialize metrics accumulator
  4. For each batch in dataloader:
     - Forward pass: outputs = model(inputs)
     - Compute metrics: batch_metrics = validator.compute_metrics(outputs, targets)
     - Accumulate metrics
  5. Calculate average metrics across batches
  6. Compare with baseline (first checkpoint):
     - If stage_index == 0: baseline = current metrics
     - Else: performance_ratio = current / baseline
  7. Check threshold: passed = performance_ratio >= performance_threshold
  8. Store in validation_metrics[stage_name]
  9. Return metrics with 'passed' flag

#### rollback_to_checkpoint(stage_name) -> bool
- Parameters:
  - stage_name: str - Stage to rollback to
- Returns: bool - Success status
- Logic:
  1. Check if rollback enabled and checkpoint exists
  2. Get checkpoint: checkpoint = checkpoints[stage_name]
  3. Load model state: model.load_state_dict(checkpoint['model_state'])
  4. Update current_stage to checkpoint['stage_index']
  5. Remove all checkpoints after this stage:
     - For stage in list(checkpoints.keys()):
       - If stage_index > target_index: del checkpoints[stage]
  6. Log rollback action
  7. Return True if successful, False otherwise

#### analyze_null_space_overlap() -> Dict[str, float]
- Parameters: None
- Returns: Dict[str, float] - Overlap analysis results
- Logic:
  1. Initialize overlap matrix
  2. For each pair of consecutive stages:
     - If both have null spaces:
       - Get null space bases from stages
       - Compute Grassmann distance
       - Store in overlap matrix
  3. Calculate statistics:
     - mean_overlap = matrix.mean()
     - max_overlap = matrix.max()
     - critical_overlaps = pairs with overlap > 0.7
  4. Generate overlap report
  5. Return analysis results

#### run_pipeline(dataloader, validator) -> nn.Module
- Parameters:
  - dataloader: DataLoader - Training/validation data
  - validator: Evaluator - Performance validator
- Returns: nn.Module - Compressed model
- Logic:
  1. Save initial checkpoint: save_checkpoint('initial', model.state_dict(), baseline_metrics)
  2. For each stage in stages:
     - Log stage start
     - Set current_stage = index
     - Try compression:
       a. Apply compression: compressed_model = stage.compress(model, stage_config)
       b. Validate performance: metrics = validate_stage(stage_name, validator, dataloader)
       c. If validation passed:
          - Save checkpoint
          - Update model reference
          - Log success
       d. If validation failed and rollback enabled:
          - Log failure
          - Rollback to previous checkpoint
          - Try alternative parameters or skip stage
       e. If validation failed and rollback disabled:
          - Log warning
          - Continue with degraded model
     - Catch exceptions:
       - Log error
       - Rollback if possible
       - Raise or continue based on configuration
  3. Analyze final compression:
     - Total compression ratio
     - Performance retention
     - Null space overlap analysis
  4. Save final model
  5. Return compressed model

#### get_pipeline_summary() -> Dict
- Parameters: None
- Returns: Dict - Complete pipeline summary
- Logic:
  1. Collect stage information:
     - For each completed stage:
       - Name, compression ratio, performance metrics
       - Execution time
       - Memory usage
  2. Calculate overall statistics:
     - Total compression: product of stage compressions
     - Final performance vs baseline
     - Total execution time
  3. Generate optimization suggestions:
     - Identify bottleneck stages
     - Suggest parameter adjustments
  4. Return comprehensive summary

---

## 3. Model Wrappers

### 3.1 CLIPWrapper Class

#### Class: CLIPWrapper
**Purpose**: Unified interface for CLIP models with compression support

**Attributes**:
- clip_model: CLIPModel - Underlying CLIP model
- vision_encoder: CLIPVisionTransformer - Vision transformer
- text_encoder: CLIPTextTransformer - Text transformer
- processor: CLIPProcessor - Input processor
- tokenizer: CLIPTokenizer - Text tokenizer
- device: torch.device - Computation device
- dtype: torch.dtype - Model precision (float32/float16)
- temperature: nn.Parameter - Learnable temperature parameter

**Methods**:

#### __init__(model_name='openai/clip-vit-base-patch32', device='cuda', dtype=torch.float32)
- Parameters:
  - model_name: str - HuggingFace model identifier
  - device: str - Device for computation
  - dtype: torch.dtype - Model precision
- Logic:
  1. Load CLIP model: clip_model = CLIPModel.from_pretrained(model_name)
  2. Move to device and dtype: clip_model.to(device, dtype)
  3. Extract components:
     - vision_encoder = clip_model.vision_model
     - text_encoder = clip_model.text_model
  4. Load processor and tokenizer:
     - processor = CLIPProcessor.from_pretrained(model_name)
     - tokenizer = CLIPTokenizer.from_pretrained(model_name)
  5. Initialize temperature: nn.Parameter(torch.tensor(0.07))
  6. Set model to eval mode by default

#### encode_image(images) -> torch.Tensor
- Parameters:
  - images: Union[torch.Tensor, List[PIL.Image]] - Input images
- Returns: torch.Tensor - Image embeddings [batch_size, embedding_dim]
- Logic:
  1. Process inputs:
     - If PIL images: processed = processor(images=images, return_tensors='pt')
     - If tensor: validate shape [B, 3, H, W]
  2. Move to device: pixel_values = processed['pixel_values'].to(device, dtype)
  3. Forward through vision encoder:
     - with torch.no_grad() if not training:
       - vision_outputs = vision_encoder(pixel_values)
  4. Extract embeddings:
     - image_embeds = vision_outputs.last_hidden_state
     - pooled_embeds = vision_outputs.pooler_output
  5. Normalize: image_embeds = F.normalize(pooled_embeds, p=2, dim=-1)
  6. Return normalized embeddings

#### encode_text(texts) -> torch.Tensor
- Parameters:
  - texts: Union[List[str], torch.Tensor] - Input texts
- Returns: torch.Tensor - Text embeddings [batch_size, embedding_dim]
- Logic:
  1. Tokenize inputs:
     - If strings: tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
     - If tensor: validate shape and use directly
  2. Move to device: input_ids = tokens['input_ids'].to(device)
  3. Forward through text encoder:
     - text_outputs = text_encoder(input_ids, attention_mask=tokens['attention_mask'])
  4. Extract embeddings:
     - text_embeds = text_outputs.last_hidden_state
     - pooled_embeds = text_outputs.pooler_output
  5. Normalize: text_embeds = F.normalize(pooled_embeds, p=2, dim=-1)
  6. Return normalized embeddings

#### forward(images=None, texts=None, return_loss=False) -> Dict[str, torch.Tensor]
- Parameters:
  - images: Optional[torch.Tensor] - Image inputs
  - texts: Optional[List[str]] - Text inputs
  - return_loss: bool - Whether to compute contrastive loss
- Returns: Dict - Model outputs
- Logic:
  1. Initialize output dict
  2. If images provided:
     - image_embeds = encode_image(images)
     - output['image_embeds'] = image_embeds
  3. If texts provided:
     - text_embeds = encode_text(texts)
     - output['text_embeds'] = text_embeds
  4. If both provided and return_loss:
     - Compute similarity matrix:
       - logits_per_image = (image_embeds @ text_embeds.T) / temperature
       - logits_per_text = logits_per_image.T
     - Create targets: targets = torch.arange(len(images)).to(device)
     - Compute losses:
       - loss_i = F.cross_entropy(logits_per_image, targets)
       - loss_t = F.cross_entropy(logits_per_text, targets)
       - total_loss = (loss_i + loss_t) / 2
     - output['loss'] = total_loss
  5. Return output dictionary

#### compute_similarity(image_embeds, text_embeds) -> torch.Tensor
- Parameters:
  - image_embeds: torch.Tensor - Image embeddings [N, D]
  - text_embeds: torch.Tensor - Text embeddings [M, D]
- Returns: torch.Tensor - Similarity matrix [N, M]
- Logic:
  1. Normalize if not already normalized:
     - image_embeds = F.normalize(image_embeds, p=2, dim=-1)
     - text_embeds = F.normalize(text_embeds, p=2, dim=-1)
  2. Compute cosine similarity:
     - similarity = image_embeds @ text_embeds.T
  3. Apply temperature scaling:
     - similarity = similarity / temperature.exp()
  4. Return similarity matrix

### 3.2 BLIPWrapper Class

#### Class: BLIPWrapper
**Purpose**: Unified interface for BLIP models with captioning support

**Attributes**:
- blip_model: BlipForConditionalGeneration - Underlying BLIP model
- vision_encoder: BlipVisionModel - Vision encoder
- text_decoder: BlipTextDecoder - Text decoder
- processor: BlipProcessor - Input processor
- tokenizer: BlipTokenizer - Text tokenizer
- device: torch.device - Computation device
- generation_config: GenerationConfig - Text generation settings

**Methods**:

#### __init__(model_name='Salesforce/blip-image-captioning-base', device='cuda')
- Parameters:
  - model_name: str - HuggingFace model identifier
  - device: str - Device for computation
- Logic:
  1. Load BLIP model: blip_model = BlipForConditionalGeneration.from_pretrained(model_name)
  2. Move to device: blip_model.to(device)
  3. Extract components:
     - vision_encoder = blip_model.vision_model
     - text_decoder = blip_model.text_decoder
  4. Load processor and tokenizer:
     - processor = BlipProcessor.from_pretrained(model_name)
     - tokenizer = BlipTokenizer.from_pretrained(model_name)
  5. Set generation config:
     - max_length = 50
     - num_beams = 5
     - temperature = 1.0
  6. Set to eval mode

#### encode_image(images) -> torch.Tensor
- Parameters:
  - images: Union[torch.Tensor, List[PIL.Image]] - Input images
- Returns: torch.Tensor - Image features [batch_size, seq_len, hidden_dim]
- Logic:
  1. Process inputs:
     - If PIL: processed = processor(images=images, return_tensors='pt')
     - If tensor: validate shape
  2. Move to device: pixel_values = processed['pixel_values'].to(device)
  3. Extract features:
     - vision_outputs = vision_encoder(pixel_values)
     - image_embeds = vision_outputs.last_hidden_state
  4. Apply projection if exists:
     - if hasattr(blip_model, 'visual_projection'):
       - image_embeds = blip_model.visual_projection(image_embeds)
  5. Return image embeddings

#### generate_caption(images, max_length=50, num_beams=5) -> List[str]
- Parameters:
  - images: Union[torch.Tensor, List[PIL.Image]] - Input images
  - max_length: int - Maximum caption length
  - num_beams: int - Beam search width
- Returns: List[str] - Generated captions
- Logic:
  1. Encode images: image_embeds = encode_image(images)
  2. Prepare decoder inputs:
     - decoder_input_ids = tokenizer.bos_token_id * torch.ones(batch_size, 1)
  3. Generate with beam search:
     - outputs = blip_model.generate(
         pixel_values=pixel_values,
         max_length=max_length,
         num_beams=num_beams,
         early_stopping=True,
         temperature=0.8
       )
  4. Decode outputs:
     - captions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
  5. Post-process captions:
     - Remove padding, clean punctuation
  6. Return caption list

#### forward(images, captions=None, return_loss=False) -> Dict[str, torch.Tensor]
- Parameters:
  - images: torch.Tensor - Input images
  - captions: Optional[List[str]] - Target captions for training
  - return_loss: bool - Whether to compute loss
- Returns: Dict - Model outputs
- Logic:
  1. Encode images: image_embeds = encode_image(images)
  2. If training (captions provided):
     - Tokenize captions: labels = tokenizer(captions, padding=True, return_tensors='pt')
     - Forward with teacher forcing:
       - outputs = blip_model(
           pixel_values=images,
           labels=labels['input_ids'],
           attention_mask=labels['attention_mask']
         )
     - Extract loss: loss = outputs.loss
  3. If inference (no captions):
     - Generate captions: captions = generate_caption(images)
  4. Return {'loss': loss, 'captions': captions, 'image_embeds': image_embeds}

---

## 4. Training System

### 4.1 CompressionTrainer Class

#### Class: CompressionTrainer
**Purpose**: Orchestrates training with compression and distillation

**Attributes**:
- model: nn.Module - Student model being trained
- teacher_model: nn.Module - Teacher model for distillation
- optimizer: torch.optim.Optimizer - Parameter optimizer
- scheduler: torch.optim.lr_scheduler - Learning rate scheduler
- criterion: nn.Module - Primary loss function
- kd_loss: KnowledgeDistillationLoss - Distillation loss
- device: torch.device - Training device
- mixed_precision: bool - Use automatic mixed precision
- scaler: torch.cuda.amp.GradScaler - AMP gradient scaler
- current_epoch: int - Current training epoch
- global_step: int - Global training step

**Methods**:

#### __init__(model, teacher_model, config)
- Parameters:
  - model: nn.Module - Student model
  - teacher_model: nn.Module - Teacher model
  - config: Dict - Training configuration
- Logic:
  1. Store models and config
  2. Initialize optimizer:
     - optimizer = AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
  3. Setup scheduler:
     - scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'])
  4. Initialize losses:
     - criterion = nn.CrossEntropyLoss()
     - kd_loss = KnowledgeDistillationLoss(temperature=config['kd_temp'])
  5. Setup mixed precision:
     - If config['mixed_precision']:
       - scaler = GradScaler()
  6. Initialize tracking variables
  7. Move models to device

#### train_epoch(dataloader) -> Dict[str, float]
- Parameters:
  - dataloader: DataLoader - Training data
- Returns: Dict[str, float] - Epoch metrics
- Logic:
  1. Set model to train mode: model.train()
  2. Set teacher to eval mode: teacher_model.eval()
  3. Initialize metric accumulators
  4. For each batch in dataloader:
     a. Move data to device
     b. Zero gradients: optimizer.zero_grad()
     c. Mixed precision context:
        - with autocast(enabled=mixed_precision):
          - Student forward: student_outputs = model(inputs)
          - Teacher forward: with torch.no_grad(): teacher_outputs = teacher_model(inputs)
          - Compute task loss: task_loss = criterion(student_outputs, targets)
          - Compute KD loss: distill_loss = kd_loss(student_outputs, teacher_outputs)
          - Total loss: loss = 0.7 * task_loss + 0.3 * distill_loss
     d. Backward pass:
        - If mixed_precision: scaler.scale(loss).backward()
        - Else: loss.backward()
     e. Gradient clipping: clip_grad_norm_(model.parameters(), max_norm=1.0)
     f. Optimizer step:
        - If mixed_precision: scaler.step(optimizer); scaler.update()
        - Else: optimizer.step()
     g. Update metrics
     h. Increment global_step
  5. Calculate epoch averages
  6. Return metrics

#### validate(dataloader) -> Dict[str, float]
- Parameters:
  - dataloader: DataLoader - Validation data
- Returns: Dict[str, float] - Validation metrics
- Logic:
  1. Set model to eval mode
  2. Disable gradients: with torch.no_grad()
  3. Initialize metric accumulators
  4. For each batch in dataloader:
     - Forward pass: outputs = model(inputs)
     - Compute metrics
     - Accumulate results
  5. Calculate averages
  6. Return validation metrics

#### train(train_loader, val_loader, num_epochs) -> Dict[str, List[float]]
- Parameters:
  - train_loader: DataLoader - Training data
  - val_loader: DataLoader - Validation data
  - num_epochs: int - Number of training epochs
- Returns: Dict[str, List[float]] - Training history
- Logic:
  1. Initialize history dict
  2. For each epoch in range(num_epochs):
     - Train one epoch: train_metrics = train_epoch(train_loader)
     - Validate: val_metrics = validate(val_loader)
     - Step scheduler: scheduler.step()
     - Log metrics
     - Save checkpoint if best validation
     - Early stopping check
     - Update history
  3. Return complete history

### 4.2 Knowledge Distillation Logic

#### compute_distillation_loss(student_logits, teacher_logits, temperature) -> torch.Tensor
- Parameters:
  - student_logits: torch.Tensor - Student model outputs [B, C]
  - teacher_logits: torch.Tensor - Teacher model outputs [B, C]
  - temperature: float - Temperature for softening
- Returns: torch.Tensor - KL divergence loss
- Logic:
  1. Apply temperature scaling:
     - student_soft = F.log_softmax(student_logits / temperature, dim=-1)
     - teacher_soft = F.softmax(teacher_logits / temperature, dim=-1)
  2. Compute KL divergence:
     - kl_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
  3. Scale by temperature squared: loss = kl_loss * (temperature ** 2)
  4. Return loss

#### setup_teacher_student(teacher_model, student_model) -> Tuple[nn.Module, nn.Module]
- Parameters:
  - teacher_model: nn.Module - Pre-trained teacher
  - student_model: nn.Module - Student to train
- Returns: Tuple[nn.Module, nn.Module] - Prepared models
- Logic:
  1. Freeze teacher model:
     - teacher_model.eval()
     - for param in teacher_model.parameters(): param.requires_grad = False
  2. Initialize student from teacher if needed:
     - Copy matching layers
     - Initialize new layers with Xavier/He initialization
  3. Set up gradient checkpointing for memory efficiency
  4. Return prepared models

### 4.3 Mixed Precision Training

#### mixed_precision_forward(model, inputs, dtype=torch.float16) -> torch.Tensor
- Parameters:
  - model: nn.Module - Model to run
  - inputs: Dict - Input batch
  - dtype: torch.dtype - Precision type
- Returns: torch.Tensor - Model outputs
- Logic:
  1. Create autocast context: with autocast(dtype=dtype):
  2. Forward pass: outputs = model(**inputs)
  3. Check for inf/nan:
     - If detected, retry with float32
  4. Return outputs

---

## 5. Evaluation Framework

### 5.1 ZeroShotEvaluator Class

#### Class: ZeroShotEvaluator
**Purpose**: Zero-shot classification evaluation on ImageNet and other datasets

**Attributes**:
- model: CLIPWrapper - Model to evaluate
- class_templates: List[str] - Text templates for classes
- class_names: List[str] - Dataset class names
- text_embeddings: torch.Tensor - Pre-computed class embeddings
- device: torch.device - Computation device
- batch_size: int - Evaluation batch size

**Methods**:

#### __init__(model, dataset_name='imagenet', batch_size=256, device='cuda')
- Parameters:
  - model: CLIPWrapper - CLIP model wrapper
  - dataset_name: str - Target dataset
  - batch_size: int - Batch size for evaluation
  - device: str - Device for computation
- Logic:
  1. Store model and configuration
  2. Load class names for dataset:
     - If imagenet: load 1000 ImageNet classes
     - If cifar10: load 10 CIFAR classes
  3. Define text templates:
     - templates = ["a photo of a {}", "an image of {}", ...]
  4. Pre-compute text embeddings:
     - text_embeddings = compute_text_embeddings(class_names, templates)
  5. Cache embeddings for efficiency

#### compute_text_embeddings(class_names, templates) -> torch.Tensor
- Parameters:
  - class_names: List[str] - Class names
  - templates: List[str] - Text templates
- Returns: torch.Tensor - Text embeddings [num_classes, embedding_dim]
- Logic:
  1. Initialize embedding list
  2. For each class_name:
     - Generate prompts: [template.format(class_name) for template in templates]
     - Encode prompts: embeddings = model.encode_text(prompts)
     - Average embeddings: class_embedding = embeddings.mean(dim=0)
     - Normalize: class_embedding = F.normalize(class_embedding, p=2)
  3. Stack all class embeddings
  4. Return tensor [num_classes, embedding_dim]

#### evaluate(dataloader) -> Dict[str, float]
- Parameters:
  - dataloader: DataLoader - Evaluation dataset
- Returns: Dict[str, float] - Evaluation metrics
- Logic:
  1. Set model to eval mode
  2. Initialize correct predictions counter
  3. For each batch in dataloader:
     - Encode images: image_embeds = model.encode_image(images)
     - Compute similarities: logits = image_embeds @ text_embeddings.T
     - Get predictions: preds = logits.argmax(dim=-1)
     - Count correct: correct += (preds == labels).sum()
  4. Calculate metrics:
     - accuracy = correct / total
     - top5_accuracy = calculate_topk_accuracy(logits, labels, k=5)
  5. Return {'accuracy': accuracy, 'top5_accuracy': top5_accuracy}

### 5.2 RetrievalEvaluator Class

#### Class: RetrievalEvaluator
**Purpose**: Image-text retrieval evaluation

**Attributes**:
- model: CLIPWrapper - Model to evaluate
- image_embeddings: torch.Tensor - Cached image embeddings
- text_embeddings: torch.Tensor - Cached text embeddings
- device: torch.device - Computation device

**Methods**:

#### compute_retrieval_metrics(image_embeds, text_embeds, img2txt_gt, txt2img_gt) -> Dict
- Parameters:
  - image_embeds: torch.Tensor - Image embeddings [N, D]
  - text_embeds: torch.Tensor - Text embeddings [M, D]
  - img2txt_gt: List[List[int]] - Ground truth text indices for each image
  - txt2img_gt: List[List[int]] - Ground truth image indices for each text
- Returns: Dict - Retrieval metrics
- Logic:
  1. Compute similarity matrix: sims = image_embeds @ text_embeds.T
  2. Image-to-text retrieval:
     - For each image:
       - Get similarity scores: scores = sims[i]
       - Rank texts: ranks = scores.argsort(descending=True)
       - Find rank of ground truth texts
       - Calculate recall@k for k in [1, 5, 10]
  3. Text-to-image retrieval:
     - Similar process with sims.T
  4. Calculate mean metrics
  5. Return comprehensive metrics dict

### 5.3 MetricComputer Class

#### Class: MetricComputer
**Purpose**: Computes various evaluation metrics

**Methods**:

#### compute_bleu_score(predictions, references, max_n=4) -> float
- Parameters:
  - predictions: List[str] - Generated captions
  - references: List[List[str]] - Reference captions
  - max_n: int - Maximum n-gram order
- Returns: float - BLEU score
- Logic:
  1. Tokenize predictions and references
  2. For each n in range(1, max_n+1):
     - Calculate n-gram precision
     - Apply brevity penalty
  3. Compute geometric mean of precisions
  4. Return BLEU score

#### compute_cider_score(predictions, references) -> float
- Parameters:
  - predictions: List[str] - Generated captions
  - references: List[List[str]] - Reference captions
- Returns: float - CIDEr score
- Logic:
  1. Compute TF-IDF weights for n-grams
  2. Calculate cosine similarity between prediction and reference vectors
  3. Average across all samples
  4. Return CIDEr score

---

## 6. Optimization

### 6.1 BayesianOptimizer Class

#### Class: BayesianOptimizer
**Purpose**: Bayesian optimization for hyperparameter tuning

**Attributes**:
- objective_function: Callable - Function to optimize
- search_space: Dict - Parameter search ranges
- optuna_study: optuna.Study - Optimization study
- best_params: Dict - Best parameters found
- best_value: float - Best objective value
- trial_results: List[Dict] - All trial results

**Methods**:

#### __init__(objective_function, search_space, direction='maximize')
- Parameters:
  - objective_function: Callable - Function to optimize
  - search_space: Dict - Parameter ranges
  - direction: str - Optimization direction
- Logic:
  1. Store objective function and search space
  2. Create Optuna study:
     - study = optuna.create_study(
         direction=direction,
         sampler=TPESampler(),
         pruner=MedianPruner()
       )
  3. Initialize results storage
  4. Set up logging

#### suggest_parameters(trial) -> Dict
- Parameters:
  - trial: optuna.Trial - Current trial
- Returns: Dict - Suggested parameters
- Logic:
  1. Initialize params dict
  2. For each param in search_space:
     - If continuous: trial.suggest_float(name, low, high, log=log_scale)
     - If integer: trial.suggest_int(name, low, high)
     - If categorical: trial.suggest_categorical(name, choices)
  3. Add dependencies between parameters
  4. Return suggested parameters

#### optimize(n_trials=100, timeout=3600) -> Dict
- Parameters:
  - n_trials: int - Number of trials
  - timeout: int - Maximum time in seconds
- Returns: Dict - Best parameters and results
- Logic:
  1. Define objective wrapper:
     - def objective(trial):
       - params = suggest_parameters(trial)
       - value = objective_function(params)
       - Store trial result
       - Return value
  2. Run optimization:
     - study.optimize(objective, n_trials=n_trials, timeout=timeout)
  3. Extract best:
     - best_params = study.best_params
     - best_value = study.best_value
  4. Generate optimization report
  5. Return results

### 6.2 Adaptive Scheduling

#### adaptive_compression_schedule(current_performance, target_performance, current_ratio) -> float
- Parameters:
  - current_performance: float - Current model performance
  - target_performance: float - Target performance threshold
  - current_ratio: float - Current compression ratio
- Returns: float - Adjusted compression ratio
- Logic:
  1. Calculate performance gap: gap = current_performance - target_performance
  2. If gap > 0.05:  # Can compress more
     - new_ratio = current_ratio * 1.1
  3. If gap < -0.02:  # Need to reduce compression
     - new_ratio = current_ratio * 0.9
  4. Else:  # Within acceptable range
     - new_ratio = current_ratio
  5. Clamp to valid range [0.1, 0.99]
  6. Return new_ratio

---

## 7. Implementation Notes

### 7.1 Numerical Stability
- Always add epsilon (1e-10) when dividing or taking logarithms
- Use log-sum-exp trick for stable softmax computation
- Clamp values before arccos/arcsin operations
- Use double precision for SVD when needed

### 7.2 Memory Management
- Use gradient checkpointing for large models
- Clear intermediate tensors explicitly when not needed
- Use in-place operations where possible
- Implement batch processing for large datasets

### 7.3 Error Handling
- Validate tensor shapes at each transformation
- Check for NaN/Inf values after critical operations
- Implement graceful degradation for compression failures
- Provide detailed error messages with tensor shapes

### 7.4 Performance Optimization
- Pre-compute embeddings when possible
- Use CUDA kernels for custom operations
- Implement parallel processing for independent operations
- Cache frequently accessed computations

### 7.5 Gradient Flow
- Ensure all compression operations maintain gradient flow
- Use straight-through estimators for non-differentiable operations
- Implement custom backward passes when needed
- Monitor gradient magnitudes during training

---

## Summary

This detailed logic design provides implementation-ready specifications for all critical components of the RCC system. Each method includes:

1. **Precise parameter specifications** with types and tensor shapes
2. **Step-by-step logic flow** that can be directly translated to code
3. **Return value specifications** with exact formats
4. **Error handling requirements** for robust implementation
5. **State management details** for maintaining consistency

The design ensures:
- **Numerical stability** through careful handling of edge cases
- **Memory efficiency** through strategic tensor management
- **Gradient flow preservation** for end-to-end training
- **Modular architecture** for easy testing and maintenance

This specification serves as a complete blueprint for implementing the RCC compression system, with all ambiguity removed and implementation details fully specified.