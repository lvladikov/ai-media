
// Model parameter counts (in billions) for dynamic RAM estimation
export const MODEL_PARAMS: Record<string, number> = {
  'deepseek-r1-qwen-7b': 7,
  'deepseek-r1-qwen-14b': 14,
  'deepseek-r1-qwen-32b': 32,
  'deepseek-r1-llama-8b': 8,
  'deepseek-r1-llama-70b': 70,
  'llama-3.1-8b': 8,
  'mistral-nemo-12b': 12,
  'qwen3-8b': 8,
  'qwen3-14b': 14,
  'qwen3-coder-30b': 3.3, // MoE
  'qwen-coder-32b': 32,
  'qwen-coder-14b': 14,
  'qwen-coder-7b': 7,
  'qwen-vl': 8,
  'qwen3-vl-4b': 4,
  'qwen3-vl-2b': 2,
  'qwen3-opus-4.5-8b': 8,
  'qwen3-opus-4.5-14b': 14,
  'qwen3-gpt-5.2-8b': 8,
  'qwen3-gpt-5.2-14b': 14,

  // Image Models (Approximate for VRAM estimation)
  'sd3.5-turbo': 8,      // ~19GB total check with overhead
  'sd3.5-medium': 4,     // ~10GB
  'sd3.5-large': 8,      // ~19GB
  'sdxl': 7,             // ~8-9GB
  'sd-1.5': 1.5,         // ~4GB
  'z-image': 12,         // ~31GB (Heavy stack)
  'qwen-image': 7,       // ~20-40GB depending on vit
  'qwen-image-2512': 14, // Heavier Vit
  'flux': 12,            // Schnell ~12GB
  'flux-dev': 12,        // Dev ~16GB (more overhead)
  'flux2': 12,           // 4-bit quantized default
  'flux2-full': 30,      // Massive >65GB
  'instruct-pix2pix': 3, // SD-based
  'qwen-image-edit': 7,  // Qwen-VL based

  // Video Models
  'wan-2.2-5b': 5,
  'wan-2.2': 14,
  'cogvideox': 5,
  'hunyuan': 13,
  'ltx-video': 2,
  'mochi-1': 10,
  'svd': 2,
  'zeroscope': 1.7,
  'zeroscope-xl': 1.7,
  'ms-1.7b': 1.7,
};

export const BYTES_PER_WEIGHT: Record<string, number> = {
  'float32': 4.0,
  'float16': 2.0,
  'bfloat16': 2.0,
  'int8': 1.0,
  'int6': 0.75,
  'int4': 0.5,
  'auto': 2.0, // Default estimate
};

/**
 * Calculates dynamic RAM usage estimate for a model.
 * Mimics backend logic: Params * BytesPerWeight * 1.2 overhead.
 */
export const getDynamicRam = (modelId: string, precision: string, framework: string = 'auto') => {
  const params = MODEL_PARAMS[modelId] || 8;

  // Identify model type based on known keys
  // Simple heuristic: check if it's in a known video/image set logic or keyword match
  // Since MODEL_PARAMS is flat, we can infer by checking keywords usually found in video models
  const isVideoOrImage = [
    'wan', 'cogvideo', 'hunyuan', 'ltx', 'mochi', 'svd', 'zeroscope', 'sd', 'flux', 'z-image', 'pix2pix'
  ].some(k => modelId.includes(k));

  // Resolve "auto" precision based on framework/platform heuristics
  let effectivePrecision = precision;

  if (precision === 'auto') {
    if (framework === 'mlx') {
      // MLX: Text is int4, but some Image/Video might be bf16. 
      // However, our standard MLX Video pipeline is int4 quantized.
      effectivePrecision = 'int4';
    } else {
      // PyTorch/Standard (CUDA/MPS):
      // - Text: Often int4 (bitsandbytes)
      // - Video/Image: Usually float16/bfloat16 (Diffusers default)
      if (isVideoOrImage) {
        effectivePrecision = 'float16';
      } else {
        effectivePrecision = 'int4';
      }
    }
  }

  const bytes = BYTES_PER_WEIGHT[effectivePrecision] || BYTES_PER_WEIGHT['int4']; // Default to 4-bit for safe estimate
  const estimate = params * bytes * 1.2; // 20% overhead
  return `~${Math.round(estimate)}GB`;
};
