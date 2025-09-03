import torch
from flash_attn import flash_attn_func
import torch.nn.functional as F

def verify_flash_attention_correctness():
    """
    This function verifies that the output of FlashAttention is numerically
    close to a standard PyTorch implementation of scaled dot-product attention.
    """
    print("--- FlashAttention Correctness Verification Script ---")

    # 1. Check for CUDA availability
    if not torch.cuda.is_available():
        print("❌ Error: PyTorch was not built with CUDA support. FlashAttention requires a GPU.")
        return

    device = "cuda"
    print(f"✅ CUDA is available. Using device: {torch.cuda.get_device_name(0)}")

    # 2. Define tensor dimensions for the test
    batch_size = 4
    seq_len = 1024
    num_heads = 12
    head_dim = 64
    dtype = torch.float16 # Use float16 for a realistic comparison

    print(f"\nCreating test tensors with shape (batch, seq_len, heads, head_dim):")
    print(f"({batch_size}, {seq_len}, {num_heads}, {head_dim})")

    # 3. Create identical random input tensors for both models
    # Use a fixed seed to ensure the inputs are the same for both runs
    torch.manual_seed(0)
    q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

    # --- 4. Run the FlashAttention implementation ---
    try:
        print("\nRunning flash_attn_func (causal=True)...")
        output_flash = flash_attn_func(q, k, v, causal=True)
        print("✅ flash_attn_func executed successfully!")
    except Exception as e:
        print(f"❌ Error running FlashAttention: {e}")
        return

    # --- 5. Run the standard PyTorch ("vanilla") implementation ---
    print("\nRunning standard PyTorch attention (causal=True)...")
    
    # Reshape for PyTorch's multi-head attention format: (batch, heads, seq_len, dim)
    q_vanilla = q.transpose(1, 2)
    k_vanilla = k.transpose(1, 2)
    v_vanilla = v.transpose(1, 2)

    # Manually implement scaled dot-product attention
    # This is what happens inside a standard transformer
    attn_scores = torch.matmul(q_vanilla, k_vanilla.transpose(-2, -1)) / (head_dim**0.5)

    # Create and apply the causal mask
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
    attn_scores.masked_fill_(mask, float('-inf'))

    # Apply softmax and multiply by V
    attn_weights = F.softmax(attn_scores, dim=-1)
    output_vanilla_transposed = torch.matmul(attn_weights, v_vanilla)

    # Transpose back to the original shape: (batch, seq_len, heads, dim)
    output_vanilla = output_vanilla_transposed.transpose(1, 2)
    print("✅ Standard attention executed successfully!")


    # --- 6. Compare the outputs ---
    print("\n--- Comparison Results ---")
    print(f"Shape of FlashAttention output: {output_flash.shape}")
    print(f"Shape of Vanilla Attention output: {output_vanilla.shape}")

    # Use torch.allclose to check for numerical similarity.
    # A small tolerance (atol) is needed due to minor differences in floating-point arithmetic.
    are_close = torch.allclose(output_flash, output_vanilla, atol=1e-3, rtol=1e-4)

    if are_close:
        print("\n🎉 SUCCESS: The outputs of FlashAttention and vanilla attention are nearly identical.")
    else:
        print("\n❌ FAILURE: The outputs are significantly different.")

    # Print the maximum absolute difference for more detail
    max_diff = torch.max(torch.abs(output_flash - output_vanilla)).item()
    print(f"   - Maximum absolute difference between outputs: {max_diff:.6f}")


if __name__ == "__main__":
    verify_flash_attention_correctness()
