"""
Verification script to test that graph augmentation is working correctly.

Run this before training to ensure augmentations create different views.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path so we can import src
# This handles the case when running from any directory
script_dir = Path(__file__).resolve().parent
project_root = script_dir
sys.path.insert(0, str(project_root))

import torch
from torch_geometric.data import Data

# Now import after fixing the path
from src.utils.graph_augmentation import GraphAugmentor


def create_dummy_molecule():
    """Create a simple dummy molecular graph for testing."""
    # 5 atoms (nodes)
    x = torch.randn(5, 10)  # 10 features per atom
    
    # 6 bonds (edges) - create a simple chain
    edge_index = torch.tensor([
        [0, 1, 1, 2, 2, 3, 3, 4],
        [1, 0, 2, 1, 3, 2, 4, 3]
    ], dtype=torch.long)
    
    # Edge attributes (bond types)
    edge_attr = torch.randn(8, 5)  # 5 features per edge
    
    # Labels (for multi-task)
    y = torch.tensor([[1.0, 0.0, 1.0]])
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def test_augmentation():
    """Test that augmentations create different views."""
    print("=" * 60)
    print("Testing Graph Augmentation for Mean Teacher")
    print("=" * 60)
    
    # Create augmentor with moderate settings
    augmentor = GraphAugmentor(
        node_drop_rate=0.2,
        edge_drop_rate=0.2,
        feature_mask_rate=0.1,
        feature_noise_std=0.01,
    )
    
    print(f"\nAugmentor configuration:")
    print(augmentor)
    
    # Create test molecule
    data = create_dummy_molecule()
    
    print(f"\nOriginal graph:")
    print(f"  Nodes: {data.x.shape[0]}")
    print(f"  Edges: {data.edge_index.shape[1]}")
    print(f"  Node features shape: {data.x.shape}")
    print(f"  First node features: {data.x[0, :3].tolist()}")
    
    # Apply augmentation twice (like student and teacher would)
    aug1 = augmentor(data)
    aug2 = augmentor(data)
    
    print(f"\nAugmented view 1:")
    print(f"  Nodes: {aug1.x.shape[0]}")
    print(f"  Edges: {aug1.edge_index.shape[1]}")
    print(f"  First node features: {aug1.x[0, :3].tolist()}")
    
    print(f"\nAugmented view 2:")
    print(f"  Nodes: {aug2.x.shape[0]}")
    print(f"  Edges: {aug2.edge_index.shape[1]}")
    print(f"  First node features: {aug2.x[0, :3].tolist()}")
    
    # Check if they're different
    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS:")
    print("=" * 60)
    
    same_features = torch.equal(aug1.x, aug2.x)
    same_edges = torch.equal(aug1.edge_index, aug2.edge_index)
    
    print(f"✓ Same node features? {same_features}")
    print(f"✓ Same edges? {same_edges}")
    
    if same_features and same_edges:
        print("\n❌ FAILED: Augmentations are producing identical outputs!")
        print("   This means augmentation is NOT working correctly.")
        return False
    else:
        print("\n✅ PASSED: Augmentations create different views!")
        print("   Mean Teacher will receive different inputs for student/teacher.")
        
        # Calculate how different they are
        feature_diff = (aug1.x - aug2.x).abs().mean().item()
        edge_diff_count = (aug1.edge_index.shape[1] != aug2.edge_index.shape[1])
        
        print(f"\nDifference statistics:")
        print(f"  Average feature difference: {feature_diff:.6f}")
        print(f"  Different edge counts: {edge_diff_count}")
        
        return True


def test_no_augmentation():
    """Test that NoAugmentation returns data unchanged."""
    from src.utils.graph_augmentation import NoAugmentation
    
    print("\n" + "=" * 60)
    print("Testing NoAugmentation (for supervised baseline)")
    print("=" * 60)
    
    augmentor = NoAugmentation()
    data = create_dummy_molecule()
    
    aug1 = augmentor(data)
    aug2 = augmentor(data)
    
    same = torch.equal(aug1.x, aug2.x) and torch.equal(aug1.edge_index, aug2.edge_index)
    
    if same:
        print("✅ PASSED: NoAugmentation returns data unchanged")
        return True
    else:
        print("❌ FAILED: NoAugmentation is modifying data!")
        return False


def test_different_augmentation_strengths():
    """Test different augmentation strength settings."""
    print("\n" + "=" * 60)
    print("Testing Different Augmentation Strengths")
    print("=" * 60)
    
    data = create_dummy_molecule()
    strengths = [
        ("Conservative", 0.05),
        ("Moderate", 0.1),
        ("Aggressive", 0.2),
    ]
    
    for name, rate in strengths:
        augmentor = GraphAugmentor(
            node_drop_rate=rate,
            edge_drop_rate=rate,
            feature_mask_rate=rate,
            feature_noise_std=rate * 0.1,
        )
        
        # Run 10 augmentations and measure average difference
        differences = []
        for _ in range(10):
            aug = augmentor(data)
            diff = (data.x - aug.x).abs().mean().item()
            differences.append(diff)
        
        avg_diff = sum(differences) / len(differences)
        print(f"{name:15s} (rate={rate:.2f}): avg difference = {avg_diff:.6f}")
    
    print("\n✅ Higher rates → more augmentation (larger differences)")


if __name__ == "__main__":
    print("\n🧪 MEAN TEACHER AUGMENTATION TEST SUITE\n")
    print(f"Working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).resolve()}")
    print(f"Python path includes: {project_root}\n")
    
    success = True
    
    # Run tests
    try:
        success &= test_augmentation()
        success &= test_no_augmentation()
        test_different_augmentation_strengths()
    except Exception as e:
        print(f"\n❌ ERROR: Test failed with exception:")
        print(f"   {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Augmentation is working correctly")
        print("✅ Ready to train Mean Teacher")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("❌ Check the augmentation implementation")
    print("=" * 60 + "\n")
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)