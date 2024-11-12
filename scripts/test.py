import tensorflow as tf
import numpy as np
import os
import sys
import tensorflow as tf

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.resnet import ResNet18, ResNet50, load_encoder_weights
from src.libs.data_loading import create_dataset


def verify_weight_loading(model, weights_path):
    """
    Verify if weights are properly loaded into a model.

    Args:
        model: Keras model instance
        weights_path: Path to weights file
    """
    # Store initial weights
    initial_weights = [layer.get_weights() for layer in model.layers]

    # Load weights
    model.load_weights(weights_path, skip_mismatch=True, by_name=True)

    # Get weights after loading
    loaded_weights = [layer.get_weights() for layer in model.layers]

    # Compare weights and print detailed info
    print("\nWeight Loading Analysis:")
    print("-" * 50)

    for layer_idx, (layer, before, after) in enumerate(
        zip(model.layers, initial_weights, loaded_weights)
    ):
        weights_changed = not all(np.array_equal(b, a) for b, a in zip(before, after))

        print(f"\nLayer {layer_idx}: {layer.name}")
        print(f"Type: {layer.__class__.__name__}")
        print(f"Weights changed: {'✓' if weights_changed else '✗'}")

        if weights_changed:
            for weight_idx, (b, a) in enumerate(zip(before, after)):
                print(f"  Weight {weight_idx}:")
                print(f"    Shape: {a.shape}")
                print(f"    Mean before: {np.mean(b):.6f}")
                print(f"    Mean after: {np.mean(a):.6f}")
                print(f"    Std before: {np.std(b):.6f}")
                print(f"    Std after: {np.std(a):.6f}")

    return True


def get_unloaded_layers(model):
    """
    Identify layers that might not have loaded weights.

    Args:
        model: Keras model instance
    """
    suspicious_layers = []

    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) > 0:  # Only check layers with weights
            # Check if weights appear to be randomly initialized
            for w in weights:
                mean = np.mean(w)
                std = np.std(w)

                # Typical initialization statistics
                if -0.1 < mean < 0.1 and 0.01 < std < 1.0:
                    suspicious_layers.append(
                        {
                            "layer_name": layer.name,
                            "layer_type": layer.__class__.__name__,
                            "weight_stats": {"mean": mean, "std": std, "shape": w.shape},
                        }
                    )
                    break

    return suspicious_layers


import tensorflow as tf
import numpy as np
import h5py


def debug_weight_loading(model, weights_path):
    """
    Advanced debugging for weight loading issues.

    Args:
        model: Keras model instance
        weights_path: Path to weights file
    """
    print("\nModel Layer Structure:")
    print("-" * 50)
    model_layers = {}
    for layer in model.layers:
        # Get full layer name including any parent names
        full_name = get_full_layer_name(layer)
        weights = layer.get_weights()
        model_layers[full_name] = {
            "type": layer.__class__.__name__,
            "weight_shapes": [w.shape for w in weights] if weights else [],
            "trainable": layer.trainable,
        }
        print(f"Layer: {full_name}")
        print(f"  Type: {layer.__class__.__name__}")
        print(f"  Weight shapes: {[w.shape for w in weights] if weights else 'No weights'}")

    print("\nWeight File Structure:")
    print("-" * 50)
    try:
        with h5py.File(weights_path, "r") as f:
            weight_layers = get_h5_structure(f)
            print("\nAvailable weight layers:")
            for name, shapes in weight_layers.items():
                print(f"Weight layer: {name}")
                print(f"  Shapes: {shapes}")

        print("\nPotential Naming Mismatches:")
        print("-" * 50)
        for model_layer in model_layers:
            matches = find_potential_matches(model_layer, weight_layers.keys())
            if matches:
                print(f"\nModel layer: {model_layer}")
                print("Potential matches in weights file:")
                for match in matches:
                    print(f"  - {match}")
            else:
                print(f"\nNo matches found for: {model_layer}")

    except Exception as e:
        print(f"Error reading weights file: {str(e)}")


def get_full_layer_name(layer):
    """Get the full hierarchical name of a layer."""
    names = []
    current = layer
    while current is not None:
        if hasattr(current, "name"):
            names.append(current.name)
        current = getattr(current, "_parent_layer", None)
    return "/".join(reversed(names))


def get_h5_structure(f):
    """Recursively get structure of H5 file."""
    result = {}

    def visit_item(name, item):
        if isinstance(item, h5py.Dataset):
            result[name] = item.shape

    f.visititems(visit_item)
    return result


def find_potential_matches(model_layer_name, weight_names, threshold=0.6):
    """Find potential matching layer names using string similarity."""
    from difflib import SequenceMatcher

    def similarity(a, b):
        return SequenceMatcher(None, a, b).ratio()

    matches = []
    model_parts = set(model_layer_name.lower().replace("/", "_").split("_"))

    for weight_name in weight_names:
        weight_parts = set(weight_name.lower().replace("/", "_").split("_"))
        # Check for common terms
        common_terms = model_parts.intersection(weight_parts)
        sim_ratio = similarity(model_layer_name, weight_name)

        if sim_ratio > threshold or len(common_terms) >= 2:
            matches.append((weight_name, sim_ratio))

    return [m[0] for m in sorted(matches, key=lambda x: x[1], reverse=True)]


def suggest_fixes(model, weights_path):
    """Suggest potential fixes for weight loading issues."""
    print("\nSuggested Fixes:")
    print("-" * 50)

    # Check if the model has name scopes
    has_name_scopes = any("/" in layer.name for layer in model.layers)
    if has_name_scopes:
        print("- Your model uses name scopes. Try flattening layer names:")
        print("  model.load_weights(weights_path, skip_mismatch=True, by_name=True)")
        print("  or rename layers to match the weights file structure")

    # Check for common naming patterns
    common_prefixes = ["layer_", "block_", "conv_", "batch_norm_"]
    has_common_prefixes = any(
        any(layer.name.startswith(prefix) for prefix in common_prefixes) for layer in model.layers
    )
    if has_common_prefixes:
        print(
            "\n- Your layers use common prefixes. Ensure weight file uses same naming convention"
        )
        print("  or try removing prefixes from layer names")

    print("\n- Try printing model.summary() and comparing with:")
    print("  tf.train.list_variables(weights_path)")

    print("\n- Consider using custom weight loading:")
    print("  for layer in model.layers:")
    print("      if layer.name in weights_dict:")
    print("          layer.set_weights(weights_dict[layer.name])")


if __name__ == "__main__":
    data_path = "/mnt/f/ssl_images/data"  # ssl_images/data  if you're Letriça
    data_dir = os.path.join(data_path, "processed", "pascal_voc")
    metadata_dir = os.path.join(data_path, "pascal_voc", "ImageSets", "Main")
    pretrained_model = os.path.join("models", "checkpoints_resnet50", "best_model.h5")  # None

    model = ResNet50((224, 224, 1), mode="colorization")
    print(model.summary())

    # if pretrained_model:
    # print("Loading model weights...")
    # load_encoder_weights(model, pretrained_model)

    # 1. First, verify the weight loading process
    verify_weight_loading(model, pretrained_model)

    # 2. Check for potentially unloaded layers
    # suspicious_layers = get_unloaded_layers(model)
    # if suspicious_layers:
    #    print("\nPotentially unloaded layers:")
    #    for layer in suspicious_layers:
    #        print(f"- {layer['layer_name']} ({layer['layer_type']})")

    # debug_weight_loading(model, pretrained_model)
    # suggest_fixes(model, pretrained_model)
