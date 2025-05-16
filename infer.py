import os
import argparse
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from skimage import io
import json

from model import Classifier


class PPGLInferenceEngine:
    def __init__(self, model_path, device=None):
        """
        Initialize the inference engine with a trained model

        Args:
            model_path (str): Path to the trained model checkpoint
            device (str, optional): Device to run inference on ('cuda' or 'cpu')
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Using device: {self.device}")

        # Initialize model
        self.model = self._load_model(model_path)

        # Set up image transformation
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.1737, 0.1737, 0.1737], std=[0.2584, 0.2584, 0.2584]
                ),
            ]
        )

        # Define class mappings
        self.genetic_clusters = {
            0: "Kinase Signaling",
            1: "SDHx",
            2: "Sporadic",
            3: "VHL/EPAS1",
        }

        self.anatomical_regions = {
            0: "Abdomen",
            1: "Chest",
            2: "Head & Neck",
            3: "Unknown",  # This is not used in the model but included for completeness
        }

    def _load_model(self, model_path):
        """
        Load the trained model from checkpoint

        Args:
            model_path (str): Path to the model checkpoint

        Returns:
            model: Loaded PyTorch model
        """
        # Initialize model with default parameters
        # These parameters don't matter for inference
        model = Classifier(lr_dino=1e-5, lr_class=1e-2, weight_decay=0.0005, k=0)

        # Load the state dict
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        model.to(self.device)

        return model

    def preprocess_image(self, image_path):
        """
        Preprocess an image for inference

        Args:
            image_path (str): Path to the image file

        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Load image
        try:
            img = io.imread(image_path)

            # Check if the image has an alpha channel and remove it
            if img.shape[-1] == 4:
                img = img[..., 0:-1]

            # Apply transformations
            img_tensor = self.transform(img)

            # Add batch dimension
            img_tensor = img_tensor.unsqueeze(0)

            return img_tensor.to(self.device)

        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def predict(self, image_path):
        """
        Run inference on a single image

        Args:
            image_path (str): Path to the image file

        Returns:
            dict: Prediction results including genetic cluster and anatomical region
        """
        # Preprocess image
        img_tensor = self.preprocess_image(image_path)
        if img_tensor is None:
            return None

        # Create dummy region tensor (not used during inference)
        dummy_region = torch.zeros((1, 4), device=self.device)

        with torch.no_grad():
            # Forward pass
            genetic_pred, region_pred = self.model.forward(
                img_tensor, dummy_region, train=False
            )

            # Get predicted classes
            genetic_class_idx = torch.argmax(genetic_pred, dim=1).item()
            region_class_idx = torch.argmax(region_pred, dim=1).item()

            # Get class probabilities
            genetic_probs = torch.softmax(genetic_pred, dim=1)[0].cpu().numpy()
            region_probs = torch.softmax(region_pred, dim=1)[0].cpu().numpy()

            # Create result dictionary
            result = {
                "genetic_cluster": {
                    "prediction": self.genetic_clusters[genetic_class_idx],
                    "probabilities": {
                        self.genetic_clusters[i]: float(genetic_probs[i])
                        for i in range(len(genetic_probs))
                    },
                },
                "anatomical_region": {
                    "prediction": self.anatomical_regions[region_class_idx],
                    "probabilities": {
                        self.anatomical_regions[i]: float(region_probs[i])
                        for i in range(len(region_probs))
                    },
                },
            }

            return result

    def visualize_prediction(self, image_path, result, output_path=None):
        """
        Visualize the prediction results

        Args:
            image_path (str): Path to the input image
            result (dict): Prediction results
            output_path (str, optional): Path to save the visualization
        """
        # Load image
        img = io.imread(image_path)
        if img.shape[-1] == 4:
            img = img[..., 0:-1]

        # Create figure
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

        # Plot image
        ax1.imshow(img)
        ax1.set_title("Input CT Image")
        ax1.axis("off")

        # Plot genetic cluster probabilities
        genetic_labels = list(result["genetic_cluster"]["probabilities"].keys())
        genetic_values = list(result["genetic_cluster"]["probabilities"].values())

        ax2.bar(genetic_labels, genetic_values, color="skyblue")
        ax2.set_title("Genetic Cluster Probabilities")
        ax2.set_ylabel("Probability")
        ax2.set_ylim(0, 1)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Plot anatomical region probabilities
        region_labels = list(result["anatomical_region"]["probabilities"].keys())
        region_values = list(result["anatomical_region"]["probabilities"].values())

        ax3.bar(region_labels, region_values, color="salmon")
        ax3.set_title("Anatomical Region Probabilities")
        ax3.set_ylabel("Probability")
        ax3.set_ylim(0, 1)
        plt.setp(ax3.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add prediction text
        fig.suptitle(
            f"Predicted Genetic Cluster: {result['genetic_cluster']['prediction']} | "
            + f"Predicted Anatomical Region: {result['anatomical_region']['prediction']}",
            fontsize=16,
        )

        plt.tight_layout()

        # Save or show
        if output_path:
            plt.savefig(output_path)
            print(f"Visualization saved to {output_path}")
        else:
            plt.show()

        plt.close()


def batch_inference(engine, input_dir, output_dir=None, visualize=False):
    """
    Run inference on all images in a directory

    Args:
        engine (PPGLInferenceEngine): Inference engine
        input_dir (str): Directory containing images
        output_dir (str, optional): Directory to save results
        visualize (bool): Whether to generate visualizations
    """
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        if visualize:
            vis_dir = os.path.join(output_dir, "visualizations")
            os.makedirs(vis_dir, exist_ok=True)

    results = {}

    # Get all image files
    image_files = [
        f
        for f in os.listdir(input_dir)
        if f.endswith((".png", ".jpg", ".jpeg", ".tiff", ".tif"))
    ]

    for img_file in image_files:
        img_path = os.path.join(input_dir, img_file)
        print(f"Processing {img_path}...")

        # Run inference
        result = engine.predict(img_path)

        if result:
            results[img_file] = result

            # Generate visualization if requested
            if visualize and output_dir:
                vis_path = os.path.join(
                    vis_dir, f"{os.path.splitext(img_file)[0]}_viz.png"
                )
                engine.visualize_prediction(img_path, result, vis_path)

    # Save results to JSON if output directory is provided
    if output_dir:
        results_path = os.path.join(output_dir, "inference_results.json")
        with open(results_path, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {results_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="PPGL Genetic Cluster Classification Inference"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint",
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Path to input image or directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Directory to save results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run inference on (cuda or cpu)",
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Generate visualizations"
    )

    args = parser.parse_args()

    # Initialize inference engine
    engine = PPGLInferenceEngine(args.model_path, args.device)

    # Check if input is a directory or a single image
    if os.path.isdir(args.input):
        # Batch inference
        batch_inference(engine, args.input, args.output_dir, args.visualize)
    else:
        # Single image inference
        result = engine.predict(args.input)

        if result:
            print("\nPrediction Results:")
            print(f"Genetic Cluster: {result['genetic_cluster']['prediction']}")
            print("Genetic Cluster Probabilities:")
            for cluster, prob in result["genetic_cluster"]["probabilities"].items():
                print(f"  {cluster}: {prob:.4f}")

            print(f"\nAnatomical Region: {result['anatomical_region']['prediction']}")
            print("Anatomical Region Probabilities:")
            for region, prob in result["anatomical_region"]["probabilities"].items():
                print(f"  {region}: {prob:.4f}")

            # Generate visualization if requested
            if args.visualize:
                if args.output_dir:
                    os.makedirs(args.output_dir, exist_ok=True)
                    vis_path = os.path.join(args.output_dir, f"visualization.png")
                    engine.visualize_prediction(args.input, result, vis_path)
                else:
                    engine.visualize_prediction(args.input, result)

            # Save result to JSON if output directory is provided
            if args.output_dir:
                os.makedirs(args.output_dir, exist_ok=True)
                results_path = os.path.join(args.output_dir, "inference_result.json")
                with open(results_path, "w") as f:
                    json.dump(result, f, indent=4)
                print(f"\nResult saved to {results_path}")


if __name__ == "__main__":
    main()
