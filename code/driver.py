import torch
import numpy as np
import toml
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from pipelines import repe_pipeline_registry
from probe import complexity_function_datast
from decision_maker import DecisionMaker


class ComplexityClassifier:
    def __init__(self, model_name):
        """Initialize model, tokenizer, and rep-reading pipeline."""
        repe_pipeline_registry()

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            device_map=None
        )

        # Load tokenizer
        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast_tokenizer,
            padding_side="left",
            legacy=False
        )
        self.tokenizer.pad_token_id = 0

        # Build rep pipeline
        self.rep_token = -1
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        self.n_difference = 1
        self.direction_method = "pca"

        self.pipeline = pipeline(
            "rep-reading",
            model=self.model,
            tokenizer=self.tokenizer
        )

        self.feature_embedding = None
        self.decision_model = None

    def train(self, data_path, sample_size, seed, batch_size):
        """Train rep reader on training dataset and build feature embeddings."""
        dataset = complexity_function_datast(
            data_path=data_path,
            sample_size=sample_size,
            seed=seed
        )

        self.feature_embedding = {}
        for complexity in ["simple_claims", "intermediate_claims", "complex_claims"]:
            self.feature_embedding[complexity] = self.pipeline.get_directions(
                dataset[complexity],
                rep_token=self.rep_token,
                hidden_layers=self.hidden_layers,
                n_difference=self.n_difference,
                direction_method=self.direction_method,
                batch_size=batch_size,
                mean_pool="mean_pooling"
            ).directions

        self.decision_model = DecisionMaker(feature_embedding=self.feature_embedding)

    def predict(self, data_path, batch_size, claim_key="claims", test=True):
        """Run predictions on test dataset."""
        if self.decision_model is None:
            raise ValueError("Model not trained. Call `.train()` first.")

        dataset = complexity_function_datast(
            data_path=data_path,
            test=test
        )
        test_claims = dataset[claim_key]

        print("Embedding test claims ...")
        test_embedding = self.pipeline._batched_string_to_hiddens(
            test_claims,
            self.rep_token,
            self.hidden_layers,
            batch_size=batch_size
        )

        # Stack embeddings across layers
        layer_list = [test_embedding[layer] for layer in test_embedding.keys()]
        test_embedding = np.transpose(np.array(layer_list), (1, 0, 2))

        print("Making predictions ...")
        return self.decision_model.make_decision(test_embedding=test_embedding)

    def save_predictions(self, predictions, output_path="output/test_decision.txt"):
        """Save predictions to text file, one per line."""
        with open(output_path, "w") as f:
            f.write("\n".join(map(str, predictions)))


if __name__ == "__main__":
    # Load config
    config = toml.load("code/config.toml")

    model_name = config["model"]["name"]
    train_cfg = config["training"]
    test_cfg = config["testing"]

    # Initialize classifier
    classifier = ComplexityClassifier(model_name=model_name)

    # Train model
    classifier.train(
        data_path=train_cfg["data_path"],
        sample_size=train_cfg["sample_size"],
        seed=train_cfg["seed"],
        batch_size=train_cfg["batch_size"],
    )

    # Predict on test set
    preds = classifier.predict(
        data_path=test_cfg["data_path"],
        batch_size=test_cfg["batch_size"],
        claim_key=test_cfg["claim_key"]
    )

    # Save results
    classifier.save_predictions(preds, output_path = test_cfg["output_path"])
