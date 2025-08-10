from kfp.v2.dsl import component, Input, Model, Artifact

@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310",
           packages_to_install=["unsloth", "transformers[vision]", "bitsandbytes", "accelerate","peft","trl",
                               "xformers", "timm", "kagglehub"])
def publish_model(
    model_path: Input[Model], 
    evaluation_metrics: Input[Artifact],
    kaggle_username: str,
    kaggle_key: str
):
    """
    Publish a finetuned model in Kaggle Models.
    
    Args:
        model_path (Input[Model]): Input trained model artifact path.
    """
    import os
    import json
    import joblib
    import time
    import kagglehub
    from pathlib import Path

    
    # Load evaluation metrics from the artifact
    with open(evaluation_metrics.path, "r") as f:
        evaluation_metrics_data = json.load(f)
            
    # Load model
    load_model_path = os.path.join(model_path.path, "model")
    
    # Publish model on Kaggle Models
    os.environ["KAGGLE_USERNAME"] = kaggle_username
    os.environ["KAGGLE_KEY"] = kaggle_key
    
    kaggle_username = os.environ["KAGGLE_USERNAME"]

    kaggle_uri = f"{kaggle_username}/gemma3n-latex-math-eq-lora-weights/transformers/gemma3n_4b_latex_math_eq_lora_weights"
    kagglehub.model_upload(kaggle_uri, 
                           load_model_path, 
                           license_name='Apache 2.0', 
                           model_metadata=evaluation_metrics_data)
    
    print("Model uploaded to: ", kaggle_uri)