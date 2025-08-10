from kfp.v2.dsl import component, Input, Output, Dataset, Model, Artifact


@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.1-13.py310",
           packages_to_install=["unsloth", "transformers[vision]", "bitsandbytes", "accelerate","peft","trl",
                               "xformers", "timm", "editdistance", "nltk"])
def evaluate_model(processed_test_data: Input[Dataset],
                   model_path: Input[Model],
                   evaluation_metrics: Output[Artifact]):
    """
    Evaluates the trained model.

    Args:
        processed_test_data (Input[Dataset]): Input processed test data.
        model (Input[Model]): Input trained model.
        evaluation_metrics (Output[Artifact]): Output evaluation metrics.
    """
    import unsloth
    from unsloth import FastVisionModel
    import os
    import gc
    import pandas as pd
    import numpy as np
    import editdistance
    import joblib
    import json
    import torch
    from transformers import AutoModelForCausalLM, AutoProcessor
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from peft import PeftModel

    # Metrics functions
    def average_edit_distance(y_true, y_pred):
        return np.mean([editdistance.eval(t, p) for t, p in zip(y_true, y_pred)])


    def normalized_edit_accuracy(y_true, y_pred):
        return np.mean([
            1 - editdistance.eval(t, p) / max(len(t), 1) for t, p in zip(y_true, y_pred)
        ])


    def average_bleu(y_true, y_pred):
        smoothie = SmoothingFunction().method4
        return np.mean([
            sentence_bleu([list(t)], list(p), smoothing_function=smoothie)
            for t, p in zip(y_true, y_pred)
        ])

    def exact_match_accuracy(y_true, y_pred):
        return np.mean([t == p for t, p in zip(y_true, y_pred)])    
    
    

    # Inference function
    def do_gemma_3n_inference(model, processor, image, max_new_tokens=128):
        instruction = "Write the LaTeX representation for this image."

        messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image", "image": image}
                    ]
                }
        ]

        # Tokenize input with chat template
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        # Generate output (no streamer as in the initial Notebook)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.95,
            top_k=64,
            do_sample=True
        )

        # Decode just the new generated tokens (excluding prompt)
        generated_text = processor.decode(
            output_ids[0][inputs['input_ids'].shape[-1]:], 
            skip_special_tokens=True
        )

        # Cleanup to reduce VRAM usage
        del inputs
        torch.cuda.empty_cache()
        gc.collect()

        return generated_text


    
    # Load processed data and the model
    test_dataset = joblib.load(processed_test_data.path)
    
    # Load model
    lora_weights_path = os.path.join(model_path.path, "model")
    
    base_model_path = "unsloth/gemma-3n-E4B"
    
    # Reconstruct model
    model, processor = FastVisionModel.from_pretrained(
        model_name = base_model_path,
        load_in_4bit = True,
        full_finetuning = False,
        dtype = None
    )

    model = PeftModel.from_pretrained(model, lora_weights_path)
    
    # Enable model for inference
    FastVisionModel.for_inference(model)
    
    # Use only a subset of test data
    subset_valid = test_dataset.select(range(50))
    
   
    # Run inference
    y_pred = []
    for i in tqdm(range(0, len(subset_valid)), desc="Running inference"):
        image = subset_valid[i]["image"]
        output = do_gemma_3n_inference(model, processor, image)
        y_pred.append(output)
    
    # Trim the response
    unwanted = ["\nuser", "\n\nuser", "Write the LaTeX representation for this image", "\n\n"]
    y_pred_trimmed = y_pred.copy()
    for un in unwanted:
        y_pred_trimmed = [pred.replace(un, "") for pred in y_pred_trimmed]
        
    y_true = []
    for sample in subset_valid:
        y_true.append(sample["text"])

    # Average edit distance
    average_edit_distance_score = average_edit_distance(y_true, y_pred_trimmed)

    # Normalized edit accuracy
    normalized_edit_accuracy_score = normalized_edit_accuracy(y_true, y_pred_trimmed)

    # Average BLEU score
    average_bleu_score = average_bleu(y_true, y_pred_trimmed)

    # Exact match accuracy
    exact_match_accuracy_score = exact_match_accuracy(y_true, y_pred_trimmed)
    
    # Generate the metrics report
    metrics_report = {
        "Average edit distance": average_edit_distance_score,
        "Normalized edit accuracy": normalized_edit_accuracy_score,
        "Average BLEU score": average_bleu_score,
        "Exact match accuracy": exact_match_accuracy_score
    }

    # Save metrics to the artifact
    with open(evaluation_metrics.path, "w") as f:
        json.dump(metrics_report, f)

    print(f"Evaluation metrics: {metrics_report}")    
