from kfp.v2.dsl import component, Input, Output, Dataset, Model


@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/pytorch-gpu.1-13.py310",
          packages_to_install=["unsloth", "transformers", "bitsandbytes", "accelerate", 
                               "xformers", "peft", "trl", "triton", "ut_cross_entropy", "unsloth_zoo", "timm"])
def train_model(processed_train_data: Input[Dataset],
                model: Output[Model]):

    """
    Train the model.

    Args:
        processed_train_data (Input[Dataset]): Input processed train data.
        train_labels_data (Input[Dataset]): Input train labels data.
        model (Output[Model])): Output trained model.
    """
    from unsloth import FastVisionModel # FastLanguageModel for LLMs
    from unsloth import get_chat_template
    from unsloth.trainer import UnslothVisionDataCollator
    from trl import SFTTrainer, SFTConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import os
    import torch

    model, processor = FastVisionModel.from_pretrained(
        model_name = "unsloth/gemma-3n-E4B", # Or "unsloth/gemma-3n-E2B-it"
        dtype = None, # None for auto detection
        max_seq_length = 1024, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )
    
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers     = True, # False if not finetuning vision layers
        finetune_language_layers   = True, # False if not finetuning language layers
        finetune_attention_modules = True, # False if not finetuning attention layers
        finetune_mlp_modules       = True, # False if not finetuning MLP layers

        r = 32,                           # The larger, the higher the accuracy, but might overfit
        lora_alpha = 32,                  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
        use_rslora = False,               # We support rank stabilized LoRA
        loftq_config = None,              # And LoftQ
        target_modules = "all-linear",    # Optional now! Can specify a list if needed
    )
    
    processor = get_chat_template(
        processor,
        "gemma-3n"
    )
   
    trainer = SFTTrainer(
        model=model,
        train_dataset=converted_dataset,
        processing_class=processor.tokenizer,
        data_collator=UnslothVisionDataCollator(model, processor),
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            max_grad_norm = 0.3,            # max gradient norm based on QLoRA paper
            warmup_ratio = 0.03,
            max_steps = 5,
            #num_train_epochs = 2,          # Set this instead of max_steps for full training runs
            learning_rate = 2e-4,
            logging_steps = 1,
            save_strategy="steps",
            optim = "adamw_torch_fused",
            weight_decay = 0.01,
            lr_scheduler_type = "cosine",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none",             # For Weights and Biases

            # You MUST put the below items for vision finetuning:
            remove_unused_columns = False,
            dataset_text_field = "",
            dataset_kwargs = {"skip_prepare_dataset": True},
            max_length = 2048,
        )
    )
    
    # Show current memory stats
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    # Show final memory and time stats
    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory, 3)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    # Save the trained model
    save_model_path = os.path.join(model.path, "model")  # Save to a directory
    model.save_pretrained(save_model_path)
    processor.save_pretrained(save_model_path)

    print(f"Model and processor saved to: {save_model_path}")
