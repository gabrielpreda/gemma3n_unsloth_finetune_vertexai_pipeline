from kfp.v2.dsl import pipeline
from components.preprocessing import preprocess_data
from components.training import train_model
# from components.evaluation import evaluate_model
# from components.register import register_model

@pipeline(name="gemma3n_unsloth_finetune", pipeline_root="gs://gemma3n-unsloth-finetune/gemma3n_unsloth_finetune_pipeline")
def gemma3n_unsloth_finetune_pipeline(
    project: str,
    location: str,
    dataset_path: str):
    """
    Pipeline for train and evaluate the Gemma3n 4B model for image to LateX conversion.

    Args:
        project: Input GCP project name.
        location: Input location (e.g. `us-central1`)
        dataset_path: Unsloth dataset path
    """
    
    # 1. Preprocess data
    preprocess_task = preprocess_data(
        dataset_path=dataset_path   
    ).set_cpu_limit('8')\
    .set_memory_limit('16G')

    # 2. Train model
    train_task = train_model(
        processed_train_data=preprocess_task.outputs['processed_train_data'],
    ).set_gpu_limit(1)\
    .set_accelerator_type("NVIDIA_L4")\
    .set_cpu_limit('8')\
    .set_memory_limit('32G')
    
#     # 3. Evaluate model
#     evaluation_task = evaluate_model(
#         processed_test_data=preprocess_task.outputs['processed_test_data'],
#         test_labels_data=preprocess_task.outputs['test_labels_data'],
#         model=train_task.outputs['model'],
#     )
    
#     # 4. Register model
#     register_model(
#         project=project,
#         location=location,
#         model=train_task.outputs['model'],
#         metrics=evaluation_task.outputs['evaluation_metrics'],
#     )
