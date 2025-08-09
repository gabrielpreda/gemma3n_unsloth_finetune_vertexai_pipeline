import argparse
from google.cloud import aiplatform
from kfp import compiler
from pipeline import gemma3n_unsloth_finetune_pipeline

compiler.Compiler().compile(
    pipeline_func=gemma3n_unsloth_finetune_pipeline,
    package_path="gemma3n_unsloth_finetune.json"
)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run Vertex AI Pipeline Job")
    parser.add_argument("--project", required=True, help="GCP Project ID")
    parser.add_argument("--location", required=True, help="GCP Location (region)")
    parser.add_argument("--dataset_path", default="unsloth/LaTeX_OCR", help="GCS Bucket for data")
    parser.add_argument("--pipeline_template", default="gemma3n_unsloth_finetune.json", help="Path to the pipeline template JSON")
    parser.add_argument("--pipeline_root", default="gs://gemma3n-unsloth-finetune/gemma3n_unsloth_finetune_pipeline", help="GCS path for pipeline root")

    args = parser.parse_args()

    # Initialize Vertex AI
    aiplatform.init(
        project=args.project,
        location=args.location
    )

        
    # Submit the pipeline job
    pipeline_job = aiplatform.PipelineJob(
        display_name="gemma3n-unsloth-finetune",
        template_path=args.pipeline_template,
        pipeline_root=args.pipeline_root,
        parameter_values={
            "project": args.project,
            "location": args.location,
            "dataset_path": args.dataset_path,
        },
        enable_caching=True  # Enable caching
    )

    pipeline_job.run(sync=True)

if __name__ == "__main__":
    main()

