from kfp.v2.dsl import component, Output, Dataset


@component(base_image="us-docker.pkg.dev/deeplearning-platform-release/gcr.io/tf2-cpu.2-17.py310",
           packages_to_install=["datasets", "transformers"])
def preprocess_data(dataset_path: str,
                    processed_train_data: Output[Dataset],
                    processed_test_data: Output[Dataset],
                   ):
    """
    Preprocess the data.

    Args:
        dataset_path (str): Unsloth dataset path.
        processed_train_data (Output[Dataset]): Path to the train data.
        processed_test_data (Output[Dataset]): Path to the test data.
    """
        
    # import packages
    from google.cloud import storage
    import pandas as pd
    
    from datasets import load_dataset

    import joblib
    
    def read_data(dataset_path="unsloth/LaTeX_OCR", 
                  split="train"):
        """
        Function to read data from unsloth
        Args:
            dataset_path (str): path to Unsloth dataset
            split (str): can be train (default) or test
            
        Returns:
            dataset: dataset with images and labels
        """
        dataset = load_dataset(dataset_path, split = split)
        print(dataset)
        
        return dataset
        
    

    def convert_to_conversation(sample):
        """
        Function to convert a sample to a conversation message
        Args:
            sample: a sample (image/text) from the dataset
            
        Returns:
            messages item: a conversation (user/assistant) with instructions and expected output
        """
        instruction = "Write the LaTeX representation for this image."
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    {"type": "image", "image": sample["image"]},
                ],
            },
            {
                "role": "assistant", 
                 "content": [
                     {"type": "text", "text": sample["text"]}
                 ]
            },
        ]
        return {"messages": conversation}

    def data_preprocessing(dataset):
        """
        Function to preprocess the (train/valid) data
        Apply conversation conversion
        Args:
            dataset: initial dataset (before conversion)
        Returns:
            converted_dataset: transformed dataset (after conversion)
        """
        converted_dataset = [convert_to_conversation(sample) for sample in dataset]
        return converted_dataset

    # Read train and test (valid) data
    train_dataset = read_data(split="train")
    test_dataset = read_data(split="test")

    # Preprocess the data
    converted_train_dataset = data_preprocessing(train_dataset)
    converted_test_dataset = data_preprocessing(test_dataset)
    
    # Write files    
    joblib.dump(converted_train_dataset, processed_train_data.path)
    joblib.dump(test_dataset, processed_test_data.path)

    

