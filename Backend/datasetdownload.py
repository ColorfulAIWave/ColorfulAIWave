from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from pydantic import BaseModel
from datasets import load_dataset, Dataset, DatasetDict, disable_caching, load_from_disk
from typing import Optional
import os
import gc
import json
import pandas as pd
import ast
import shutil
import logging
from fastapi.responses import FileResponse



router = APIRouter()

disable_caching()

# Define the static directory to save the datasets
SAVE_DIR = "./local_datasets"

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the data model for the expected JSON body
class DatasetRequest(BaseModel):
    name: str
    version: str = None

def add_system_message(example):
    messages = example["messages"]
    # Add an empty system message if there is none
    if messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})
    example["messages"] = messages  # Ensure the modified messages are saved back
    return example

def find_and_process_messages_column(dataset_dict):
    """
    Iterates over all available datasets (e.g., train, test) in the dataset dictionary
    to find a column that contains a list of dictionaries with 'content' and 'role' keys.

    Parameters:
    - dataset_dict: The dataset dictionary, typically containing splits like train, test, etc.

    Returns:
    - A dictionary where keys are dataset split names (e.g., train, test) and values are 
      the identified columns with the required structure, or an empty dict if none found.
    """
    columns_to_process = {}

    for split_name, dataset in dataset_dict.items():
        # Iterate over columns to find a list of dictionaries with "content" and "role" keys
        for column in dataset.column_names:
            if isinstance(dataset[column][0], list):
                if all(isinstance(item, dict) for item in dataset[column][0]):
                    # Check if dictionaries have "content" and "role" keys
                    if all("content" in item and "role" in item for item in dataset[column][0]):
                        columns_to_process[split_name] = column
                        break  # Exit loop after finding the first suitable column for this split

    return columns_to_process


@router.post("/download-dataset/")
async def download_dataset(request: DatasetRequest):
    """
    Downloads a dataset from the Hugging Face Hub and saves it locally.

    Parameters:
    - name: The name of the dataset to download (from JSON body).
    - version: The specific version of the dataset (optional, from JSON body).

    Returns:
    - A success message or an error if the dataset cannot be downloaded.
    """
    try:
        # Extract dataset name and version from the request body
        name = request.name
        version = request.version

        # Load the dataset from Hugging Face
        dataset = load_dataset(name, version)

        # Ensure the save directory exists
        dataset_save_path = os.path.join(SAVE_DIR, name)
        os.makedirs(dataset_save_path, exist_ok=True)

        # Save the dataset to disk
        dataset.save_to_disk(dataset_save_path)

        # Explicitly delete the dataset object to free memory
        del dataset

        # Manually invoke garbage collection to clear any lingering objects
        gc.collect()

        return {"status": "success", "dataset": name, "saved_to": dataset_save_path}
    except Exception as e:
        # Return an error message if the download or saving fails
        raise HTTPException(status_code=400, detail=f"Failed to download and save dataset: {str(e)}")


@router.post("/download-and-process-dataset/")
async def download_and_process_dataset(request: DatasetRequest):
    """
    Downloads a dataset, processes it by adding a system message or creating
    a messages column if necessary, and saves it locally.

    Parameters:
    - name: The name of the dataset to download (from JSON body).
    - version: The specific version of the dataset (optional, from JSON body).

    Returns:
    - A success message or an error if the dataset cannot be downloaded, processed, or saved.
    """
    try:
        # Extract dataset name and version from the request body
        name = request.name.replace("/", "--")  # Replace slashes to avoid nested folders
        version = request.version

        # Load the dataset from Hugging Face
        raw_dataset = load_dataset(request.name, version)

        # Check if the dataset is 'izumi-lab/llm-japanese-dataset'
        if request.name == "izumi-lab/llm-japanese-dataset":
            # Define the transformation function for the 'izumi-lab/llm-japanese-dataset'
            def transform_example(example):
                input_instruction = example["input"] + " " + example["instruction"]
                output = example["output"]

                # Create the new structure
                transformed_example = {
                    "prompt": input_instruction,
                    "messages": [
                        {"role": "user", "content": input_instruction},
                        {"role": "assistant", "content": output}
                    ]
                }
                return transformed_example

            # Apply the transformation to the entire dataset (all splits)
            for split in raw_dataset.keys():
                raw_dataset[split] = raw_dataset[split].map(
                    transform_example,
                    num_proc=10,
                    desc=f"Transforming {split} - creating messages column",
                )
        else:
            # Find appropriate columns to process in all splits
            columns_to_process = find_and_process_messages_column(raw_dataset)

            if columns_to_process:
                # Apply the system message addition to the identified columns in all splits
                for split_name, column in columns_to_process.items():
                    raw_dataset[split_name] = raw_dataset[split_name].map(
                        lambda example: add_system_message({"messages": example[column]}),
                        num_proc=10,
                        desc=f"Adding system message to {split_name} - {column} column",
                    )
            else:
                raise HTTPException(status_code=400, detail="No suitable column found for processing.")

        # Save the processed dataset to disk
        dataset_save_path = os.path.join(SAVE_DIR, name)
        os.makedirs(dataset_save_path, exist_ok=True)
        raw_dataset.save_to_disk(dataset_save_path)

        # Explicitly delete the datasets to free memory
        del raw_dataset

        # Manually invoke garbage collection to clear any lingering objects
        gc.collect()

        return {"status": "success", "dataset": name, "saved_to": dataset_save_path}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download, process, or save dataset: {str(e)}")


# Define the data model for the expected JSON body
class FilePathRequest(BaseModel):
    path: str

# Define the data model for the expected JSON body
class FilePathRequest(BaseModel):
    path: str

@router.post("/upload-jsonl/")
async def upload_jsonl(request: FilePathRequest):
    """
    Uploads a JSONL file from a given file path and checks if each line contains a 'messages' key.

    Parameters:
    - path: The file path of the JSONL file to upload (from JSON body).

    Returns:
    - A success message if the file is uploaded and validated, or an error if the 'messages' key is missing.
    """
    try:
        # Read the file from the specified path
        file_path = request.path
        if not os.path.exists(file_path):
            raise HTTPException(status_code=400, detail=f"File not found: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()

        # Check each line for the 'messages' key
        for i, line in enumerate(lines):
            try:
                data = json.loads(line)
                if "messages" not in data:
                    raise HTTPException(status_code=400, detail=f"'messages' key not found in line {i + 1}.")
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"JSON decoding error in line {i + 1}: {str(e)}")

        # Create a directory inside SAVE_DIR named after the file (without extension)
        base_name = os.path.splitext(os.path.basename(file_path))[0]  # Get file name without extension
        save_dir = os.path.join(SAVE_DIR, base_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save the file within the new directory
        save_path = os.path.join(save_dir, os.path.basename(file_path))
        with open(save_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        return {"status": "success", "filename": os.path.basename(file_path), "saved_to": save_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")

def get_directory_size(directory):
    """
    Returns the size of a directory in kilobytes (KB).
    """
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            if os.path.exists(file_path):
                total_size += os.path.getsize(file_path)
    return total_size # Convert to KB

@router.get("/list_datasets/")
def get_available_datasets():
    """
    Returns a list of available datasets saved in the SAVE_DIR, including their size in KB.

    Returns:
    - A JSON list of dictionaries, each containing the "name" of the dataset (folder name),
      the "path" to it, and its "size" in kilobytes.
    """
    try:
        # List all directories in the SAVE_DIR
        datasets = []
        if os.path.exists(SAVE_DIR):
            for folder_name in os.listdir(SAVE_DIR):
                folder_path = os.path.join(SAVE_DIR, folder_name)
                if os.path.isdir(folder_path):
                    # Get the size of the directory
                    size_in_kb = get_directory_size(folder_path)
                    datasets.append({
                        "name": folder_name,
                        "path": folder_path,
                        "size": round(size_in_kb, 2)  # Round size to 2 decimal places
                    })
        
        return datasets
    except Exception as e:
        # Return an error message if something goes wrong
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available datasets: {str(e)}")
    
@router.get("/list_uploaded_datasets/")
def get_uploaded_datasets():
    """
    Returns a list of available dataset names saved in the SAVE_DIR.
    """
    try:
        datasets = []
        if os.path.exists(SAVE_DIR):
            for folder_name in os.listdir(SAVE_DIR):
                folder_path = os.path.join(SAVE_DIR, folder_name)
                if os.path.isdir(folder_path):
                    # Append only the folder name to the list of datasets
                    datasets.append(folder_name)
        
        return datasets
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve available datasets: {str(e)}")


class FilePathRequest(BaseModel):
    path: str

def convert_messages_column(df):
    """
    Converts a 'messages' column from a string back to a list of dictionaries using literal_eval.
    """
    try:
        df['messages'] = df['messages'].apply(ast.literal_eval)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to convert 'messages' column: {str(e)}")
    return df

def transform_prompt_output(df):
    """
    Transforms the dataset with 'prompt' and 'output' columns into a 'messages' column.
    """
    try:
        df['messages'] = df.apply(lambda x: [
            {"role": "user", "content": x['prompt']},
            {"role": "assistant", "content": x['output']}
        ], axis=1)
        return df
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to transform dataset: {str(e)}")

# @router.post("/uploaddataset/")
# async def process_csv(request: FilePathRequest):
#     """
#     Processes a CSV file specified by the file path in the JSON body. 
#     It checks for a 'messages' column to convert back to a list of dicts, 
#     or processes 'prompt' and 'output' columns to create a 'messages' column.

#     Parameters:
#     - path: The file path of the CSV file (from JSON body).

#     Returns:
#     - A success message if the file is processed and saved, or an error if something goes wrong.
#     """
#     try:
#         # Extract the file path from the request
#         file_path = request.path
        
#         # Check if the file exists
#         if not os.path.exists(file_path):
#             raise HTTPException(status_code=400, detail=f"File not found: {file_path}")
        
#         # Load the CSV file into a Pandas DataFrame
#         df = pd.read_csv(file_path)

#         # Process the CSV file depending on the columns it contains
#         if 'messages' in df.columns:
#             df = convert_messages_column(df)
#         elif all(col in df.columns for col in ['prompt', 'output']):
#             df = transform_prompt_output(df)
#         else:
#             raise HTTPException(status_code=400, detail="CSV does not have the required columns ('prompt', 'output').")
        
#         # Convert the DataFrame to a Hugging Face Dataset
#         hf_dataset = Dataset.from_pandas(df)

#         # Create a directory inside SAVE_DIR named after the file (without extension)
#         base_name = os.path.splitext(os.path.basename(file_path))[0]
#         save_dir = os.path.join(SAVE_DIR, base_name)
#         os.makedirs(save_dir, exist_ok=True)

#         # # Save the processed DataFrame to a new CSV file within the new directory
#         # save_path = os.path.join(save_dir, f"{base_name}_processed.csv")
#         # Save the dataset in the Hugging Face format
#         hf_dataset.save_to_disk(save_dir)

#         # Explicitly delete the DataFrame to free memory
#         del df, hf_dataset
#         gc.collect()

#         return {"status": "success", "filename": os.path.basename(file_path), "saved_to": save_dir}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.post("/uploaddataset/")
async def process_csv(file: UploadFile = File(...)):
    """
    Processes an uploaded CSV file. 
    It checks for a 'messages' column to convert back to a list of dicts, 
    or processes 'prompt' and 'output' columns to create a 'messages' column.

    Parameters:
    - file: The uploaded CSV file (from form data).

    Returns:
    - A success message if the file is processed and saved, or an error if something goes wrong.
    """
    try:
        # Ensure that SAVE_DIR exists
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # Read the contents of the uploaded file
        contents = await file.read()

        # Save the uploaded file to a temporary location
        file_path = os.path.join(SAVE_DIR, file.filename)
        with open(file_path, 'wb') as f:
            f.write(contents)

        # Load the CSV file into a Pandas DataFrame
        df = pd.read_csv(file_path)

        # Process the CSV file depending on the columns it contains
        if 'messages' in df.columns:
            df = convert_messages_column(df)
        elif all(col in df.columns for col in ['prompt', 'output']):
            df = transform_prompt_output(df)
        else:
            raise HTTPException(status_code=400, detail="CSV does not have the required columns ('prompt', 'output').")

        # Convert the DataFrame to a Hugging Face Dataset
        hf_dataset = Dataset.from_pandas(df)

        # Create a directory inside SAVE_DIR named after the file (without extension)
        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        save_dir = os.path.join(SAVE_DIR, base_name)
        os.makedirs(save_dir, exist_ok=True)

        # Save the dataset in the Hugging Face format
        hf_dataset.save_to_disk(save_dir)

        # Delete the original CSV file after processing
        os.remove(file_path)

        # Explicitly delete the DataFrame to free memory
        del df, hf_dataset
        gc.collect()

        return {"status": "success", "filename": file.filename, "saved_to": save_dir}
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")
    
@router.delete("/deletedataset/{dataset_name}")
async def delete_dataset(dataset_name: str):
    """
    Deletes a dataset by its name from the SAVE_DIR.

    Parameters:
    - dataset_name: The name of the dataset to delete (from the URL).

    Returns:
    - A success message or an error if the dataset cannot be found or deleted.
    """
    try:
        # Construct the dataset directory path
        dataset_path = os.path.join(SAVE_DIR, dataset_name)

        # Check if the dataset directory exists
        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Delete the dataset directory and all its contents
        shutil.rmtree(dataset_path)

        return {"status": "success", "message": f"Dataset '{dataset_name}' deleted successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete dataset: {str(e)}")
    
@router.get("/dataset_info/{dataset_name}")
async def get_dataset_info(dataset_name: str, page: int = Query(1), rows_per_page: int = Query(10)):
    try:
        dataset_path = os.path.join(SAVE_DIR, dataset_name)

        logger.info(f"Loading dataset from: {dataset_path}")

        if not os.path.exists(dataset_path):
            raise HTTPException(status_code=404, detail="Dataset not found")

        dataset = load_from_disk(dataset_path)

        if isinstance(dataset, DatasetDict):
            if "train" in dataset:
                dataset_split = dataset["train"]
            else:
                first_split = list(dataset.keys())[0]
                dataset_split = dataset[first_split]
        else:
            dataset_split = dataset

        total_rows = len(dataset_split)
        start = (page - 1) * rows_per_page
        end = start + rows_per_page

        if start >= total_rows:
            raise HTTPException(status_code=400, detail="Page number out of range")

        # Fetch the page data as columnar structure
        page_data = dataset_split[start:end]

        # Get the columns
        columns = dataset_split.column_names

        # Convert columnar data to row-based data (a list of dictionaries)
        page_data_list = []
        for i in range(len(page_data[columns[0]])):
            row = {col: page_data[col][i] for col in columns}
            # Convert any non-string column into a string
            for key, value in row.items():
                if not isinstance(value, str):
                    row[key] = str(value)  # Convert non-string values to string
            page_data_list.append(row)

        return {
            "dataset_name": dataset_name,
            "row_count": total_rows,
            "columns": columns,
            "data_preview": page_data_list
        }
    

    except Exception as e:
        logger.error(f"Failed to retrieve dataset info: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dataset info: {str(e)}")

@router.get("/files")
async def list_files():
    # List available datasets
    file_map = {
        "sample.csv": os.path.join(SAVE_DIR, "sample.csv"),
        "sample.jsonl": os.path.join(SAVE_DIR, "sample.jsonl"),
    }
    return [{"name": file_name} for file_name in file_map.keys()]

@router.get("/files/{file_type}")
async def get_sample_file(file_type: str):
    # Create a mapping of file types to file paths
    file_map = {
        "sample.csv": os.path.join(SAVE_DIR, "sample.csv"),
        "sample.jsonl": os.path.join(SAVE_DIR, "sample.jsonl"),
    }

    # Get the file path based on the file type
    file_path = file_map.get(file_type)

    try:
        # Check if the file path is valid and exists
        if not file_path or not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path)

    except Exception as e:
        # Handle any unexpected errors
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

 