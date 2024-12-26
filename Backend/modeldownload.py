from fastapi import APIRouter, HTTPException, File, UploadFile, Request
from pydantic import BaseModel
from huggingface_hub import snapshot_download
import os
import pathlib
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from fastapi.responses import StreamingResponse, FileResponse
import gc
import torch
import requests
import json
import shutil
import tempfile
from fastapi import FastAPI

router = APIRouter()

# Define the root directory where the models will be saved
models_directory = "models"

# Define the path for the download records JSON file
records_file_path = os.path.join(models_directory, "download_records.json")

# lock = Lock()

# Ensure the models directory exists
if not os.path.exists(models_directory):
    os.makedirs(models_directory)


# Ensure the download records JSON file exists
if not os.path.exists(records_file_path):
    with open(records_file_path, "w") as records_file:
        json.dump([], records_file)  # Initialize with an empty list


def update_download_records(name: str, status: str, type_: str, path: str):
    # with lock:  # Ensure thread-safe access to the file
    with open(records_file_path, "r+") as file:
        # Load existing records
        records = json.load(file)

        # Add new record
        records.append({"name": name, "status": status, "type": type_, "path": path})

        # Write updated records back to the file
        file.seek(0)
        json.dump(records, file, indent=4)
        file.truncate()  # Remove any leftover content


# Pydantic model for the JSON body for repo download
class RepoDownloadRequest(BaseModel):
    repo_id: str
    hf_token: str = None


# Pydantic model for GGUF to Hugging Face conversion
class GGUFConversionRequest(BaseModel):
    model_id: str
    model_file_name: str


class DeleteModelRequest(BaseModel):
    model_name: str


@router.post("/delete_model/")
async def delete_model(request: DeleteModelRequest):
    model_name = request.model_name.replace("/", "--")  # Replace slashes with --

    try:
        # Construct the path directly from the model name
        model_path = os.path.join(models_directory, model_name)

        # Check if the path exists
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found")

        # Delete the file or directory
        if os.path.isdir(model_path):
            shutil.rmtree(model_path)  # Recursively delete a directory
        else:
            os.remove(model_path)  # Delete a single file

        # Update the JSON records
        # with lock:
        with open(records_file_path, "r+") as file:
            records = json.load(file)

            # Filter out the deleted model from records
            records = [record for record in records if record["name"] != model_name]

            # Write the updated records back to the file
            file.seek(0)
            json.dump(records, file, indent=4)
            file.truncate()  # Remove any leftover content

        return {"message": f"Model '{model_name}' deleted successfully!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting the model: {e}")


@router.get("/list_models/")
async def list_models():
    try:
        # with lock:  # Ensure thread-safe access to the file
        with open(records_file_path, "r") as file:
            records = json.load(file)
        return records
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading the download records: {e}"
        )


@router.get("/list_saved_models/")
async def list_saved_models():
    try:
        with open(records_file_path, "r") as file:
            records = json.load(file)
        # Extract only the 'name' field for models with status "untrained"
        untrained_model_names = [
            record["name"] for record in records if record.get("status") == "untrained"
        ]
        return untrained_model_names
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading the download records: {e}"
        )


@router.get("/list_chat_models/")
async def list_chat_models():
    try:
        with open(records_file_path, "r") as file:
            records = json.load(file)
        # Extract only the 'name' field for models with status "untrained"
        untrained_model_names = [
            record["name"] for record in records if record.get("type") == "huggingface"
        ]
        return untrained_model_names
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading the download records: {e}"
        )


@router.get("/list_processed_models/")
async def list_trained_models():
    try:
        with open(records_file_path, "r") as file:
            records = json.load(file)
        # Extract only the 'name' field for models with status "untrained"
        trained_model_names = [
            record["name"] for record in records if record.get("status") != "untrained"
        ]
        return trained_model_names
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading the download records: {e}"
        )


# Route for file upload (previously defined)
@router.post("/upload_model/")
async def upload_file(file: UploadFile = File(...)):
    file_name = pathlib.Path(file.filename).stem
    file_directory = os.path.join(models_directory, file_name)

    # Ensure the directory for the file exists
    if not os.path.exists(file_directory):
        os.makedirs(file_directory)

    file_path = os.path.join(file_directory, file.filename)

    with open(file_path, "wb") as buffer:
        buffer.write(await file.read())
    update_download_records(
        name=file_name.replace("/", "--"),
        status="untrained",
        type_="GGUF",
        path=file_path,
    )
    return {
        "filename": file.filename,
        "directory": file_directory,
        "message": "File uploaded successfully!",
    }


def execute_command(command):
    # Use 'Popen' with line-buffered output
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,  # Line-buffered mode
        universal_newlines=True,  # Text mode
    )

    for stdout_line in iter(process.stdout.readline, ""):
        yield stdout_line
    for stderr_line in iter(process.stderr.readline, ""):
        yield stderr_line
    process.stdout.close()
    process.stderr.close()
    process.wait()


@router.post("/download_general_model/")
async def download_repo(request: RepoDownloadRequest):
    repo_name = request.repo_id.replace("/", "--")  # Replace slashes with --

    # Determine the model type based on the presence of "onnx" in repo_id
    model_type = "ONNX" if "onnx" in request.repo_id.lower() else "huggingface"

    # Use the modified repo_name to create a subdirectory in the models directory
    model_directory = os.path.join(models_directory, repo_name)

    # Ensure the directory for the model exists
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    try:
        # Construct the huggingface-cli command to download the model
        command = [
            "huggingface-cli",
            "download",
            request.repo_id,
            "--local-dir",
            model_directory,
        ]
        if request.hf_token:
            command.extend(["--use-auth-token", request.hf_token])

        # Execute the command and capture the output
        process = subprocess.run(
            command, text=True, capture_output=True
        )  # Blocking execution

        # Check if the command was successful
        if process.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to download model: {process.stderr.strip()}",
            )

        # Update download records after successful execution
        update_download_records(
            name=repo_name, status="untrained", type_=model_type, path=model_directory
        )

        # Return success message
        return {
            "message": f"Model {repo_name} downloaded successfully to {model_directory}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error executing huggingface-cli: {e}"
        )


@router.post("/download_model-1/")
async def convert_gguf_to_hf(request: GGUFConversionRequest):
    model_id = request.model_id
    filename = request.model_file_name

    try:
        # Load the tokenizer and model using the GGUF file
        tokenizer = AutoTokenizer.from_pretrained(model_id, gguf_file=filename)
        model = AutoModelForCausalLM.from_pretrained(model_id, gguf_file=filename)

        # Define the directory where the converted model will be saved
        conversion_directory = os.path.join(
            models_directory, model_id.replace("/", "--")
        )
        if not os.path.exists(conversion_directory):
            os.makedirs(conversion_directory)

        # Save the tokenizer and model in Hugging Face format
        tokenizer.save_pretrained(conversion_directory)
        model.save_pretrained(conversion_directory)

        # Explicitly delete the model and tokenizer to free up memory
        del model
        del tokenizer

        # If using a GPU, clear the cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Run garbage collection
        gc.collect()

        return {
            "message": "Model and tokenizer converted and saved successfully!",
            "directory": conversion_directory,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error converting GGUF to Hugging Face format: {e}"
        )


@router.post("/download_model/")
async def convert_gguf_to_hf(request: GGUFConversionRequest, fastapi_request: Request):
    model_id = request.model_id
    filename = request.model_file_name
    fastapi_request.app.state.is_downloading = True

    try:
        base_url = f"https://huggingface.co/{model_id}/resolve/main/{filename}"

        # Make the request to download the GGUF file
        response = requests.get(base_url)
        response.raise_for_status()  # Raise an error for bad responses

        # Define the directory where the GGUF file will be saved
        save_directory = os.path.join(models_directory, model_id.replace("/", "--"))
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Define the full path for the file to be saved
        file_path = os.path.join(save_directory, filename)

        # Save the GGUF file to the specified directory
        with open(file_path, "wb") as file:
            file.write(response.content)

        update_download_records(
            name=model_id.replace("/", "--"),
            status="untrained",
            type_="GGUF",
            path=file_path,
        )
        fastapi_request.app.state.is_downloading = False

        return {
            "message": "GGUF file downloaded and saved successfully!",
            "file_path": file_path,
        }

    except requests.exceptions.RequestException as e:
        fastapi_request.app.state.is_downloading = False
        raise HTTPException(
            status_code=500, detail=f"Error downloading the GGUF file: {e}"
        )


@router.get("/downloaded_model/{filename}")
async def download(filename: str):
    try:
        # Construct the full directory path
        directory_path = os.path.join(models_directory, filename.replace("/", "--"))

        # Check if the directory exists
        if not os.path.exists(directory_path) or not os.path.isdir(directory_path):
            raise HTTPException(status_code=404, detail="Directory not found")

        # Define the path where the zip file will be created in the models directory
        zip_file_path = os.path.join(models_directory, f"{filename}.zip")

        # Create the zip file
        shutil.make_archive(zip_file_path[:-4], "zip", directory_path)

        # Serve the zip file as an attachment
        return FileResponse(
            path=zip_file_path, filename=f"{filename}.zip", media_type="application/zip"
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error zipping directory: {str(e)}"
        )

    # Optionally, you can add a cleanup step to delete the zip file after it has been served.


data_file = os.path.join(
    "./", "model_data.json"
)  # Use os.path.join to handle file path


@router.get("/models_info")
async def get_models():
    try:
        with open(data_file, "r") as file:
            data = json.load(file)  # Load the JSON data from the file

        return data  # Return the data to the client
    except Exception as e:
        print(f"Error reading the file: {e}")
        return {"error": "Failed to load model data."}
