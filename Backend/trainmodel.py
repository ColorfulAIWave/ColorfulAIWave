from fastapi import APIRouter, HTTPException, Request
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, TrainerCallback
from sse_starlette.sse import EventSourceResponse
from datasets import load_from_disk
from pydantic import BaseModel
from peft import LoraConfig, PeftConfig, PeftModel
from trl import SFTTrainer
import torch
import os
from optimum.gptq import GPTQQuantizer
import shutil
import gc
import logging
import traceback
import asyncio
import json

# Initialize logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

router = APIRouter()

models_directory = "./models"

# Define the static directory to save the datasets
SAVE_DIR = "./local_datasets"

records_file_path = os.path.join(models_directory, "download_records.json")

progress_queue = asyncio.Queue()

@router.get("/training_progress")
async def training_progress():
    async def event_generator():
        while True:
            progress = await progress_queue.get()
            yield {"data": progress}

    return EventSourceResponse(event_generator())

class ProgressCallback(TrainerCallback):
    def __init__(self, progress_queue):
        super().__init__()
        self.progress_queue = progress_queue
        self.logs = []  # List to accumulate all logs

    def on_step_end(self, args, state, control, **kwargs):
        try:
            # Try to get the loss value; default to 'N/A' if not available
            loss = state.log_history[-1]['loss'] if 'loss' in state.log_history[-1] else 'N/A'
        except IndexError:
            loss = 'N/A'
        
        # Construct the progress message
        progress_message = {
            "step": state.global_step,
            "max_steps": state.max_steps,
            "loss": loss,
            "state": state
        }
        
        # Accumulate the log for later retrieval
        self.logs.append(progress_message)

        # Send the progress message to the queue for real-time update
        asyncio.run(self.progress_queue.put(progress_message))

        return control

    def on_train_end(self, args, state, control, **kwargs):
        # Called at the end of training, send a "Training complete!" message
        asyncio.run(self.progress_queue.put({"message": "Training complete!", "logs": self.logs}))

class ModelLoadRequest(BaseModel):
    model_name: str
    dataset_path: str
    type: str = "huggingface"  # Default type is "huggingface", can also be "gguf"
    production_training: bool = False  # Flag to indicate production training

    # New parameters with default values
    torch_dtype: str = "bfloat16"
    bf16: bool = False
    learning_rate: float = 5.0e-06
    num_train_epochs: int = 1
    attn_implementation: str = "eager"
    per_device_eval_batch_size: int = 2
    per_device_train_batch_size: int = 2
    save_steps: int = 200
    save_total_limit: int = 3
    seed: int = 0
    gradient_checkpointing: bool = True
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    gguf_file: str = None  # Only used if type is "gguf"
    bits: int = 8
    dataset_id: str = "wikitext2"
    quant_method: str = "gptq"
    training: bool = False
    finetuning: bool = False
    quantization: bool = False
    onnx: bool = False

class QuantizeRequest(BaseModel):
    model_path: str
    bits: int = 8
    dataset_id: str = "wikitext2"
    quant_method: str = "gptq"

def update_download_records(name: str, type_: str, path: str):
    # with lock:  # Ensure thread-safe access to the file
    with open(records_file_path, "r+") as file:
        # Load existing records
        records = json.load(file)
        
        # Add new record
        records.append({"name": name, "type": type_, "path": path})
        
        # Write updated records back to the file
        file.seek(0)
        json.dump(records, file, indent=4)
        file.truncate()  # Remove any leftover content

@router.post("/quantize")
def quantize_model(request: QuantizeRequest):
    try:
        # Load the tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(request.model_path, trust_remote_code=True)
        plan_model = AutoModelForCausalLM.from_pretrained(
            request.model_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )

        # Initialize the quantizer with the current bit setting
        quantizer = GPTQQuantizer(bits=request.bits, dataset=request.dataset_id, model_seqlen=2048)
        quantizer.quant_method = request.quant_method

        # Quantize the model
        gptq_model = quantizer.quantize_model(plan_model, tokenizer)

        # Create a directory path for saving the quantized model
        save_dir = f'./Quantized/{request.model_path.replace("/", "_")}_{request.bits}bit'
        os.makedirs(save_dir, exist_ok=True)

        # Save the quantized model and tokenizer
        gptq_model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)

        return {"status": "success", "message": f"Quantized model saved in {save_dir}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train_model")
def load_model_and_finetune(request: ModelLoadRequest):
    try:
        model_path = os.path.join(models_directory, request.model_name)
        dataset_path = os.path.join(SAVE_DIR, request.dataset_path)
        # Determine the dtype
        dtype = getattr(torch, request.torch_dtype)

        # Load the model based on the type
        if request.type == "gguf":
            if not request.gguf_file:
                raise HTTPException(status_code=400, detail="gguf_file must be provided when type is 'gguf'.")

            tokenizer = AutoTokenizer.from_pretrained(model_path, gguf_file=request.gguf_file)
            model = AutoModelForCausalLM.from_pretrained(model_path, gguf_file=request.gguf_file)

        elif request.type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_kwargs = {
                "use_cache": False,
                "trust_remote_code": True,
                "attn_implementation": request.attn_implementation,
                "torch_dtype": dtype,
                "device_map": 'auto'
            }
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported type '{request.type}'.")

        # Load the dataset
        processed_dataset = load_from_disk(dataset_path)

        # Add this to your dataset processing step in FastAPI
        def apply_chat_template(example, tokenizer):
            messages = example["messages"]
            # Add an empty system message if there is none
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "user", "content": ""})
            example["text"] = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            return example

        # Split the dataset into train and test sets
        split_dataset = processed_dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

        # Get the column names for removal later
        column_names = list(train_dataset.features)

        # Apply chat template to both train and test datasets
        processed_train_dataset = train_dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=10,
            remove_columns=column_names,  # Remove the original columns after transformation
            desc="Applying chat template to train_sft",
        )

        processed_test_dataset = test_dataset.map(
            apply_chat_template,
            fn_kwargs={"tokenizer": tokenizer},
            num_proc=10,
            remove_columns=column_names,  # Remove the original columns after transformation
            desc="Applying chat template to test_sft",
        )

        logger.info(f"Train dataset size: {len(processed_train_dataset)}")
        logger.info(f"Test dataset size: {len(processed_test_dataset)}")

        # Training configuration
        training_config = {
            "bf16": request.bf16,
            "do_eval": False,
            "learning_rate": request.learning_rate,
            "log_level": "info",
            "logging_steps": 50,
            "logging_strategy": "steps",
            "lr_scheduler_type": "cosine",
            "num_train_epochs": request.num_train_epochs,
            "max_steps": -1,
            "output_dir": "./Complete-JPTraining",
            "overwrite_output_dir": True,
            "per_device_eval_batch_size": request.per_device_eval_batch_size,
            "per_device_train_batch_size": request.per_device_train_batch_size,
            "remove_unused_columns": True,
            "save_steps": request.save_steps,
            "save_total_limit": request.save_total_limit,
            "seed": request.seed,
            "gradient_checkpointing": request.gradient_checkpointing,
            "gradient_checkpointing_kwargs":{"use_reentrant": False},
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.2,
        }
        peft_config = {
            "r": request.r,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": request.target_modules,
            "modules_to_save": None,
        }

        train_conf = TrainingArguments(**training_config)
        peft_conf = LoraConfig(**peft_config)

        trainer = SFTTrainer(
            model=model,
            args=train_conf,
            peft_config=peft_conf,
            train_dataset=processed_train_dataset,
            eval_dataset=processed_test_dataset,
            max_seq_length=2048,
            dataset_text_field="text",  # The field now holds the chat template output
            tokenizer=tokenizer,
            callbacks=[ProgressCallback(progress_queue)],
            packing=True
        )

        # Start the training process
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        trained_model_path = 'models/jp-smolLM-trained-v1'
        trainer.save_model(trained_model_path)

        if request.production_training:
            # Clear memory
            del trainer
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Reload the original model and the adapter
            tokenizer = AutoTokenizer.from_pretrained(request.model_name)
            original_model = AutoModelForCausalLM.from_pretrained(request.model_name, **model_kwargs)
            adapter_model = PeftModel.from_pretrained(original_model, trained_model_path)

            # Merge the adapter with the original model
            adapter_model.save_pretrained('./merged_models/japaneseV1')
            tokenizer.save_pretrained('./merged_models/japaneseV1')

            # Convert the merged model to ONNX format
            os.system('python -m onnxruntime_genai.models.builder -i "./merged_models/japaneseV1" -o "./ONNX/japaneseV1_directML" -p int4 -e cpu')

            # Clean up
            shutil.rmtree(trained_model_path)  # Delete the adapter model directory
            shutil.rmtree('./merged_models/japaneseV1')  # Optionally delete the merged model directory

        del trainer
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return {"status": "success", "message": "Model finetuned successfully", "metrics": metrics}

    except Exception as e:
        logger.error(f"Error during model fine-tuning: {str(e)}")
        logger.error(traceback.format_exc())  # Log the full stack trace
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/process_model")
def complete_training(request: ModelLoadRequest, fastapi_request: Request):
    
    try:
        fastapi_request.app.state.is_training = True
        model_path = os.path.join(models_directory, request.model_name)
        dataset_path = os.path.join(SAVE_DIR, request.dataset_path)
        if request.finetuning or request.training:
            # Define the model and dataset paths from the request
            

            # Check if the paths exist
            if not os.path.exists(model_path):
                raise HTTPException(status_code=404, detail=f"Model path {model_path} does not exist.")
            if not os.path.exists(dataset_path):
                raise HTTPException(status_code=404, detail=f"Dataset path {dataset_path} does not exist.")

            # Load the tokenizer and model
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model_kwargs = dict(
                use_cache=False,
                trust_remote_code=True,
                attn_implementation=request.attn_implementation,  # From request
                torch_dtype=torch.bfloat16,
                device_map='auto'
            )
            model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

            # Check if the model is loaded on the 'meta' device and move it to 'cuda' or 'cpu'
            if model.device == torch.device('meta'):
                logging.info("Model is on the 'meta' device, moving to 'cuda'.")
                model.to_empty(device=torch.device('cuda'))  # Move model to GPU

            # Load the dataset
            processed_dataset = load_from_disk(dataset_path)

            # Dataset processing step (template application)
            def apply_chat_template(example, tokenizer):
                messages = example["messages"]
                if messages[0]["role"] != "system":
                    messages.insert(0, {"role": "user", "content": ""})
                example["text"] = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False)
                return example

            # Split the dataset into training and test sets
            split_dataset = processed_dataset.train_test_split(test_size=0.2)
            train_dataset = split_dataset['train']
            test_dataset = split_dataset['test']

            # Remove columns
            column_names = list(train_dataset.features)

            # Apply chat template to both train and test datasets
            processed_train_dataset = train_dataset.map(
                apply_chat_template,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=10,
                remove_columns=column_names,
                desc="Applying chat template to train dataset",
            )

            processed_test_dataset = test_dataset.map(
                apply_chat_template,
                fn_kwargs={"tokenizer": tokenizer},
                num_proc=10,
                remove_columns=column_names,
                desc="Applying chat template to test dataset",
            )

            # Define training and LoRA configurations
            training_config = {
                "bf16": request.bf16,
                "do_eval": False,
                "learning_rate": request.learning_rate,
                "log_level": "info",
                "logging_steps": 50,
                "logging_strategy": "steps",
                "lr_scheduler_type": "cosine",
                "num_train_epochs": request.num_train_epochs,
                "max_steps": -1,
                "output_dir": "./Complete-JPTraining",
                "overwrite_output_dir": True,
                "per_device_eval_batch_size": request.per_device_eval_batch_size,
                "per_device_train_batch_size": request.per_device_train_batch_size,
                "remove_unused_columns": True,
                "save_steps": request.save_steps,
                "save_total_limit": request.save_total_limit,
                "seed": request.seed,
                "gradient_checkpointing": request.gradient_checkpointing,
                "gradient_accumulation_steps": 1,
                "warmup_ratio": 0.2,
            }

            peft_config = {
                "r": request.r,
                "lora_alpha": request.lora_alpha,
                "lora_dropout": request.lora_dropout,
                "bias": "none",
                "task_type": "CAUSAL_LM",
                "target_modules": request.target_modules,
                "modules_to_save": None,
            }

            # Initialize the trainer with SFTTrainer
            train_conf = TrainingArguments(**training_config)
            peft_conf = LoraConfig(**peft_config)

            trainer = SFTTrainer(
                model=model,
                args=train_conf,
                peft_config=peft_conf,
                train_dataset=processed_train_dataset,
                eval_dataset=processed_test_dataset,
                max_seq_length=2048,
                dataset_text_field="text",
                tokenizer=tokenizer,
                packing=True
            )

        if request.finetuning:
            # Start the training process
            train_result = trainer.train()
            finetune_path = model_path+"_finetuned_Lora_1"
            finetune_name = request.model_name+"_finetuned_Lora_1"
            trainer.save_model(finetune_path)
            update_download_records(name=finetune_name, type_="Lora", path=finetune_path)

        if request.training:
            train_result = trainer.train()
            finetune_path = model_path+"_finetuned_Lora_1"
            finetune_name = request.model_name+"_finetuned_Lora_1"
            trainer.save_model(finetune_path)
            adapter_id = finetune_path
            config = PeftConfig.from_pretrained(adapter_id)
            quantized_model = PeftModel.from_pretrained(model, adapter_id)
            quantized_model = quantized_model.merge_and_unload()
            trained_path = model_path+"_trained_model_1"
            trained_name = request.model_name+"_trained_model_1"
            quantized_model.save_pretrained(trained_path)
            tokenizer.save_pretrained(trained_path)
            update_download_records(name=trained_name, type_="Huggingface", path=trained_path)

        if request.quantization:
            # Load the tokenizer and model
            try:
                quantized_model
            except NameError:
                quantized_model = None
            if quantized_model is None:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model_kwargs = dict(
                    use_cache=False,
                    trust_remote_code=True,
                    attn_implementation=request.attn_implementation,  # From request
                    torch_dtype=torch.bfloat16,
                    device_map='auto'
                )
                quantized_model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)

                # Dynamically get quantization bits, dataset, and method from request
                quantizer = GPTQQuantizer(
                    bits=request.bits,  # Bits from the request
                    dataset=request.dataset_id,  # Dataset from the request
                    model_seqlen=2048
                )
                quantizer.quant_method = request.quant_method  # Quantization method

                # Quantize the model
                gptq_model = quantizer.quantize_model(quantized_model, tokenizer)
            else:
                # Dynamically get quantization bits, dataset, and method from request
                quantizer = GPTQQuantizer(
                    bits=request.bits,  # Bits from the request
                    dataset=request.dataset_id,  # Dataset from the request
                    model_seqlen=2048
                )
                quantizer.quant_method = request.quant_method  # Quantization method

                # Quantize the model
                gptq_model = quantizer.quantize_model(quantized_model, tokenizer)

            # Save the quantized model
            gptq_save_path = model_path+f"_quantized_model_{request.bits}_bits"
            gptq_name = request.model_name+f"_quantized_model_{request.bits}_bits"
            gptq_model.save_pretrained(gptq_save_path)
            tokenizer.save_pretrained(gptq_save_path)
            update_download_records(name=gptq_name, type_="Quantized", path=gptq_save_path)

        if request.onnx:
            # Convert to ONNX if necessary
            onnx_path = model_path+"_onnx"
            onnx_name = request.model_name+"_onnx"
            if not request.training:
                trained_path = model_path
            os.system(f'python -m onnxruntime_genai.models.builder -m {trained_path} -o {onnx_path} -p int4 -e cpu')
            update_download_records(name=onnx_name, type_="ONNX", path=onnx_path )
        fastapi_request.app.state.is_training = False

        return {"status": "success", "message": "Training and quantization complete"}

    except Exception as e:
        fastapi_request.app.state.is_training = False
        logger.error(f"Error in complete training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))




@router.post("/finetune")
def load_model_and_finetune(request: ModelLoadRequest):
    try:
        # Determine the dtype
        dtype = getattr(torch, request.torch_dtype)

        # Load the model based on the type
        if request.type == "gguf":
            if not request.gguf_file:
                raise HTTPException(status_code=400, detail="gguf_file must be provided when type is 'gguf'.")

            tokenizer = AutoTokenizer.from_pretrained(request.model_name, gguf_file=request.gguf_file)
            model = AutoModelForCausalLM.from_pretrained(request.model_name, gguf_file=request.gguf_file)

        elif request.type == "huggingface":
            tokenizer = AutoTokenizer.from_pretrained(request.model_name)
            model_kwargs = {
                "use_cache": False,
                "trust_remote_code": True,
                "attn_implementation": request.attn_implementation,
                "torch_dtype": dtype,
                "device_map": 'cuda'
            }
            model = AutoModelForCausalLM.from_pretrained(request.model_name, **model_kwargs)
        
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported type '{request.type}'.")

        # Load the dataset
        raw_dataset = load_from_disk(request.dataset_path)

        # Optionally append a system message at the beginning of each conversation row
        def append_system_message(example):
            messages = example["messages"]
            if messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": "System message placeholder"})
            example["messages"] = messages
            return example

        # Check if the dataset has a 'train' split
        if 'train' in raw_dataset:
            processed_dataset = raw_dataset['train'].map(
                append_system_message,
                num_proc=10,
                desc="Appending system message to conversations (train)"
            )
        else:
            # If there is no 'train' split, apply to the entire dataset
            processed_dataset = raw_dataset.map(
                append_system_message,
                num_proc=10,
                desc="Appending system message to conversations (full dataset)"
            )

        split_dataset = processed_dataset.train_test_split(test_size=0.2)
        train_dataset = split_dataset['train']
        test_dataset = split_dataset['test']

        # Training configuration
        training_config = {
            "bf16": request.bf16,
            "do_eval": False,
            "learning_rate": request.learning_rate,
            "log_level": "info",
            "logging_steps": 50,
            "logging_strategy": "steps",
            "lr_scheduler_type": "cosine",
            "num_train_epochs": request.num_train_epochs,
            "max_steps": -1,
            "output_dir": "./Complete-JPTraining",
            "overwrite_output_dir": True,
            "per_device_eval_batch_size": request.per_device_eval_batch_size,
            "per_device_train_batch_size": request.per_device_train_batch_size,
            "remove_unused_columns": True,
            "save_steps": request.save_steps,
            "save_total_limit": request.save_total_limit,
            "seed": request.seed,
            "gradient_checkpointing": request.gradient_checkpointing,
            "gradient_checkpointing_kwargs":{"use_reentrant": False},
            "gradient_accumulation_steps": 1,
            "warmup_ratio": 0.2,
        }
        peft_config = {
            "r": request.r,
            "lora_alpha": request.lora_alpha,
            "lora_dropout": request.lora_dropout,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": request.target_modules,
            "modules_to_save": None,
        }

        train_conf = TrainingArguments(**training_config)
        peft_conf = LoraConfig(**peft_config)

        trainer = SFTTrainer(
            model=model,
            args=train_conf,
            peft_config=peft_conf,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            max_seq_length=2048,
            dataset_text_field="messages",
            tokenizer=tokenizer,
            # callbacks=[ProgressCallback(progress_queue)],
            packing=True
        )

        # Start the training process
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        trained_model_path = 'models/jp-smolLM-trained-v1'
        trainer.save_model(trained_model_path)

        if request.production_training:
            # Clear memory
            del trainer
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()

            # Reload the original model and the adapter
            tokenizer = AutoTokenizer.from_pretrained(request.model_name)
            original_model = AutoModelForCausalLM.from_pretrained(request.model_name, **model_kwargs)
            adapter_model = PeftModel.from_pretrained(original_model, trained_model_path)

            # Merge the adapter with the original model
            adapter_model.save_pretrained('./merged_models/japaneseV1')
            tokenizer.save_pretrained('./merged_models/japaneseV1')

            # Convert the merged model to ONNX format
            os.system('python -m onnxruntime_genai.models.builder -i "./merged_models/japaneseV1" -o "./ONNX/japaneseV1_directML" -p int4 -e cpu')

            # Clean up
            shutil.rmtree(trained_model_path)  # Delete the adapter model directory
            shutil.rmtree('./merged_models/japaneseV1')  # Optionally delete the merged model directory

        del trainer
        del model
        del tokenizer
        torch.cuda.empty_cache()
        gc.collect()

        return {"status": "success", "message": "Model finetuned successfully", "metrics": metrics}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
