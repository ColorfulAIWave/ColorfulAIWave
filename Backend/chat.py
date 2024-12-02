from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import logging
from model_manager import ModelManager  # Import the ModelManager class
import torch
import os
import json
import uuid

router = APIRouter()

# Directory to store chat history
CHAT_DIR = "./chats"
os.makedirs(CHAT_DIR, exist_ok=True)

# Define chat history request/response models
class ChatRequest(BaseModel):
    chat_history: list  # The current chat history
    user_input: str     # The latest user message

class ChatSession(BaseModel):
    id: str
    messages: list
    model: str

# Define request models
class LoadModelRequest(BaseModel):
    model_name: str

# Define the data model for the chat request
class ChatRequest(BaseModel):
    chat_history: list  # This will store the conversation history
    user_input: str     # The new user input

@router.get("/list_chats")
async def list_chats():
    """
    List all chat sessions (JSON files) in the chats directory.
    Returns a list of chat session IDs.
    """
    try:
        chat_files = os.listdir(CHAT_DIR)
        chats = []
        for file_name in chat_files:
            if file_name.endswith(".json"):
                chat_id = file_name.replace(".json", "")
                # Load the chat to get additional details if needed
                with open(os.path.join(CHAT_DIR, file_name), "r") as f:
                    chat_data = json.load(f)
                    chats.append({
                        "id": chat_id,
                        "message_count": len(chat_data["messages"]),
                        "model": chat_data.get("model", "Unknown Model")
                    })
        return chats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list chat sessions: {str(e)}")


# Create a new chat session
@router.post("/chats")
async def create_chat():
    """
    Create a new chat session and return its ID.
    """
    try:
        chat_id = str(uuid.uuid4())  # Generate a unique chat ID
        chat_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        
        # Create a new chat with an empty history
        chat_data = {"id": chat_id, "messages": []}
        with open(chat_path, "w") as f:
            json.dump(chat_data, f)

        return chat_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create a new chat: {str(e)}")

# Get the chat history by chat_id
@router.get("/chats/{chat_id}")
async def get_chat(chat_id: str):
    """
    Retrieve a chat session by its ID.
    """
    try:
        chat_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        
        if not os.path.exists(chat_path):
            raise HTTPException(status_code=404, detail="Chat not found")

        with open(chat_path, "r") as f:
            chat_data = json.load(f)
        
        return chat_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat: {str(e)}")

# Save the updated chat history by chat_id
@router.put("/chats/{chat_id}")
async def save_chat(chat_id: str, chat_session: ChatSession):
    """
    Save or update a chat session's history.
    """
    try:
        chat_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        
        with open(chat_path, "w") as f:
            json.dump(chat_session.dict(), f)

        return {"status": "success", "chat_id": chat_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save chat: {str(e)}")

# Route to load a model
@router.post("/load_model/")
async def load_model(request: LoadModelRequest):
    """
    Load the specified model into memory. This only needs to be called once per model.
    """
    try:
        manager = ModelManager.get_instance()
        manager.load_model(request.model_name)
        return {"status": "success", "message": f"Model {request.model_name} loaded successfully."}
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

# Chat Route
@router.post("/chat/")
async def chat(request: ChatRequest):
    """
    Generate a response from the loaded model based on the user's input and chat history.
    """
    try:
        # Get the model and tokenizer from ModelManager
        manager = ModelManager.get_instance()
        if manager.model is None or manager.tokenizer is None:
            raise HTTPException(status_code=400, detail="Model not loaded. Please load a model first.")

        # Append the new user input to the chat history
        chat_history = request.chat_history
        chat_history.append({"role": "user", "content": request.user_input})

        # Prepare the prompt using the chat template
        prompt = manager.tokenizer.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)

        # Encode the prompt
        token_ids = manager.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        # Perform model inference
        with torch.no_grad():
            output_ids = manager.model.generate(
                token_ids.to(manager.model.device),
                do_sample=True,
                temperature=0.6,
                max_new_tokens=4096,
            )

        # Decode the model output to get the response text
        output = manager.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)

        # Append the model response to the chat history
        chat_history.append({"role": "assistant", "content": output})

        # Return the model response and updated chat history to the client
        return {
            "response": output,
            "chat_history": chat_history  # Return the updated chat history for the next turn
        }

    except Exception as e:
        logging.error(f"Error during chat inference: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate response: {str(e)}")
    


# Delete a chat by chat_id
@router.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str):
    """
    Delete a chat session by its ID.
    """
    try:
        chat_path = os.path.join(CHAT_DIR, f"{chat_id}.json")
        
        if not os.path.exists(chat_path):
            raise HTTPException(status_code=404, detail="Chat not found")
        
        # Delete the chat file
        os.remove(chat_path)
        
        return {"status": "success", "message": f"Chat {chat_id} deleted successfully."}
    except Exception as e:
        logging.error(f"Error deleting chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete chat: {str(e)}")



# Rename a chat file by chat_id
@router.put("/chats/{old_chat_id}/rename/{new_chat_id}")
async def rename_chat(old_chat_id: str, new_chat_id: str):
    """
    Rename a chat session file from old_chat_id to new_chat_id.
    """
    try:
        old_chat_path = os.path.join(CHAT_DIR, f"{old_chat_id}.json")
        new_chat_path = os.path.join(CHAT_DIR, f"{new_chat_id}.json")

        if not os.path.exists(old_chat_path):
            raise HTTPException(status_code=404, detail="Chat not found")

        if os.path.exists(new_chat_path):
            raise HTTPException(status_code=400, detail="New chat ID already exists")

        # Rename the chat file
        os.rename(old_chat_path, new_chat_path)

        return {
            "status": "success",
            "message": f"Chat {old_chat_id} renamed to {new_chat_id} successfully."
        }
    except Exception as e:
        logging.error(f"Error renaming chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to rename chat: {str(e)}")

