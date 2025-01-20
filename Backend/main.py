from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modeldownload import router as modeldownload_router
from datasetdownload import router as datasetdownload_router
from trainmodel import router as trainmodel_router  # Import the new router
from chat import router as chat_router  # Import the new router
from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize global state variables here
    app.state.is_downloading = False
    app.state.is_training = False

    # Before app startup
    yield

    # After app shutdown (you can put any cleanup code here if necessary)


app = FastAPI(lifespan=lifespan)


# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)


# Include the router for the model download route
app.include_router(modeldownload_router)

# Include the router for the dataset download route
app.include_router(datasetdownload_router)

# Include the router for the model training route
app.include_router(trainmodel_router)
app.include_router(chat_router)  # Include the chat router


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/status")
async def get_status():
    return {
        "is_downloading": app.state.is_downloading,
        "is_training": app.state.is_training,
    }
