<p>
  <img src="screenshots/CLF-logo_main_black.png" alt="Colorful Logo" width="200">
</p>

At **[COLORFUL](https://colorful-inc.jp/)**, we specialize in cutting-edge AI and generative technology solutions. Our mission is to empower developers and businesses by providing intuitive tools for managing and deploying large language models. With **TideAI**, we aim to simplify complex AI workflows, bringing innovation and ease to every user.

---

## TideAI

**Introducing Tide: Your All-in-One Local LLM Solution**  
Tide, a product proudly developed by **[COLORFUL](https://colorful-inc.jp/)**, is an innovative web-based platform designed to simplify managing, fine-tuning, and deploying large language models (LLMs). Whether you are an experienced developer or new to generative AI, Tide empowers you to harness the full potential of LLMs effortlessly and securely.

---

## Key Features 🚀

1. **Simplified Model Downloads** 📥  
   Access the latest LLMs, including `.gguf`, Hugging Face, and ONNX versions, with just a few clicks. No complex configurations—download and start exploring.

2. **Intuitive Dataset Management** 📂  
   Organize and manage your prompt data with ease. Tailor datasets for model training and fine-tuning while ensuring complete security of your sensitive information. Create personalized bots without compromising your data privacy.

3. **Streamlined Model Operations** ⚙️  
   Train, fine-tune, and quantize LLMs with a straightforward interface designed for efficiency. Tide’s user-friendly tools eliminate the technical barriers to working with advanced AI models.

4. **Interactive Chatbot Testing** 💬  
   Experience and test your trained models with Tide’s built-in chatbot feature. See your personalized bot in action and refine its performance with real-time feedback.

---

## Why Choose Tide? 🌟

Tide is engineered to run LLMs locally, ensuring hassle-free operations without the need for extensive technical expertise. Its intuitive design makes advanced generative AI capabilities accessible to everyone, fostering innovation and creativity at every level.  
Step into the future of AI with Tide—where simplicity meets power.

---

## Requirements 🛠️

To get started with TideAI, ensure the following are installed:

- [Python](https://www.python.org/) (Version 3.8 or above) 🐍
- [Node.js](https://nodejs.org/en) (Version 16 or above) 🌐

---

## Account Requirements 🔐

- A GitHub account is required for code collaboration and management.

---

## Installation Guide 📖

You can set up **TideAI** in two ways:

---

### 1. Via Docker 🐳

Follow these steps to set up TideAI using Docker:

1. **Clone the Repository** 📂

```
   git clone https://github.com/ColorfulAIWave/TideAI.git
   cd TideAI
```

2. **Install Docker**
   Download and install Docker from the official Docker website.

3. **Enable Docker**
   Ensure Docker is running on your system.

4. **Run TideAI with Docker Compose**
   Open a terminal, navigate to the TideAI folder, and run:
   ```
   docker-compose up
   ```

---

### 2. Manual Installation 🛠️

Follow these steps for manual installation:

1. **Clone the Repository** 📂

```
git clone https://github.com/ColorfulAIWave/TideAI.git
cd TideAI
```

2. **Frontend Installation** 🌐

Navigate to the frontend directory:

```
cd Frontend/client
```

Install Node.js dependencies:

```
npm instal
```

The frontend requirements are now installed! 🎉

3. **Backend Installation** ⚙️

Go back to the backend directory:

```
cd ../../Backend
```

Create a Python virtual environment:

```
python -m venv venv
```

Activate the virtual environment:

For Mac/Linux:

```
source venv/bin/activate
```

For Windows:

Open the Windows Command Prompt (required for virtual environment activation).
Navigate to the Tide directory:

```
cd PATH_TO_TIDE_FOLDER
```

Activate the virtual environment:

```
venv\Scripts\activate
```

Install backend dependencies:

```
pip install -r requirements.txt
```

Install PyTorch locally based on your system requirements:

```
PyTorch Installation Guide
```

(Optional) Install additional dependencies if needed:

```
pip install python-multipart
```

4. **Running the Application** ▶️

4(A) **_Start the Backend server:_**

```
uvicorn main:app --reload
```

4(B) **_Start the Frontend server:_**

```
cd Frontend/client
npm install -g serve
serve -s build
```

---

# Tide App Operational Manual

## Features Overview

### 1. Login to the Tide App

Start by logging into the application using your credentials. The authentication system ensures a secure and personalized experience for every user.

![Login Page](screenshots/login.png)

### 2. Dashboard

The dashboard serves as the central hub of the application, providing access to all major features. It is designed for ease of navigation and offers an overview of available options and functionalities.

![Dashboard](screenshots/dashboard.png)

### 3. Model Management (Upload/Download LLM Models)

The application enables users to upload or download different variations of LLM models locally. This feature supports flexibility in managing model versions and facilitates experimentation with various architectures.

![Model Management](screenshots/modelManagement.png)

### 4. Datasets

Easily integrate custom datasets for your models. This feature is particularly useful for training and fine-tuning models with domain-specific prompts, enhancing their performance in specialized tasks.

![Datasets](screenshots/datasets.png)

### 5. Model Operations

Perform advanced operations on models, including:

- **Training**: Develop models with new data.
- **Fine-tuning**: Adapt pre-trained models to specific tasks.
- **Quantization**: Optimize model size and speed without significant performance loss.

![Model Operations](screenshots/modelOperations.png)

### 6. Custom Chatbot Features

#### 6(a). Build a Custom Chatbot

Select a specific model from the library to create your own chatbot. Customize it with tailored datasets and configurations to suit your requirements.

![Build Chatbot](screenshots/chat1.png)

#### 6(b). Chatbot Interface

This feature provides a user-friendly interface for interacting with your custom chatbot. Test its functionality, refine its responses, and deploy it for real-world applications.

![Chatbot Interface](screenshots/chat2.png)

## Contact Us 📞

For any issues, queries, or suggestions, feel free to reach out:

📧 Email: wave@aiglow.ai
🌐 Website: [COLORFUL](https://colorful-inc.jp/)
🐞 GitHub Issues: [Report an Issue](https://github.com/ColorfulAIWave/TideAI/issues)
