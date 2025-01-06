# Installation 🛠️

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

Install Node.js (version 22.\* LTS version from https://nodejs.org/en) and also all the dependencies required for the project using following command:

```
npm install
npm install -g serve
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

Activate the virtual environment (MAC):

For Mac/Linux:

```
source venv/bin/activate
```

Activate the virtual environment (Windows):

```
venv\Scripts\activate
```

Install backend dependencies:

```
pip install -r requirements.txt
```

The backend requirements are now installed! 🎉

4. **Running the Application** ▶️

4(A) **_Start the Backend server:_**

```
uvicorn main:app --reload
```

4(B) **_Start the Frontend server:_**

```
serve -s build
```

## Contact Us 📞

For any issues, queries, or suggestions, feel free to reach out:

📧 Email: wave@aiglow.ai
🌐 Website: [COLORFUL](https://colorful-inc.jp/)
🐞 GitHub Issues: [Report an Issue](https://github.com/ColorfulAIWave/TideAI/issues)
