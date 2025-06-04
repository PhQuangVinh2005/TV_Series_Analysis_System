# 🧠 AI/NLP TV Series Analysis System

This repository hosts a robust system designed for in-depth analysis of TV series using cutting-edge Artificial Intelligence (AI), Large Language Models (LLMs), and Natural Language Processing (NLP) techniques. From scraping raw data to building interactive AI applications, this project demonstrates a full pipeline for understanding and interacting with your favorite TV series content.

## ✨ Project Overview

The "AI/NLP TV Series Analysis System" allows users to delve deep into the narrative and character dynamics of a TV series. It starts by gathering vast amounts of data from the web, then processes this data through various AI and NLP models to extract themes, map character relationships, classify specific elements (like "Jutsu" types for Naruto), and even create an interactive chatbot that imitates series characters.

This project is built as a practical demonstration of how modern AI/NLP tools can be applied to rich, unstructured text data like TV series subtitles and descriptions.

## 🚀 Key Features

This system encompasses several distinct yet interconnected modules, each leveraging powerful AI/NLP libraries:

1.  ### **Web Scraping & Data Collection (Scrapy)**
    *   **Purpose:** Efficiently gathers and scrapes relevant data (e.g., episode information, character details, special abilities like "Jutsu") from various online sources.
    *   **Implementation:** Utilizes the `Scrapy` framework to define spiders for targeted data extraction, processing raw HTML into structured formats.
    *   **Output:** Populates the `data/` directory with structured information like `jutsus.jsonl` and `Naruto.csv`, including subtitle files.

2.  ### **Theme Classification (Hugging Face Transformers - Zero-Shot)**
    *   **Purpose:** Identifies the predominant themes within the series' content (e.g., "Friendship," "War," "Sacrifice") without requiring explicit, pre-labeled training data for these specific themes.
    *   **Implementation:** Leverages pre-trained Large Language Models (LLMs) from `Hugging Face Transformers` to perform zero-shot text classification, enabling flexible theme detection.
    *   **Output:** Generates insights into the thematic progression or presence across episodes, demonstrated via interactive visualizations.

3.  ### **Character Network Analysis (SpaCy, NetworkX, Pyviz)**
    *   **Purpose:** Visualizes the prominence of characters and the strength of their relationships within the series.
    *   **Implementation:**
        *   `SpaCy`'s Named Entity Recognition (NER) model is used to accurately extract character names from dialogue and narrative text.
        *   `NetworkX` constructs a graph representing characters as nodes and their interactions/co-occurrences as edges.
        *   `Pyviz` (or similar visualization libraries) renders an interactive network graph, allowing exploration of character centrality and relational dynamics.
    *   **Output:** An interactive HTML visualization (`Naruto.html`) showcasing the character network.

4.  ### **Custom Text Classification (Hugging Face Transformers - Fine-tuning)**
    *   **Purpose:** Trains a specialized text classifier on a custom dataset for a specific task, such as categorizing different types of "Jutsu" (combat techniques) in the Naruto series.
    *   **Implementation:** Demonstrates how to fine-tune state-of-the-art `Hugging Face Transformers` models on your own labeled dataset to achieve high-performance classification.
    *   **Output:** A trained model capable of classifying text into predefined custom categories.

5.  ### **Character Chatbot (Hugging Face Transformers - LLM Fine-tuning)**
    *   **Purpose:** Creates an interactive chatbot that can imitate the conversational style and persona of specific characters from the TV series.
    *   **Implementation:** Involves advanced LLM techniques, potentially including instruction tuning or fine-tuning pre-trained models on character-specific dialogue to generate coherent and in-character responses.
    *   **Output:** A conversational AI that allows users to "chat" with their favorite characters.

6.  ### **Interactive Demos (Gradio)**
    *   **Purpose:** Provides user-friendly web interfaces for interacting with the various AI models developed.
    *   **Implementation:** `Gradio` is used to quickly build shareable web demos for the theme classifier, character network visualization, and the character chatbot, making the project accessible without deep technical knowledge.

## 🛠️ Technologies Used

*   **Python:** The primary programming language.
*   **Scrapy:** For efficient web crawling and data extraction.
*   **Hugging Face Transformers:** Core library for Large Language Models, zero-shot classification, custom text classification, and chatbot development.
*   **SpaCy:** For high-performance Named Entity Recognition (NER) and other NLP tasks.
*   **NetworkX:** For creating, manipulating, and studying the structure, dynamics, and functions of complex networks.
*   **Pyviz (or similar like Plotly, Dash):** For interactive data visualization, especially network graphs.
*   **Gradio:** For rapid prototyping and deployment of interactive web demos for AI models.
*   **Pandas:** For powerful data manipulation and analysis.
*   **NumPy:** For fundamental numerical computing.
*   **PyTorch:** The underlying deep learning framework for Hugging Face models.
*   **TRL (Transformer Reinforcement Learning):** Potentially used for advanced fine-tuning techniques for the chatbot.
*   **`python-dotenv`:** For managing environment variables (e.g., API tokens).


## ⚙️ Setup and Installation

To set up and run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd TV_SERIES
    ```
    *(Replace `<repository_url>` with the actual URL of your Git repository.)*

2.  **Create a virtual environment (highly recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**
    *   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```

4.  **Install project dependencies:**
    *(It is recommended to create a `requirements.txt` file from the project's dependencies for easier installation. If not provided, install manually as follows:)*
    ```bash
    pip install scrapy transformers spacy networkx gradio pandas numpy torch sentencepiece accelerate bitsandbytes python-dotenv trl
    ```
    *   **Install SpaCy model:**
        ```bash
        python -m spacy download en_core_web_sm
        ```
    *   **For GPU support (if applicable):**
        Ensure you install the correct PyTorch version for your CUDA setup. For example, for CUDA 11.8:
        ```bash
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ```

5.  **Set up Hugging Face API Token (Optional, for model deployment/private access):**
    If you plan to push models to the Hugging Face Hub or access certain private models, create a `.env` file in the root `TV_SERIES` directory and add your token:
    ```
    HUGGING_FACE_TOKEN="hf_YOUR_ACTUAL_HUGGING_FACE_TOKEN"
    ```

## 🏃 Usage

Each module can be run independently, or you can use the central Gradio application for a unified experience.

### 1. Data Scraping

To collect data using the Scrapy crawler:
```bash
cd crawler
scrapy crawl jutsu_crawler # Assumes the spider name is 'jutsu_crawler' in jutsu_crawler.py
```

### 2. Running the Gradio Application

The primary way to interact with the various AI components is through the Gradio web interface.
```bash
python .gradio/app.py
```
After running, open your web browser and navigate to the local address provided by Gradio (typically http://127.0.0.1:7860). From there, you can access the theme classifier, character network visualization, and the character chatbot.

### 3. Individual Module Execution
You can also run specific scripts for individual modules

- Character Network Generation:
```bash
cd character_network
python named_entity_recognizer.py
python character_network_generator.py
# The output HTML will be generated, e.g., Naruto.html in the root.
```
- Custom Text Classifier Training:
```bash
cd text_classification
python custom_trainer.py # This will train the Jutsu classifier
```
Refer to jutsu_classifier_development.ipynb for detailed steps on preparing data and training.

- Character Chatbot (standalone):
```bash
cd character_chatbot
python character_chatbot.py # This might run a standalone chatbot interface or serve as an API for Gradio
```