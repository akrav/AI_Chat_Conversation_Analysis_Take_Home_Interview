### Architecture Document: AI Conversation Analytics

**Version:** 1.0
\<br\> **Date:** August 28, 2025
\<br\> **Author:** Adam Kravitz

-----

### 1\. System Overview

This document outlines the technical architecture for the AI Conversation Analytics project. It is a data science-focused system, not a web application, and is designed to transform raw conversational data into actionable insights through a reproducible, script-based pipeline. The system's core is a hybrid analytical approach, combining an unsupervised topic modeling technique (**BERTopic**) with the nuanced capabilities of a large language model (**LLM**).

The architecture is structured to support a complete data science lifecycle, from data acquisition to the final delivery of insights. The primary outputs are not a deployed service, but a set of well-documented files that encapsulate the entire analytical process.

-----

### 2\. Data Ingestion & Storage

  * **Source:** The primary data source is the `allenai/WildChat` dataset, publicly available on the Hugging Face Hub.
  * **Ingestion Method:** The dataset will be programmatically loaded using the `huggingface/datasets` library within a Python environment. For efficiency and to manage the large size of the dataset, a subsample may be used as per the project requirements.
  * **Local Storage:** The raw and intermediate processed data will be stored locally within a structured project directory. Following best practices for reproducibility, the raw data will not be modified directly.

-----

### 3\. Processing & Analysis Pipeline

This section details the technical workflow, following the CRISP-DM methodology outlined in the Product Requirements Document.

#### 3.1. Data Exploration and Preparation

  * **Tool:** Python with core libraries such as Pandas and NumPy will be used for data loading, cleaning, and exploratory data analysis (EDA).
  * **Process:**
    1.  The `WildChat` dataset is loaded into a Pandas DataFrame.
    2.  Initial EDA is performed to understand the data's structure, identify missing values, and analyze metadata like `language`, `model`, and `timestamp`.
    3.  A dedicated `Data Exploration File` (e.g., a Jupyter Notebook or R Markdown) will be created to document the findings and the cleaning process.

#### 3.2. Topic Modeling (BERTopic)

  * **Tool:** The BERTopic library will be used for unsupervised topic modeling.
  * **Process:**
    1.  Conversation text data will be preprocessed (e.g., cleaning, removing special characters) to prepare it for embedding.
    2.  BERTopic leverages pre-trained transformer models (like those from the `huggingface/transformers` library) to create contextual embeddings of the text.
    3.  These embeddings are then clustered, and representative words for each cluster are extracted using c-TF-IDF to define the topics.
    4.  The output will be a set of identified topics, their frequencies, and a list of conversations associated with each topic.

#### 3.3. Fine-Grained Analysis (LLMs)

  * **Tool:** A Large Language Model (LLM) will be accessed via an OpenAI API key.
  * **Process:**
    1.  A subset of conversations from the topics identified by BERTopic will be selected for deeper analysis.
    2.  Prompts will be carefully engineered to perform specific, fine-grained tasks such as sentiment analysis, intent classification, or entity extraction. This is a zero-shot learning approach, as no pre-labeled training data is provided.
    3.  The LLM's API will be called programmatically to process the conversation snippets.
    4.  The output from the LLM (e.g., sentiment scores, intent labels) will be collected and appended back to the dataset for further analysis.
    5.  A Python script or Jupyter Notebook will be used to manage API calls, rate limits, and error handling.

-----

### 4\. Project Structure & Deliverables

The project will be organized to ensure reproducibility and clarity for stakeholders. The final output is not a deployed service but a collection of files within a comprehensive repository.

#### 4.1. Repository Structure

The project will adhere to a standard, reproducible data science project structure, such as the one recommended by the cookiecutter-data-science template.

```
/project_root
├── data/
│   ├── 01_raw/        # untouched raw data (e.g., wildchat.json)
│   ├── 02_interim/    # intermediate data (e.g., preprocessed text)
│   └── 03_processed/  # final analytical datasets
├── notebooks/         # All Jupyter Notebooks for exploration and analysis
├── src/               # Reusable code, functions, and scripts
├── tests/             # Testing suite for key functions
├── reports/
│   └── final_insights.md # Final markdown report for stakeholders
├── Build Documentation/ # Documentation to assist the AI Coder
│   ├── Troubleshooting.md
│   ├── structure.md
│   └── Sprint-Progress.md
└── README.md
```

#### 4.2. Deliverables

  * **Product Requirements Document (PRD):** A markdown file (already created) that serves as the guiding document.
  * **Data Exploration File:** A Jupyter Notebook located in the `notebooks/` directory that documents the data loading, cleaning, and initial insights.
  * **Task Files:** Jupyter Notebooks or Python scripts in the `notebooks/` and `src/` directories that contain the code for the BERTopic and LLM analysis pipelines.
  * **Final Insights Document:** A markdown file (`reports/final_insights.md`) that summarizes the key findings, visualizations, and actionable insights for brand strategists.

-----

### 5\. Technical Stack

  * **Programming Language:** Python 3.9+
  * **Core Libraries:**
      * **Pandas:** For data manipulation and analysis.
      * **NumPy:** For numerical operations.
  * **NLP & Modeling Libraries:**
      * **Hugging Face `datasets`:** For efficient data loading from the Hugging Face Hub.
      * **BERTopic:** For unsupervised topic modeling.
      * **Hugging Face `transformers`:** For generating text embeddings for BERTopic.
  * **LLM Integration:**
      * **`openai` library:** For programmatic access to the OpenAI API.
      * **`requests` library:** For API calls and error handling.
  * **Reproducibility & Version Control:**
      * **Git:** For version control of all code and documents.
      * **Virtual Environments (e.g., `venv`, `conda`):** To manage project dependencies and ensure reproducibility.

-----

### 6\. Testing & Documentation

#### 6.1. Testing Suite

A testing suite will be created within the `/tests` folder to ensure the reliability and integrity of the key functions in the `src/` directory. This includes:

  * **Unit Tests:** To test individual functions for data cleaning and preprocessing.
  * **Integration Tests:** To verify that different parts of the pipeline (e.g., data loading and topic modeling) work together as expected.

#### 6.2. Build Documentation

A dedicated `Build Documentation` folder will be established to help the AI Coder manage the project. This folder will contain key documents for project management and troubleshooting, including:

  * `Troubleshooting.md`: To log and document solutions for common issues and errors encountered during development.
  * `structure.md`: To provide a clear overview of the project's file and folder structure.
  * `Sprint-Progress.md`: To track progress on sprints, tasks, and project milestones.