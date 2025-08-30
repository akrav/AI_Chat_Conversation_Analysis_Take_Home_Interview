### Project Requirements Document: AI Conversation Analytics

**Version:** 1.0
\<br\> **Date:** August 28, 2025
\<br\> **Author:** Adam Kravitz

-----

### 1\. Introduction & Vision

#### 1.1. Vision

To provide a validated framework for conversational analytics by transforming unstructured conversational data into strategic intelligence for brands. This framework will enable brands to understand user interactions with AI systems and inform actionable strategies.

#### 1.2. Problem Statement

Currently, brands lack a clear, proven method for analyzing real-world AI conversations to derive meaningful insights. Uncontrolled, real-world user interactions with chatbots and virtual assistants represent a critical unknown, and there is a need to translate these conversational logs into a strategic intelligence asset.

#### 1.3. MVP Goal

The goal of this project is to apply a structured, hybrid data science approach to the `allenai/WildChat` dataset to uncover insights that inform brand strategies. The output will be a set of files that document the entire process from data exploration to final analysis.

-----

### 2\. User Roles & Personas

  * **Data Scientist (Analyst):** The primary user who will execute the project, requiring clear instructions and a documented approach.
  * **Product Manager/Brand Strategist (Stakeholder):** The end user of the insights, needing actionable intelligence rather than raw data.

-----

### 3\. Functional Requirements

#### 3.1. Project Methodology

The project will follow a customized **CRISP-DM** (Cross-Industry Standard Process for Data Mining) methodology, which is a widely-used analytics model for guiding data mining efforts. This approach ensures a systematic and structured process from start to finish.

#### 3.2. Data Exploration & Understanding

  * **Requirement:** Translate the business objective of finding brand-relevant insights from AI conversations into a data science problem.
  * **Requirement:** Identify and access the `allenai/WildChat` dataset.
  * **Requirement:** Create a `Data Exploration File` that outlines the structure and content of the dataset, including metadata like `conversation_id`, `language`, `model`, and `timestamp`. This file will verify data quality, check for missing values, and provide initial descriptive statistics.

#### 3.3. Data Preparation & Modeling

  * **Requirement:** Develop a hybrid analytical approach that combines efficient semantic models with the nuanced interpretability of LLMs.
  * **Requirement:** Implement **BERTopic** for initial, broad topic modeling of the conversational data. This will allow for the identification of the overarching landscape of topics and their relative frequencies. BERTopic is ideal for this as it leverages BERT embeddings and c-TF-IDF to create dense clusters for easily interpretable topics.
  * **Requirement:** Use **LLMs** for fine-grained tasks that require a deeper understanding of conversation context, such as identifying specific user intents or sentiment. This will be used to analyze themes and underserved areas of interest.
  * **Requirement:** Generate `Task Files` that document the code and process for data cleaning, preprocessing, and the execution of the hybrid topic modeling and intent analysis.

#### 3.4. Evaluation & Business Value

A crucial part of any data science project is a rigorous evaluation of the models and the insights they produce. For this project, evaluation will focus on two key areas: technical performance and business value.

  * **Technical Validation:**

      * **Requirement:** Validate the BERTopic model's performance. This can be done by using internal coherence metrics or by manually reviewing a sample of documents for topic-assignment accuracy.
      * **Requirement:** Given the open-ended nature of the problem, a **zero-shot learning** approach will be adopted for LLM tasks. This means the model will be asked to perform classifications or extractions on data without prior, specific examples. This requires careful prompt engineering to ensure the model understands the task, and the output will be evaluated for its logical consistency and relevance to the prompt.
      * **Requirement:** The `Task Files` must include a section on the chosen validation approach and the results, demonstrating the reliability of the models.

  * **Business Validation:**

      * **Requirement:** Evaluate the quality of the insights by ensuring they are actionable for brands.
      * **Requirement:** Provide clear, completed insights rather than partial ones.
      * **Requirement:** Translate quantitative metrics and qualitative classifications into tangible business value, such as pinpointing customer pain points or tailoring marketing messages.
      * **Requirement:** The final insights should be presented in a way that directly addresses the problem statement, linking conversational patterns (e.g., high-frequency topics, negative sentiment) to potential business actions (e.g., product improvements, new marketing campaigns).

#### 3.5. Deployment

For a data science analysis project with no UI, "deployment" refers to the final delivery of the project's outputs in a clear, accessible, and reproducible format.

  * **Requirement:** The final output will be a comprehensive repository containing all project files.
  * **Requirement:** The `Data Exploration File` and `Task Files` must be well-documented with code, comments, and explanations. This ensures the entire analytical process is transparent and reproducible.
  * **Requirement:** The final insights, derived from the evaluation phase, will be presented in a separate markdown file or presentation-ready format. This document will serve as a high-level summary for stakeholders and will include:
      * An executive summary of key findings.
      * Visualizations of topic frequencies and trends.
      * Specific examples of conversational snippets that illustrate key insights.
      * A clear link between the data-driven insights and potential brand strategies.
  * **Requirement:** Present the final output as a `Product Requirements Document` markdown file, as well as a `Data Exploration File` and `Task Files` as requested.

-----

### 4\. Non-Functional & Technical Requirements

  * **Tools:** The project is open to the use of any tools, but should prioritize the hybrid approach of BERTopic and LLMs.
  * **Dataset:** The project will use the `allenai/WildChat` dataset, or a subsample of it.
  * **Security:** Standard security practices should be followed, especially if an OpenAI API key is used, as mentioned in the assignment.
  * **Timeline:** The project is estimated to be approximately 12 hours of work.