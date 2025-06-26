Python Code Smell Severity Classification Using ML and LLM
This repository contains the full source code, datasets, and experimental results for the research thesis: "Python Code Smell Severity Classification Using Machine Learning and Large Language Models." This study develops a robust framework to detect and classify the severity of four common Python code smells and provides a comparative analysis of traditional ML models against fine-tuned LLMs.

📝 Abstract
Maintaining high source code quality is essential for developing reliable systems. "Code smells"—structural characteristics indicating potential design flaws—can significantly impair this quality. This research addresses the challenge of code smell severity classification in Python by developing a systematic methodology for creating a severity-annotated dataset and evaluating the performance of both traditional Machine Learning (ML) models and Large Language Models (LLMs). Our framework constructs a balanced dataset of 19,500 samples for four key smells: Large Class, Long Method, Long Parameter List, and Deep Nesting. A comparative study shows that while a well-tuned XGBoost model achieves an F1-Score of 0.94, a fine-tuned StarCoder2-3B LLM is highly competitive with an F1-Score of 0.91, demonstrating the powerful capabilities of both paradigms.

✨ Key Features
Novel Python Dataset Methodology: A systematic, reproducible process for creating a Python-specific code smell dataset with multi-level severity annotations.

Heuristic-Based Severity Framework: A defined framework using software metrics (LOC, NOM, NEST, etc.) to classify smell severity into four levels (None, Low, Medium, High).

Comprehensive Comparative Study: A rigorous evaluation and benchmark of traditional ML classifiers (SVM, Random Forest, XGBoost) against modern LLMs (CodeBERT, CodeT5, StarCoder2-3B).

PEFT/LoRA Fine-Tuning: Demonstrates the use of Parameter-Efficient Fine-Tuning (PEFT) with LoRA to efficiently adapt large models for the specific task of code quality analysis.

Model Interpretability: Includes SHAP analysis to identify the most influential software metrics driving the predictions of the ML models.

🛠️ Methodology Overview
The research follows a structured pipeline, starting with data collection and ending with a comparative analysis of the two modeling paradigms. The complete workflow is illustrated below.



⚙️ Setup and Installation
To replicate this study, please follow these steps:

Clone the repository:

git clone [URL_to_repository]
cd [repository_name]

Install dependencies:
It is recommended to use a virtual environment.

pip install -r requirements.txt

Note: A requirements.txt file should be created containing all necessary libraries such as pandas, scikit-learn, xgboost, transformers, peft, lizard, etc.

🚀 Usage and Replication
The project is organized into a series of numbered Python scripts and notebooks that should be run in order.

Data Collection & Processing (Scripts 01 to 07):

01Search.py: Searches GitHub for Python repositories.

02clone.py: Clones the identified repositories.

03extractcode.py: Extracts .py files from the repositories.

04computemetrics.py: Calculates software metrics and applies heuristic rules to annotate smells and severity.

06preprocess.py: Balances the dataset to create the final 19,500 sample set.

07validate.py: Creates the final train/validation/test splits for the ML models.

Machine Learning Experiments (Scripts 08 and 09):

08train.py: Trains, evaluates, and saves the SVM, Random Forest, and XGBoost models.

09MetricsImp.py: Performs SHAP analysis on the trained ML models.

Large Language Model Experiments (Notebooks):

Notebook 1 - Data Reformatting: Prepares the balanced dataset for LLM fine-tuning, including chunking.

LLM Fine-Tuning Scripts: Separate scripts/notebooks for fine-tuning CodeBERT, CodeT5, and StarCoder2 using the prepared data.

📈 Results Summary
Top ML Model: XGBoost was the best-performing traditional ML model, achieving a Macro F1-Score of 0.9403 on the test set.

Top LLM Model: StarCoder2-3B was the most effective LLM, achieving a highly competitive aggregated Macro F1-Score of 0.9126.

Key Insight: The study confirms that both traditional ML models (with strong feature engineering) and large-scale LLMs (with metric-enriched prompts) are highly effective paradigms for Python code smell severity classification.

[LAN XIJIANG]. ([2025]). Python Code Smell Severity Classification Using Machine Learning and Large Language Models. [Universiti Malaya].

🙏 Acknowledgements
I would like to thank my supervisor, Dr. Hema A/P Subramaniam, for her invaluable guidance. I also thank my friends Ding Jun Yang, Wang YingKai, Zhang WenQi, and Sang MeiLing for their insightful discussions and encouragement.
