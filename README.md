# English-Urdu Neural Machine Translation using Transformers

This repository contains my implementation of a neural machine translation system that translates English text to Urdu, leveraging a custom-built **Transformer** model. An additional implementation of **LSTM-based** sequence to sequence model has also been added for comparison and understanding.

## Project Overview

This project focuses on building and training a Transformer model from scratch to perform English-Urdu translation. It utilizes the **UMC005 parallel corpus**. The project highlights:

*   **Custom Transformer Architecture:**  A ground-up implementation of the Transformer model, implementing its components like Multi-Headed Attention, Encoder-Decoder layers, and Positional Encoding.
*   **Urdu Language Focus:**  Addresses the nuances of Urdu, a language with rich morphology, using SentencePiece for effective tokenization.
*   **Efficient Training:** Implements techniques like learning rate scheduling, early stopping, and checkpointing for optimal training.
*   **Streamlit Frontend:** Provides a user-friendly web interface for interactive translation, built using Streamlit.
*   **Comparative Analysis:** The project also provides a comparative analysis with a traditional **LSTM-based** model.
*   **Weights & Biases Integration:** Uses Weights and Biases (wandb) for experiment tracking and visualizing training metrics.
*   **BLEU Score Evaluation:** Employs the BLEU score, a standard metric for evaluating the quality of machine-translated text.

## Key Features

*   **End-to-End Translation:** Translates English sentences to Urdu using a custom-trained Transformer model.
*   **Custom Transformer:**  Built from the ground up using pytorch.
*   **SentencePiece Tokenization:**  Effectively handles Urdu's complex morphology through subword tokenization.
*   **Interactive Web App:**  User-friendly interface for real-time translation using Streamlit.
*   **Attention Visualization:** Visualize attention to see how the model focuses on different parts of a sentence.
*   **BLEU Score Evaluation:**  Quantifies translation quality using the widely-used BLEU score.
*   **Experiment Tracking:** Leverages Weights and Biases to monitor experiments and log results.

## Model Architecture

The core of this project is a custom **Transformer** model, inspired by the original "Attention is All You Need" paper. It consists of:

*   **Encoder:**
    *   **Embedding Layer:** Converts input tokens into dense vector representations.
    *   **Positional Encoding:**  Adds information about the position of words in the sentence.
    *   **Multiple Encoder Layers:** Each layer consists of:
        *   **Multi-Headed Self-Attention:**  Allows the model to attend to different parts of the input sequence.
        *   **Feed-Forward Neural Network:**  Further processes the information from the attention layer.
        *   **Residual Connections and Layer Normalization:**  Improve training stability and performance.
*   **Decoder:**
    *   **Embedding Layer:** Converts target tokens into dense vector representations.
    *   **Positional Encoding:** Adds positional information to the target sequence.
    *   **Multiple Decoder Layers:** Each layer consists of:
        *   **Masked Multi-Headed Self-Attention:**  Prevents the model from "peeking" at future tokens during training.
        *   **Multi-Headed Attention (Encoder-Decoder):** Allows the decoder to attend to the output of the encoder.
        *   **Feed-Forward Neural Network:**  Further processes the information.
        *   **Residual Connections and Layer Normalization:**  Improve training and performance.
*   **Linear Output Layer:**  Projects the decoder output to the vocabulary space, producing probabilities for each target token.
*   **LSTM-Based Model:**
    *   **Embedding Layer:** Converts input tokens into dense vector representations.
    *   **LSTM Layer:** Recurrent layer capable of capturing sequential dependencies.
    *   **Linear Output Layer:** Projects the output of the LSTM layer to the vocabulary space.
    *   **Attention Mechanism:** Used to enable decoder to attend to different parts of the input sequences.

## Dataset

The model is trained on the **UMC005: English-Urdu Parallel Corpus** (https://ufal.mff.cuni.cz/umc/005-en-ur/). This dataset contains a collection of English-Urdu sentence pairs, suitable for training machine translation models.

The dataset is split into three parts:

*   **Training Set:** Used to train the Transformer and LSTM-based models.
*   **Validation Set:** Used to monitor performance during training and tune hyperparameters.
*   **Test Set:** Used to evaluate the final performance of the trained models.

## Training

The training process involves the following key steps:

1. **Data Preprocessing:**
    *   Loading and cleaning the UMC005 dataset.
    *   Tokenizing the sentences using SentencePiece.
    *   Creating training, validation, and test sets.
    *   Padding sequences to a maximum length for efficient batch processing.
2. **Model Initialization:**
    *   Creating instances of the custom Transformer and LSTM-based models.
    *   Initializing model weights using Xavier initialization.
3. **Optimization:**
    *   Using the Adam optimizer with a custom learning rate scheduler.
    *   Employing Cross-Entropy Loss for training.
4. **Training Loop:**
    *   Iterating through epochs and batches of training data.
    *   Forward and backward passes to update model weights.
    *   Monitoring training and validation loss using Weights and Biases.
    *   Implementing early stopping to prevent overfitting.
5. **Model Checkpointing:**
    *   Saving the best-performing model based on validation loss.

## Evaluation

The model's performance is evaluated using the **BLEU (Bilingual Evaluation Understudy)** score, a standard metric for machine translation. BLEU measures the n-gram overlap between the generated translation and a reference translation.


## Getting Started

### Prerequisites

*   Python 3.10.14
*   PyTorch 2.4.0
*   Transformers 4.45.1
*   Datasets 3.0.1
*   SentencePiece 0.2.0
*   SacreBLEU 2.4.3
*   Streamlit
*   Wandb 0.18.3
*   Evaluate 0.4.3
*   Rouge Score 0.1.2 

Install the required libraries using pip:

```bash
pip install torch transformers datasets sentencepiece sacrebleu streamlit wandb evaluate
```

### Dataset Setup

1. **Download the UMC005 dataset:** You can find it at https://ufal.mff.cuni.cz/umc/005-en-ur/
2. **Place the dataset** in the `data/` directory. The directory structure should look like this:
```
umc005-en-ur-translation/
├── data/
│   ├── umc005-corpus/
│   │   ├── bible/
│   │   │   ├── train.en
│   │   │   ├── train.ur
│   │   │   ├── dev.en
│   │   │   ├── dev.ur
│   │   │   ├── test.en
│   │   │   └── test.ur
│   │   └── news/
│   │       ├── train.en
│   │       ├── train.ur
│   │       ├── dev.en
│   │       ├── dev.ur
│   │       ├── test.en
│   │       └── test.ur
├── final-mbart-en-ur-finetuned/
│   └── ...
├── UMC005_en.model
├── UMC005_en.vocab
├── UMC005_ur.model
├── UMC005_ur.vocab
├── final_model.pt
├── umc_machine_trans.ipynb
└── ...
```
### Running the Code
1. Clone this repository:

    ```bash
    git clone [repository URL]
    cd [repository directory]
    ```
2. Train the model:

    *   Open the Jupyter Notebook `training_notebook.ipynb`.
    *   Adjust the paths to the dataset if necessary.
    *   Run all cells to train the model.
3. Run the Streamlit app:
    ```bash
    streamlit run app.py
    ```
### Usage
**Using the Web Interface:**

1. Launch the Streamlit app using the command above.
2. Enter an English sentence in the input box.
3. Click the "Translate" button.
4. The translated Urdu sentence will be displayed.

## Repository Files

*   **`umc_machine_trans.ipynb`:** Jupyter Notebook for training the Transformer model.
*   **`app.py`:** Python script for the Streamlit web application.
*   **`UMC005_en.model`, `UMC005_en.vocab`:** SentencePiece model and vocabulary for English.
*   **`UMC005_ur.model`, `UMC005_ur.vocab`:** SentencePiece model and vocabulary for Urdu.
*   **`final_model.pt`:** Saved weights of the trained Transformer model.
*   **`final-mbart-en-ur-finetuned`:** Saved weights of the fine tuned m-bart model.
*   **`data/`:** Directory for storing the UMC005 dataset.
*   **`Custom Transformer Implementation for Machine Translation on UMC005 English-Urdu Parallel Corpus.pdf`:** The research paper detailing the project.
*   **`README.md`:** This file.

## Acknowledgements

*   **UMC005 Dataset:** Thanks to the creators of the UMC005 dataset for providing the valuable parallel corpus.
*   **"Attention is All You Need" Paper:** This project is inspired by the groundbreaking work on Transformers.
*   **Weights & Biases** For providing a platform for experiment tracking.

## Images

![{E7110A7E-94BD-4107-8CC5-DE2B5F954284}](https://github.com/user-attachments/assets/484d010e-ca3d-4f9a-86aa-88fb7de53b09)
![{8124D16C-A58E-476A-A4B1-8A2843DBAADA}](https://github.com/user-attachments/assets/2734cb3e-1e34-4dbd-bf65-774e00295dc3)

