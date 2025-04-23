# 🎬 Movie Dialogue Text Autocompletion using Bi-LSTM

This project builds a **text autocompletion model** using a **Bidirectional LSTM (Long Short-Term Memory)** network trained on the Cornell Movie Dialogues dataset. The model learns to predict the next word in a sequence, enabling autocomplete functionality similar to predictive typing or chatbot text generation.

## ✨ Features

- Clean and tokenize movie dialogues
- Build input sequences using n-gram modeling
- Use Bi-LSTM for contextual understanding in both directions
- Train on real-world conversational data
- Predict the next 20 words given a starting prompt
- Visualize training progress with accuracy and loss plots

---

## 🧠 What is LSTM and Why Use It?

### 🔍 Overview of LSTM

**LSTM (Long Short-Term Memory)** is a type of Recurrent Neural Network (RNN) designed to learn and remember long sequences of data — such as sentences or time-series information. Unlike standard RNNs, LSTMs overcome the **vanishing gradient problem**, allowing them to preserve context over long distances.

### 📘 How LSTM Works

An LSTM unit is composed of three main gates:

1. **Forget Gate**: Decides what information to discard.
2. **Input Gate**: Decides what new information to store.
3. **Output Gate**: Decides what information to output.

Each gate uses learned weights and activation functions to control the flow of information, enabling the model to **retain relevant context** across long text inputs.


| Feature                        | Standard RNN | LSTM     |
|-------------------------------|--------------|----------|
| Handles long-term dependencies| ❌           | ✅       |
| Learns context effectively    | ❌           | ✅       |
| Prevents vanishing gradients  | ❌           | ✅       |
| Suitable for text generation  | ⚠️ Limited   | ✅ Best  |

In this project, we go one step further and use a **Bidirectional LSTM**, which reads the input both forward and backward, allowing it to capture more complex patterns in language.

### ✅ Why LSTM Works Well for Autocompletion

- **Memory of context**: LSTM remembers previous words in a sentence, which is crucial for predicting the next word in a grammatically and contextually correct way.
- **Handles long-term dependencies**: Better than standard RNNs, LSTMs mitigate the vanishing gradient problem.
- **Bidirectional LSTM** used here allows the model to learn from both past (left context) and future (right context) during training, improving prediction accuracy.

---


---

## 📂 Dataset

- **Name**: Cornell Movie Dialogues Corpus
- **File Used**: `movie_lines.txt`
- **Lines Processed**: First 5,000 (can be expanded)
- **Cleaning Steps**:
  - Remove punctuation
  - Convert to lowercase
  - Filter out non-alphabetic characters

---

## 🏗️ Model Architecture

Embedding ➝ Bidirectional LSTM ➝ Dense Softmax

- **Embedding Layer**: Converts words into vector representations
- **Bidirectional LSTM**: Learns dependencies in both directions
- **Dense Layer**: Outputs a probability distribution over the vocabulary

---

## ⚙️ Setup Instructions

### 📦 Dependencies

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib
```

## Future Improvements:

- Train on the full dataset (200k+ lines)

- Use temperature-based sampling for more diverse predictions

- Replace greedy decoding with beam search

- Add attention mechanism or transition to transformers
