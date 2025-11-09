# **Sentiment-LSTM**
Project Objective: Build and train an LSTM model to classify movie reviews as positive or negative.

---

## **Overview**
* [1 Introduction](#1-introduction)  
* [2 Getting Started](#2-getting-started)  
  * [2.1 Preparations](#21-preparations)  
  * [2.2 Install All Dependencies](#22-install-all-dependencies)  
  * [2.3 Prepare the Data](#23-prepare-the-data)  
* [3 Run Files](#3-run-files)  
  * [3.1 preprocessing.py](#31-preprocessingpy)  
  * [3.2 DataLoader.py](#32-dataloaderpy)  
  * [3.3 LSTM.py](#33-lstmpy)  
  * [3.4 main_sentiment.py](#34-main_sentimentpy)  
* [4 Other Documentations](#4-other-documentations)  
* [5 Bonus: Pre-trained Embedding Comparison](#5-bonus-pre-trained-embedding-comparison)

---

# **1. Introduction**

This project implements a **Long Short-Term Memory (LSTM)** network for **sentiment analysis** on movie reviews.  

The goal is to classify each review as **positive (1)** or **negative (0)** using a neural network trained on textual data.  

The workflow includes:
1. **Data Preprocessing** – cleaning, tokenizing, encoding, and padding sequences  
2. **Custom DataLoader** – converting CSV data into PyTorch-ready batches  
3. **LSTM Model** – embedding layer + recurrent network for text classification  
4. **Training and Evaluation** – binary cross-entropy loss optimization and accuracy evaluation  

This project demonstrates end-to-end NLP model training in **PyTorch**, following a standard deep-learning workflow.

---

# **2. Getting Started**

This project uses **Python 3.9+** and works on **macOS, Linux, and Windows**.

---

## **2.1 Preparations**

(1) Clone this repository:  
```shell
git clone https://github.com/cathieG/LSTM_Movie_Sentiment.git
```
(2) Navigate into the repository:
```shell
cd LSTM_Movie_Sentiment
```
(3) Set up a virtual environment and activate it:

For macOS/Linux:
```shell
python -m venv ./venv/
source venv/bin/activate
```
For Windows:
```shell
python -m venv venv
venv\Scripts\activate
```

To deactivate the virtual environment, use the command:
```shell
deactivate
```
## **2.2 Install All Dependencies**

Install the required dependencies:
```shell
pip install -r requirements.txt
```
