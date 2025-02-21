
# 🤖 Generative AI Chatbot by Nandesh Kalashetti

[![Watch Demo](chatbot.png)](https://youtu.be/V0ffoYtm_Uk)

## Overview

Welcome to the **Generative AI Chatbot** repository! This project is a personal chatbot designed to showcase the expertise and innovative mindset of **Nandesh Kalashetti** – a visionary full‑stack web developer and mentor. Leveraging advanced techniques like **Retrieval Augmented Generation (RAG)**, **advanced prompt engineering**, and **optimized inference**, this chatbot dynamically integrates data from an uploaded PDF (my resume) to provide accurate, context-aware, and human-like responses. 😊🚀

## Key Features

- **📄 PDF Integration:**  
  Extracts and indexes key details from my resume, including my education, technical skills, projects, certifications, and more.

- **🧠 Advanced Models:**  
  - Uses the **sentence-transformers/all-MiniLM-L6-v2** model for semantic embeddings.  
  - Powered by the **Groq Chat model (Llama3-70b-8192)** for generating precise and human-like responses.

- **💬 Adaptive Response:**  
  - Provides **short, emoji-enhanced answers** for simple queries.  
  - Delivers **detailed, structured explanations** for complex questions, adapting the response length based on user needs.

- **📚 Context-Aware:**  
  Retrieves relevant context from a vector database (ChromaDB) to ensure responses are both accurate and relevant.

- **✨ Modern UI:**  
  Built with **Streamlit** and enhanced with custom CSS for a sleek and interactive user experience.

## How It Works

1. **Data Ingestion:**  
   The chatbot extracts text from an uploaded PDF (my resume), splits it into manageable chunks, and stores it in ChromaDB using semantic embeddings.

2. **Query Processing:**  
   When a user asks a question, the chatbot retrieves the most relevant context from the stored data and combines it with recent conversation history.

3. **Response Generation:**  
   A sophisticated prompt guides the Groq Chat model to generate responses that are precise, empathetic, and tailored to the query—all while maintaining a warm, human tone. 👍

## Getting Started

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/YourUsername/YourRepo.git
   cd YourRepo
   ```

2. **Create & Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Chatbot:**
   ```bash
   streamlit run FinalProject1.py
   ```
   Then, open the provided URL (usually `http://localhost:8501`) in your browser.

## Demo Video

Watch the demo video on YouTube to see the chatbot in action:

[![Watch Demo](chatbot.png)](https://youtu.be/V0ffoYtm_Uk)

## Why This Chatbot?

- **Innovative Integration:** Combines modern AI techniques with personal data to deliver context-rich responses.  
- **User-Centric Design:** Adapts response length and tone based on user needs, ensuring both brevity and depth when required.  
- **Future-Ready:** Designed to demonstrate real-world applicability, making it a powerful tool for both personal growth and professional presentations.

---
