# PDF Quiz Generator 📚

The **PDF Quiz Generator** is a Streamlit-based web application that allows users to upload a PDF document and generate quiz questions based on its content. The app leverages Groq's language models and LangChain to create multiple-choice, true/false, and fill-in-the-blank questions.

## Features

- **PDF Upload**: Upload any PDF document to extract its text.
- **Question Types**: Generate multiple-choice, true/false, and fill-in-the-blank questions.
- **Customizable Difficulty**: Choose between Easy, Medium, and Hard difficulty levels.
- **Dynamic Question Count**: Select the number of questions to generate (1–20).
- **Interactive Quiz**: Take the quiz directly in the app and view your results.
- **Score Feedback**: Get detailed feedback on your performance, including correct answers and explanations.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/pdf-quiz-generator.git
   cd pdf-quiz-generator

2. Create a virtual environment:
python -m venv env
source env/Scripts/activate  # On Windows
source env/bin/activate      # On macOS/Linux

3. Install dependencies:
pip install -r requirements.txt