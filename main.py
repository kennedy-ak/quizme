

import streamlit as st
import os
import tempfile
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, Union

import random
from dotenv import load_dotenv
load_dotenv()


st.set_page_config(
    page_title="PDF Quiz Generator",
    page_icon="📚",
    layout="wide"
)


st.markdown("""
<style>
.quiz-card {
    background-color: #f8f9fa;
    border-radius: 10px;
    padding: 20px;
    margin-bottom: 20px;
    border: 1px solid #dee2e6;
    width: 100%;
    display: block;
    box-sizing: border-box;
    overflow: hidden;
}
.score-display {
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    margin: 20px 0;
    padding: 15px;
    background-color: #e9ecef;
    border-radius: 10px;
    width: 100%;
}
.correct-answer {
    color: #28a745;
    font-weight: bold;
}
.incorrect-answer {
    color: #dc3545;
    font-weight: bold;
}
h3 {
    margin-top: 0;
    margin-bottom: 15px;
}
p {
    margin-bottom: 10px;
}
.stMarkdown {
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'show_results' not in st.session_state:
    st.session_state.show_results = False

class QuizQuestion(BaseModel):
    question: str = Field(description="The text of the question")
    type: str = Field(description="Type of question: 'multiple_choice', 'true_false', or 'fill_in_blank'")
    options: Optional[List[str]] = Field(description="Answer options for multiple choice questions", default=None)
    correct_answer: Union[str, bool] = Field(description="The correct answer to the question")
    explanation: str = Field(description="Explanation of why the answer is correct")

class QuizQuestions(BaseModel):
    questions: List[QuizQuestion] = Field(description="List of quiz questions")

def extract_text_from_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = temp_file.name

    text = ""
    with open(temp_path, "rb") as file:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    os.unlink(temp_path)
    return text


def generate_questions(pdf_text, num_questions, question_types, difficulty):
   
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")
        return None
    
  
    model_name = "llama-3.3-70b-versatile"  
    
  
    if len(pdf_text) > 20000:
        model_options = {
            "llama3-70b-8192": "Llama 3 70B (8K context)",
            "llama3-8b-8192": "Llama 3 8B (8K context)",
            "mixtral-8x7b-32768": "Mixtral 8x7B (32K context)",
            "gemma2-27b-it": "Gemma 2 27B (8K context)"
        }
        
        selected_model = st.selectbox(
            "Select Groq model (larger context windows can handle more text):",
            options=list(model_options.keys()),
            format_func=lambda x: model_options[x],
            index=0
        )
        model_name = selected_model
    
    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=model_name,
    )
    
    # Create a specific prompt for generating questions
    template = """
    You are an expert educational content creator. Your task is to create {num_questions} quiz questions based on the following text. 
    
    TEXT:
    {text}
    
    Create {num_questions} questions of the following type(s): {question_types}.
    The difficulty level should be: {difficulty}.
    
    For multiple-choice questions:
    - Include 4 options (A, B, C, D)
    - Make sure there is exactly one correct answer
    - Ensure wrong answers are plausible
    
    For true/false questions:
    - Create balanced true and false statements
    - Make false statements challenging but clearly incorrect
    
    For fill-in-the-blank questions:
    - Ensure there is a clear, specific answer
    - The blank should replace a key concept or term
    
    For each question, include a brief explanation of why the correct answer is right.
    
    Format your response as a JSON object that strictly follows this structure:
    ```json
    {{
        "questions": [
            {{
                "question": "Question text here",
                "type": "multiple_choice",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_answer": "Option A",
                "explanation": "Explanation of why Option A is correct"
            }},
            {{
                "question": "True or false: Statement here",
                "type": "true_false",
                "options": null,
                "correct_answer": true,
                "explanation": "Explanation of why this is true"
            }},
            {{
                "question": "Statement with a _____ to fill in.",
                "type": "fill_in_blank",
                "options": null,
                "correct_answer": "word",
                "explanation": "Explanation of why 'word' is correct"
            }}
        ]
    }}
    ```
    
    Ensure that the JSON is valid and follows the structure exactly.
    """
    
    prompt = PromptTemplate(
        template=template,
        input_variables=["text", "num_questions", "question_types", "difficulty"]
    )
    
    # Create the parser
    parser = JsonOutputParser(pydantic_object=QuizQuestions)
    
    # Create the chain
    chain = prompt | llm | parser
    
    # Truncate PDF text if it's too long
    max_tokens = 20000 
    if len(pdf_text) > max_tokens:
        pdf_text = pdf_text[:max_tokens]
        st.warning("The PDF content was truncated due to length constraints. The quiz will be based on the first part of the document.")
        
    
    if len(pdf_text) > max_tokens:
        if st.checkbox("My document is very large. Use document chunking for better coverage", value=False):
           
            chunk_size = max_tokens
            overlap = int(chunk_size * 0.2)
            chunks = []
            
           
            for i in range(0, len(pdf_text), chunk_size - overlap):
                end = min(i + chunk_size, len(pdf_text))
                chunks.append(pdf_text[i:end])
                if end == len(pdf_text):
                    break
            
           
            if len(chunks) > 3:
                selected_chunks = random.sample(chunks, 3)  
                
                pdf_text = " ".join(selected_chunks)
                
                if len(pdf_text) > max_tokens:
                    pdf_text = pdf_text[:max_tokens]
                st.info("Using document chunking to generate questions from different parts of your document.")
    
    try:
        result = chain.invoke({
            "text": pdf_text,
            "num_questions": num_questions,
            "question_types": question_types,
            "difficulty": difficulty
        })
        return result
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

# Function to handle question navigation
def go_to_next_question():
  
    question_type = st.session_state.questions[st.session_state.current_question]["type"]
    
    if question_type == "multiple_choice":
        selected_option = st.session_state.get(f"question_{st.session_state.current_question}", None)
        st.session_state.answers.append(selected_option)
    elif question_type == "true_false":
        selected_bool = st.session_state.get(f"question_{st.session_state.current_question}", None)
        st.session_state.answers.append(selected_bool)
    elif question_type == "fill_in_blank":
        filled_answer = st.session_state.get(f"question_{st.session_state.current_question}", "")
        st.session_state.answers.append(filled_answer)
    
    # Move to the next question or show results
    if st.session_state.current_question < len(st.session_state.questions) - 1:
        st.session_state.current_question += 1
    else:
        st.session_state.show_results = True

def go_to_prev_question():
    if st.session_state.current_question > 0:
        st.session_state.current_question -= 1

def restart_quiz():
    st.session_state.questions = []
    st.session_state.current_question = 0
    st.session_state.answers = []
    st.session_state.show_results = False

# Main app layout
st.title("📚 PDF Quiz Generator")

if not st.session_state.questions:
   
    st.write("Upload a PDF and generate quiz questions to test your knowledge!")
    
    with st.form("quiz_setup_form"):
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        col1, col2 = st.columns(2)
        
        with col1:
            num_questions = st.slider("Number of questions", min_value=1, max_value=20, value=5)
            
            st.write("Question Types:")
            mc_box = st.checkbox("Multiple Choice", value=True)
            tf_box = st.checkbox("True/False", value=True)
            fib_box = st.checkbox("Fill in the Blank", value=True)
        
        with col2:
            difficulty = st.select_slider(
                "Difficulty Level",
                options=["Easy", "Medium", "Hard"],
                value="Medium"
            )
        
        submit_button = st.form_submit_button("Generate Quiz")
        
        if submit_button:
            if uploaded_file is None:
                st.error("Please upload a PDF file.")
            else:
                question_types = []
                if mc_box:
                    question_types.append("multiple_choice")
                if tf_box:
                    question_types.append("true_false")
                if fib_box:
                    question_types.append("fill_in_blank")
                
                if not question_types:
                    st.error("Please select at least one question type.")
                else:
                    with st.spinner("Generating quiz questions..."):
                        # Extract text from PDF
                        pdf_text = extract_text_from_pdf(uploaded_file)
                        
                        # Generate questions
                        questions_data = generate_questions(
                            pdf_text, 
                            num_questions, 
                            ", ".join(question_types), 
                            difficulty
                        )
                        
                        if questions_data and "questions" in questions_data:
                            # Store questions in session state
                            st.session_state.questions = questions_data["questions"]
                            st.rerun()
                        else:
                            st.error("Failed to generate questions. Please try again.")

elif st.session_state.show_results:
    # Results page
    st.header("Quiz Results")
    
    correct_answers = 0
    total_questions = len(st.session_state.questions)
    
    for i, (question, answer) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
        
        with st.container():
            st.markdown(f"<div class='quiz-card'>", unsafe_allow_html=True)
            st.markdown(f"<h3>Question {i+1}: {question['question']}</h3>", unsafe_allow_html=True)
            
            is_correct = False
            
            if question["type"] == "multiple_choice":
                user_answer = answer
                correct = question["correct_answer"]
                
                if user_answer == correct:
                    is_correct = True
                    correct_answers += 1
                
                st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer or 'No answer'}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
            elif question["type"] == "true_false":
                user_answer = "True" if answer else "False" if answer is not None else "No answer"
                correct = "True" if question["correct_answer"] else "False"
                
                if (answer is True and question["correct_answer"] is True) or (answer is False and question["correct_answer"] is False):
                    is_correct = True
                    correct_answers += 1
                
                st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
            elif question["type"] == "fill_in_blank":
                user_answer = answer
                correct = question["correct_answer"]
                
                
                if user_answer.lower().strip() == correct.lower().strip():
                    is_correct = True
                    correct_answers += 1
                
                st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer or 'No answer'}</span></p>", unsafe_allow_html=True)
                st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
            st.markdown(f"<p><strong>Explanation:</strong> {question['explanation']}</p>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
    
    # Display final score
    percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
    st.markdown(f"""
    <div class="score-display">
        Your Score: {correct_answers}/{total_questions} ({percentage:.1f}%)
    </div>
    """, unsafe_allow_html=True)
    
    # Feedback based on score
    if percentage >= 90:
        st.success("Excellent work! You've mastered this material.")
    elif percentage >= 70:
        st.success("Good job! You understand most of the content.")
    elif percentage >= 50:
        st.warning("You're on the right track, but might need to review some concepts.")
    else:
        st.error("You might need to spend more time with this material. Don't give up!")
    
    # Button to restart the quiz
    if st.button("Generate New Quiz"):
        restart_quiz()

else:
    # Quiz taking page
    current_q = st.session_state.current_question
    question = st.session_state.questions[current_q]
    
    progress = f"{current_q + 1}/{len(st.session_state.questions)}"
    st.header(f"Question {progress}")
    
    with st.form(f"question_form_{current_q}"):
        st.markdown(f"### {question['question']}")
        
        if question["type"] == "multiple_choice":
            options = question["options"]
            st.radio(
                "Select your answer:",
                options=options,
                key=f"question_{current_q}",
                index=None
            )
        
        elif question["type"] == "true_false":
            st.radio(
                "Select your answer:",
                options=["True", "False"],
                key=f"question_{current_q}",
                index=None,
                format_func=lambda x: x
            )
        
        elif question["type"] == "fill_in_blank":
            st.text_input(
                "Enter your answer:",
                key=f"question_{current_q}"
            )
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if current_q > 0:
                st.form_submit_button("Previous", on_click=go_to_prev_question)
            else:
                st.form_submit_button("Previous", disabled=True)
        
        with col2:
            next_button_text = "Next" if current_q < len(st.session_state.questions) - 1 else "Finish"
            st.form_submit_button(next_button_text, on_click=go_to_next_question)

# Footer
st.markdown("---")
st.markdown("Made with ❤️ for Effe , using LangChain, and Groq")
