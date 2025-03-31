

# # import streamlit as st
# # import os
# # import tempfile
# # from PyPDF2 import PdfReader
# # from langchain_groq import ChatGroq
# # from langchain.prompts import PromptTemplate
# # from langchain_core.output_parsers import JsonOutputParser
# # from langchain.schema import StrOutputParser
# # from pydantic import BaseModel, Field
# # from typing import List, Optional, Union

# # import random
# # from dotenv import load_dotenv
# # load_dotenv()

# # # Set page configuration
# # st.set_page_config(
# #     page_title="PDF Quiz Generator",
# #     page_icon="📚",
# #     layout="wide"
# # )

# # # Add custom CSS
# # st.markdown("""
# # <style>
# # .quiz-card {
# #     background-color: #f8f9fa;
# #     border-radius: 10px;
# #     padding: 20px;
# #     margin-bottom: 20px;
# #     border: 1px solid #dee2e6;
# #     width: 100%;
# #     display: block;
# #     box-sizing: border-box;
# #     overflow: hidden;
# # }
# # .score-display {
# #     font-size: 24px;
# #     font-weight: bold;
# #     text-align: center;
# #     margin: 20px 0;
# #     padding: 15px;
# #     background-color: #e9ecef;
# #     border-radius: 10px;
# #     width: 100%;
# # }
# # .correct-answer {
# #     color: #28a745;
# #     font-weight: bold;
# # }
# # .incorrect-answer {
# #     color: #dc3545;
# #     font-weight: bold;
# # }
# # h3 {
# #     margin-top: 0;
# #     margin-bottom: 15px;
# # }
# # p {
# #     margin-bottom: 10px;
# # }
# # .stMarkdown {
# #     width: 100%;
# # }
# # </style>
# # """, unsafe_allow_html=True)

# # # Initialize session states
# # if 'questions' not in st.session_state:
# #     st.session_state.questions = []
# # if 'current_question' not in st.session_state:
# #     st.session_state.current_question = 0
# # if 'answers' not in st.session_state:
# #     st.session_state.answers = []
# # if 'show_results' not in st.session_state:
# #     st.session_state.show_results = False

# # # Define the expected schema for the LLM output
# # class QuizQuestion(BaseModel):
# #     question: str = Field(description="The text of the question")
# #     type: str = Field(description="Type of question: 'multiple_choice', 'true_false', or 'fill_in_blank'")
# #     options: Optional[List[str]] = Field(description="Answer options for multiple choice questions", default=None)
# #     correct_answer: Union[str, bool] = Field(description="The correct answer to the question")
# #     explanation: str = Field(description="Explanation of why the answer is correct")

# # class QuizQuestions(BaseModel):
# #     questions: List[QuizQuestion] = Field(description="List of quiz questions")

# # # Function to extract text from a PDF file
# # def extract_text_from_pdf(uploaded_file):
# #     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
# #         temp_file.write(uploaded_file.getvalue())
# #         temp_path = temp_file.name

# #     text = ""
# #     with open(temp_path, "rb") as file:
# #         pdf_reader = PdfReader(file)
# #         for page in pdf_reader.pages:
# #             text += page.extract_text()
    
# #     os.unlink(temp_path)
# #     return text

# # # Function to generate questions using Groq
# # def generate_questions(pdf_text, num_questions, question_types, difficulty):
# #     # Initialize the Groq LLM
# #     api_key = os.getenv("GROQ_API_KEY")
    
# #     if not api_key:
# #         st.error("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")
# #         return None
    
# #     # Select the model based on document size
# #     model_name = "llama3-70b-8192"  # Default model
    
# #     # For very large documents, offer the option to use a model with a larger context window
# #     if len(pdf_text) > 8000:
# #         model_options = {
# #             "llama3-70b-8192": "Llama 3 70B (8K context)",
# #             "llama3-8b-8192": "Llama 3 8B (8K context)",
# #             "mixtral-8x7b-32768": "Mixtral 8x7B (32K context)",
# #             "gemma2-27b-it": "Gemma 2 27B (8K context)"
# #         }
        
# #         selected_model = st.selectbox(
# #             "Select Groq model (larger context windows can handle more text):",
# #             options=list(model_options.keys()),
# #             format_func=lambda x: model_options[x],
# #             index=0
# #         )
# #         model_name = selected_model
    
# #     llm = ChatGroq(
# #         groq_api_key=api_key,
# #         model_name=model_name,
# #     )
    
# #     # Create a specific prompt for generating questions
# #     template = """
# #     You are an expert educational content creator. Your task is to create {num_questions} quiz questions based on the following text. 
    
# #     TEXT:
# #     {text}
    
# #     Create {num_questions} questions of the following type(s): {question_types}.
# #     The difficulty level should be: {difficulty}.
    
# #     For multiple-choice questions:
# #     - Include 4 options (A, B, C, D)
# #     - Make sure there is exactly one correct answer
# #     - Ensure wrong answers are plausible
    
# #     For true/false questions:
# #     - Create balanced true and false statements
# #     - Make false statements challenging but clearly incorrect
    
# #     For fill-in-the-blank questions:
# #     - Ensure there is a clear, specific answer
# #     - The blank should replace a key concept or term
    
# #     For each question, include a brief explanation of why the correct answer is right.
    
# #     Format your response as a JSON object that strictly follows this structure:
# #     ```json
# #     {{
# #         "questions": [
# #             {{
# #                 "question": "Question text here",
# #                 "type": "multiple_choice",
# #                 "options": ["Option A", "Option B", "Option C", "Option D"],
# #                 "correct_answer": "Option A",
# #                 "explanation": "Explanation of why Option A is correct"
# #             }},
# #             {{
# #                 "question": "True or false: Statement here",
# #                 "type": "true_false",
# #                 "options": null,
# #                 "correct_answer": true,
# #                 "explanation": "Explanation of why this is true"
# #             }},
# #             {{
# #                 "question": "Statement with a _____ to fill in.",
# #                 "type": "fill_in_blank",
# #                 "options": null,
# #                 "correct_answer": "word",
# #                 "explanation": "Explanation of why 'word' is correct"
# #             }}
# #         ]
# #     }}
# #     ```
    
# #     Ensure that the JSON is valid and follows the structure exactly.
# #     """
    
# #     prompt = PromptTemplate(
# #         template=template,
# #         input_variables=["text", "num_questions", "question_types", "difficulty"]
# #     )
    
# #     # Create the parser
# #     parser = JsonOutputParser(pydantic_object=QuizQuestions)
    
# #     # Create the chain
# #     chain = prompt | llm | parser
    
# #     # Truncate PDF text if it's too long
# #     max_tokens = 16000  # Increased from 8000 to support longer documents
# #     if len(pdf_text) > max_tokens:
# #         pdf_text = pdf_text[:max_tokens]
# #         st.warning("The PDF content was truncated due to length constraints. The quiz will be based on the first part of the document.")
        
# #     # For very large documents, offer chunking option
# #     if len(pdf_text) > max_tokens:
# #         if st.checkbox("My document is very large. Use document chunking for better coverage", value=False):
# #             # Simple chunking strategy - divide the document into multiple chunks with 20% overlap
# #             chunk_size = max_tokens
# #             overlap = int(chunk_size * 0.2)
# #             chunks = []
            
# #             # Create overlapping chunks
# #             for i in range(0, len(pdf_text), chunk_size - overlap):
# #                 end = min(i + chunk_size, len(pdf_text))
# #                 chunks.append(pdf_text[i:end])
# #                 if end == len(pdf_text):
# #                     break
            
# #             # Select random chunks to generate questions from - this will give broader coverage
# #             if len(chunks) > 3:
# #                 selected_chunks = random.sample(chunks, 3)  # Select 3 random chunks
# #                 # Join the selected chunks
# #                 pdf_text = " ".join(selected_chunks)
# #                 # Truncate again if needed
# #                 if len(pdf_text) > max_tokens:
# #                     pdf_text = pdf_text[:max_tokens]
# #                 st.info("Using document chunking to generate questions from different parts of your document.")
    
# #     # Generate the questions
# #     try:
# #         result = chain.invoke({
# #             "text": pdf_text,
# #             "num_questions": num_questions,
# #             "question_types": question_types,
# #             "difficulty": difficulty
# #         })
# #         return result
# #     except Exception as e:
# #         st.error(f"Error generating questions: {str(e)}")
# #         return None

# # # Function to handle question navigation
# # def go_to_next_question():
# #     # Save the current answer
# #     question_type = st.session_state.questions[st.session_state.current_question]["type"]
    
# #     if question_type == "multiple_choice":
# #         selected_option = st.session_state.get(f"question_{st.session_state.current_question}", None)
# #         st.session_state.answers.append(selected_option)
# #     elif question_type == "true_false":
# #         selected_bool = st.session_state.get(f"question_{st.session_state.current_question}", None)
# #         st.session_state.answers.append(selected_bool)
# #     elif question_type == "fill_in_blank":
# #         filled_answer = st.session_state.get(f"question_{st.session_state.current_question}", "")
# #         st.session_state.answers.append(filled_answer)
    
# #     # Move to the next question or show results
# #     if st.session_state.current_question < len(st.session_state.questions) - 1:
# #         st.session_state.current_question += 1
# #     else:
# #         st.session_state.show_results = True

# # def go_to_prev_question():
# #     if st.session_state.current_question > 0:
# #         st.session_state.current_question -= 1

# # def restart_quiz():
# #     st.session_state.questions = []
# #     st.session_state.current_question = 0
# #     st.session_state.answers = []
# #     st.session_state.show_results = False

# # # Main app layout
# # st.title("📚 PDF Quiz Generator")

# # if not st.session_state.questions:
# #     # Initial setup page
# #     st.write("Upload a PDF and generate quiz questions to test your knowledge!")
    
# #     with st.form("quiz_setup_form"):
# #         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
# #         col1, col2 = st.columns(2)
        
# #         with col1:
# #             num_questions = st.slider("Number of questions", min_value=1, max_value=20, value=5)
            
# #             st.write("Question Types:")
# #             mc_box = st.checkbox("Multiple Choice", value=True)
# #             tf_box = st.checkbox("True/False", value=True)
# #             fib_box = st.checkbox("Fill in the Blank", value=True)
        
# #         with col2:
# #             difficulty = st.select_slider(
# #                 "Difficulty Level",
# #                 options=["Easy", "Medium", "Hard"],
# #                 value="Medium"
# #             )
        
# #         submit_button = st.form_submit_button("Generate Quiz")
        
# #         if submit_button:
# #             if uploaded_file is None:
# #                 st.error("Please upload a PDF file.")
# #             else:
# #                 question_types = []
# #                 if mc_box:
# #                     question_types.append("multiple_choice")
# #                 if tf_box:
# #                     question_types.append("true_false")
# #                 if fib_box:
# #                     question_types.append("fill_in_blank")
                
# #                 if not question_types:
# #                     st.error("Please select at least one question type.")
# #                 else:
# #                     with st.spinner("Generating quiz questions..."):
# #                         # Extract text from PDF
# #                         pdf_text = extract_text_from_pdf(uploaded_file)
                        
# #                         # Generate questions
# #                         questions_data = generate_questions(
# #                             pdf_text, 
# #                             num_questions, 
# #                             ", ".join(question_types), 
# #                             difficulty
# #                         )
                        
# #                         if questions_data and "questions" in questions_data:
# #                             # Store questions in session state
# #                             st.session_state.questions = questions_data["questions"]
# #                             st.rerun()
# #                         else:
# #                             st.error("Failed to generate questions. Please try again.")

# # elif st.session_state.show_results:
# #     # Results page
# #     st.header("Quiz Results")
    
# #     correct_answers = 0
# #     total_questions = len(st.session_state.questions)
    
# #     for i, (question, answer) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
# #         # Use a cleaner approach with individual elements instead of continued markdown
# #         with st.container():
# #             st.markdown(f"<div class='quiz-card'>", unsafe_allow_html=True)
# #             st.markdown(f"<h3>Question {i+1}: {question['question']}</h3>", unsafe_allow_html=True)
            
# #             is_correct = False
            
# #             if question["type"] == "multiple_choice":
# #                 user_answer = answer
# #                 correct = question["correct_answer"]
                
# #                 if user_answer == correct:
# #                     is_correct = True
# #                     correct_answers += 1
                
# #                 st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer or 'No answer'}</span></p>", unsafe_allow_html=True)
# #                 st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
# #             elif question["type"] == "true_false":
# #                 user_answer = "True" if answer else "False" if answer is not None else "No answer"
# #                 correct = "True" if question["correct_answer"] else "False"
                
# #                 if (answer is True and question["correct_answer"] is True) or (answer is False and question["correct_answer"] is False):
# #                     is_correct = True
# #                     correct_answers += 1
                
# #                 st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer}</span></p>", unsafe_allow_html=True)
# #                 st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
# #             elif question["type"] == "fill_in_blank":
# #                 user_answer = answer
# #                 correct = question["correct_answer"]
                
# #                 # Case-insensitive comparison for fill-in-the-blank
# #                 if user_answer.lower().strip() == correct.lower().strip():
# #                     is_correct = True
# #                     correct_answers += 1
                
# #                 st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer or 'No answer'}</span></p>", unsafe_allow_html=True)
# #                 st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
# #             st.markdown(f"<p><strong>Explanation:</strong> {question['explanation']}</p>", unsafe_allow_html=True)
# #             st.markdown("</div>", unsafe_allow_html=True)
    
# #     # Display final score
# #     percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
# #     st.markdown(f"""
# #     <div class="score-display">
# #         Your Score: {correct_answers}/{total_questions} ({percentage:.1f}%)
# #     </div>
# #     """, unsafe_allow_html=True)
    
# #     # Feedback based on score
# #     if percentage >= 90:
# #         st.success("Excellent work! You've mastered this material.")
# #     elif percentage >= 70:
# #         st.success("Good job! You understand most of the content.")
# #     elif percentage >= 50:
# #         st.warning("You're on the right track, but might need to review some concepts.")
# #     else:
# #         st.error("You might need to spend more time with this material. Don't give up!")
    
# #     # Button to restart the quiz
# #     if st.button("Generate New Quiz"):
# #         restart_quiz()

# # else:
# #     # Quiz taking page
# #     current_q = st.session_state.current_question
# #     question = st.session_state.questions[current_q]
    
# #     progress = f"{current_q + 1}/{len(st.session_state.questions)}"
# #     st.header(f"Question {progress}")
    
# #     with st.form(f"question_form_{current_q}"):
# #         st.markdown(f"### {question['question']}")
        
# #         if question["type"] == "multiple_choice":
# #             options = question["options"]
# #             st.radio(
# #                 "Select your answer:",
# #                 options=options,
# #                 key=f"question_{current_q}",
# #                 index=None
# #             )
        
# #         elif question["type"] == "true_false":
# #             st.radio(
# #                 "Select your answer:",
# #                 options=["True", "False"],
# #                 key=f"question_{current_q}",
# #                 index=None,
# #                 format_func=lambda x: x
# #             )
        
# #         elif question["type"] == "fill_in_blank":
# #             st.text_input(
# #                 "Enter your answer:",
# #                 key=f"question_{current_q}"
# #             )
        
# #         col1, col2 = st.columns([1, 1])
        
# #         with col1:
# #             if current_q > 0:
# #                 st.form_submit_button("Previous", on_click=go_to_prev_question)
# #             else:
# #                 st.form_submit_button("Previous", disabled=True)
        
# #         with col2:
# #             next_button_text = "Next" if current_q < len(st.session_state.questions) - 1 else "Finish"
# #             st.form_submit_button(next_button_text, on_click=go_to_next_question)

# # # Footer
# # st.markdown("---")
# # st.markdown("Made with ❤️ for Effe , using LangChain, and Groq")

# import streamlit as st
# import os
# import tempfile
# import random
# from PyPDF2 import PdfReader
# from langchain_groq import ChatGroq
# from langchain.prompts import PromptTemplate
# from langchain_core.output_parsers import JsonOutputParser
# from langchain.schema import StrOutputParser
# from pydantic import BaseModel, Field
# from typing import List, Optional, Union

# from dotenv import load_dotenv
# load_dotenv()

# # Set page configuration
# st.set_page_config(
#     page_title="PDF Quiz Generator",
#     page_icon="📚",
#     layout="wide"
# )

# # Add custom CSS
# st.markdown("""
# <style>
# .quiz-card {
#     background-color: #f8f9fa;
#     border-radius: 10px;
#     padding: 20px;
#     margin-bottom: 20px;
#     border: 1px solid #dee2e6;
#     width: 100%;
#     display: block;
#     box-sizing: border-box;
#     overflow: hidden;
# }
# .score-display {
#     font-size: 24px;
#     font-weight: bold;
#     text-align: center;
#     margin: 20px 0;
#     padding: 15px;
#     background-color: #e9ecef;
#     border-radius: 10px;
#     width: 100%;
# }
# .correct-answer {
#     color: #28a745;
#     font-weight: bold;
# }
# .incorrect-answer {
#     color: #dc3545;
#     font-weight: bold;
# }
# h3 {
#     margin-top: 0;
#     margin-bottom: 15px;
# }
# p {
#     margin-bottom: 10px;
# }
# .stMarkdown {
#     width: 100%;
# }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session states
# if 'questions' not in st.session_state:
#     st.session_state.questions = []
# if 'current_question' not in st.session_state:
#     st.session_state.current_question = 0
# if 'answers' not in st.session_state:
#     st.session_state.answers = []
# if 'show_results' not in st.session_state:
#     st.session_state.show_results = False
# if 'randomized_options' not in st.session_state:
#     st.session_state.randomized_options = {}

# # Define the expected schema for the LLM output
# class QuizQuestion(BaseModel):
#     question: str = Field(description="The text of the question")
#     type: str = Field(description="Type of question: 'multiple_choice', 'true_false', or 'fill_in_blank'")
#     options: Optional[List[str]] = Field(description="Answer options for multiple choice questions", default=None)
#     correct_answer: Union[str, bool] = Field(description="The correct answer to the question")
#     explanation: str = Field(description="Explanation of why the answer is correct")

# class QuizQuestions(BaseModel):
#     questions: List[QuizQuestion] = Field(description="List of quiz questions")

# # Function to extract text from a PDF file
# def extract_text_from_pdf(uploaded_file):
#     with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
#         temp_file.write(uploaded_file.getvalue())
#         temp_path = temp_file.name

#     text = ""
#     with open(temp_path, "rb") as file:
#         pdf_reader = PdfReader(file)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
    
#     os.unlink(temp_path)
#     return text

# # Function to generate questions using Groq
# def generate_questions(pdf_text, num_questions, question_types, difficulty):
#     # Initialize the Groq LLM
#     api_key = os.getenv("GROQ_API_KEY")
    
#     if not api_key:
#         st.error("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")
#         return None
    
#     # Select the model based on document size
#     model_name = "llama3-70b-8192"  # Default model
    
#     # For very large documents, offer the option to use a model with a larger context window
#     if len(pdf_text) > 8000:
#         model_options = {
#             "llama3-70b-8192": "Llama 3 70B (8K context)",
#             "llama3-8b-8192": "Llama 3 8B (8K context)",
#             "mixtral-8x7b-32768": "Mixtral 8x7B (32K context)",
#             "gemma2-27b-it": "Gemma 2 27B (8K context)"
#         }
        
#         selected_model = st.selectbox(
#             "Select Groq model (larger context windows can handle more text):",
#             options=list(model_options.keys()),
#             format_func=lambda x: model_options[x],
#             index=0
#         )
#         model_name = selected_model
    
#     llm = ChatGroq(
#         groq_api_key=api_key,
#         model_name=model_name,
#     )
    
#     # Create a specific prompt for generating questions
#     template = """
#     You are an expert educational content creator. Your task is to create {num_questions} quiz questions based on the following text. 
    
#     TEXT:
#     {text}
    
#     Create {num_questions} questions of the following type(s): {question_types}.
#     The difficulty level should be: {difficulty}.
    
#     For multiple-choice questions:
#     - Include 4 options (A, B, C, D)
#     - Make sure there is exactly one correct answer
#     - VERY IMPORTANT: Ensure the correct answer appears in DIFFERENT positions (A, B, C, or D) across different questions
#     - Ensure wrong answers are plausible
    
#     For true/false questions:
#     - Create a balanced mix of true and false statements (roughly 50% true and 50% false)
#     - Make false statements challenging but clearly incorrect
    
#     For fill-in-the-blank questions:
#     - Ensure there is a clear, specific answer
#     - The blank should replace a key concept or term
    
#     For each question, include a brief explanation of why the correct answer is right.
    
#     Format your response as a JSON object that strictly follows this structure:
#     ```json
#     {{
#         "questions": [
#             {{
#                 "question": "Question text here",
#                 "type": "multiple_choice",
#                 "options": ["Option A", "Option B", "Option C", "Option D"],
#                 "correct_answer": "Option B",
#                 "explanation": "Explanation of why Option B is correct"
#             }},
#             {{
#                 "question": "True or false: Statement here",
#                 "type": "true_false",
#                 "options": null,
#                 "correct_answer": false,
#                 "explanation": "Explanation of why this is false"
#             }},
#             {{
#                 "question": "Statement with a _____ to fill in.",
#                 "type": "fill_in_blank",
#                 "options": null,
#                 "correct_answer": "word",
#                 "explanation": "Explanation of why 'word' is correct"
#             }}
#         ]
#     }}
#     ```
    
#     Ensure that the JSON is valid and follows the structure exactly.
#     """
    
#     prompt = PromptTemplate(
#         template=template,
#         input_variables=["text", "num_questions", "question_types", "difficulty"]
#     )
    
#     # Create the parser
#     parser = JsonOutputParser(pydantic_object=QuizQuestions)
    
#     # Create the chain
#     chain = prompt | llm | parser
    
#     # Truncate PDF text if it's too long
#     max_tokens = 16000  # Increased from 8000 to support longer documents
#     if len(pdf_text) > max_tokens:
#         pdf_text = pdf_text[:max_tokens]
#         st.warning("The PDF content was truncated due to length constraints. The quiz will be based on the first part of the document.")
        
#     # For very large documents, offer chunking option
#     if len(pdf_text) > max_tokens:
#         if st.checkbox("My document is very large. Use document chunking for better coverage", value=False):
#             # Simple chunking strategy - divide the document into multiple chunks with 20% overlap
#             chunk_size = max_tokens
#             overlap = int(chunk_size * 0.2)
#             chunks = []
            
#             # Create overlapping chunks
#             for i in range(0, len(pdf_text), chunk_size - overlap):
#                 end = min(i + chunk_size, len(pdf_text))
#                 chunks.append(pdf_text[i:end])
#                 if end == len(pdf_text):
#                     break
            
#             # Select random chunks to generate questions from - this will give broader coverage
#             if len(chunks) > 3:
#                 selected_chunks = random.sample(chunks, 3)  # Select 3 random chunks
#                 # Join the selected chunks
#                 pdf_text = " ".join(selected_chunks)
#                 # Truncate again if needed
#                 if len(pdf_text) > max_tokens:
#                     pdf_text = pdf_text[:max_tokens]
#                 st.info("Using document chunking to generate questions from different parts of your document.")
    
#     # Generate the questions
#     try:
#         result = chain.invoke({
#             "text": pdf_text,
#             "num_questions": num_questions,
#             "question_types": question_types,
#             "difficulty": difficulty
#         })
        
#         # Post-process the questions for additional randomization
#         questions = result["questions"]
        
#         # Further randomize multiple-choice options
#         for i, question in enumerate(questions):
#             if question["type"] == "multiple_choice" and question["options"]:
#                 # Get the correct answer
#                 correct_answer = question["correct_answer"]
                
#                 # Randomize the options
#                 options = question["options"].copy()
#                 correct_index = options.index(correct_answer)
                
#                 # Shuffle the options
#                 random.shuffle(options)
                
#                 # Update the correct answer to match the new position
#                 new_correct_index = options.index(correct_answer)
#                 question["options"] = options
#                 question["correct_answer"] = options[new_correct_index]
            
#             # Randomize true/false answers to ensure balance
#             if question["type"] == "true_false":
#                 # If we have an imbalance in the questions, let's try to balance them
#                 if i > 0 and i % 2 == 1:  # For odd-indexed questions
#                     # Make half the questions true and half false
#                     if (i // 2) % 2 == 0:
#                         # No change needed - keep as LLM generated
#                         pass
#                     else:
#                         # Flip the answer for balance
#                         question["correct_answer"] = not question["correct_answer"]
                        
#                         # Update explanation accordingly
#                         if question["correct_answer"]:
#                             question["explanation"] = "This statement is actually true: " + question["explanation"].replace("This is false", "This is true")
#                         else:
#                             question["explanation"] = "This statement is actually false: " + question["explanation"].replace("This is true", "This is false")
                            
#                         # Update question text if needed
#                         question_text = question["question"]
#                         if question_text.startswith("True or false:"):
#                             # Get the statement part
#                             statement = question_text[len("True or false:"):].strip()
                            
#                             # Modify the statement to flip its meaning, if simple negation possible
#                             if "not" in statement.lower():
#                                 modified_statement = statement.replace(" not ", " ").replace(" Not ", " ")
#                             else:
#                                 # Add negation if possible, otherwise keep as is
#                                 words = statement.split()
#                                 if len(words) > 3:  # Only try to negate if statement is long enough
#                                     if words[0].lower() in ["the", "a", "an"]:
#                                         modified_statement = f"{words[0]} {words[1]} is not {' '.join(words[2:])}"
#                                     else:
#                                         modified_statement = f"{words[0]} does not {' '.join(words[1:])}"
#                                 else:
#                                     modified_statement = statement
                            
#                             question["question"] = f"True or false: {modified_statement}"
        
#         result["questions"] = questions
#         return result
#     except Exception as e:
#         st.error(f"Error generating questions: {str(e)}")
#         return None

# # Function to handle question navigation
# def go_to_next_question():
#     # Save the current answer
#     question_type = st.session_state.questions[st.session_state.current_question]["type"]
    
#     if question_type == "multiple_choice":
#         selected_option = st.session_state.get(f"question_{st.session_state.current_question}", None)
#         st.session_state.answers.append(selected_option)
#     elif question_type == "true_false":
#         selected_answer = st.session_state.get(f"question_{st.session_state.current_question}", None)
#         # Convert string "True"/"False" to boolean True/False
#         if selected_answer == "True":
#             selected_bool = True
#         elif selected_answer == "False":
#             selected_bool = False
#         else:
#             selected_bool = None
#         st.session_state.answers.append(selected_bool)
#     elif question_type == "fill_in_blank":
#         filled_answer = st.session_state.get(f"question_{st.session_state.current_question}", "")
#         st.session_state.answers.append(filled_answer)
    
#     # Move to the next question or show results
#     if st.session_state.current_question < len(st.session_state.questions) - 1:
#         st.session_state.current_question += 1
#     else:
#         st.session_state.show_results = True

# def go_to_prev_question():
#     if st.session_state.current_question > 0:
#         st.session_state.current_question -= 1

# def restart_quiz():
#     st.session_state.questions = []
#     st.session_state.current_question = 0
#     st.session_state.answers = []
#     st.session_state.show_results = False
#     st.session_state.randomized_options = {}

# # Main app layout
# st.title("📚 PDF Quiz Generator")

# if not st.session_state.questions:
#     # Initial setup page
#     st.write("Upload a PDF and generate quiz questions to test your knowledge!")
    
#     with st.form("quiz_setup_form"):
#         uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
#         col1, col2 = st.columns(2)
        
#         with col1:
#             num_questions = st.slider("Number of questions", min_value=1, max_value=20, value=5)
            
#             st.write("Question Types:")
#             mc_box = st.checkbox("Multiple Choice", value=True)
#             tf_box = st.checkbox("True/False", value=True)
#             fib_box = st.checkbox("Fill in the Blank", value=True)
        
#         with col2:
#             difficulty = st.select_slider(
#                 "Difficulty Level",
#                 options=["Easy", "Medium", "Hard"],
#                 value="Medium"
#             )
        
#         submit_button = st.form_submit_button("Generate Quiz")
        
#         if submit_button:
#             if uploaded_file is None:
#                 st.error("Please upload a PDF file.")
#             else:
#                 question_types = []
#                 if mc_box:
#                     question_types.append("multiple_choice")
#                 if tf_box:
#                     question_types.append("true_false")
#                 if fib_box:
#                     question_types.append("fill_in_blank")
                
#                 if not question_types:
#                     st.error("Please select at least one question type.")
#                 else:
#                     with st.spinner("Generating quiz questions..."):
#                         # Extract text from PDF
#                         pdf_text = extract_text_from_pdf(uploaded_file)
                        
#                         # Generate questions
#                         questions_data = generate_questions(
#                             pdf_text, 
#                             num_questions, 
#                             ", ".join(question_types), 
#                             difficulty
#                         )
                        
#                         if questions_data and "questions" in questions_data:
#                             # Store questions in session state
#                             st.session_state.questions = questions_data["questions"]
                            
#                             # Randomize options on page load
#                             for i, question in enumerate(st.session_state.questions):
#                                 if question["type"] == "multiple_choice" and question["options"]:
#                                     correct_answer = question["correct_answer"]
#                                     randomized_options = question["options"].copy()
#                                     random.shuffle(randomized_options)
#                                     # Update the correct_answer to match the new position
#                                     question["correct_answer"] = correct_answer
#                                     question["options"] = randomized_options
                            
#                             st.rerun()
#                         else:
#                             st.error("Failed to generate questions. Please try again.")

# elif st.session_state.show_results:
#     # Results page
#     st.header("Quiz Results")
    
#     correct_answers = 0
#     total_questions = len(st.session_state.questions)
    
#     for i, (question, answer) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
#         # Use a cleaner approach with individual elements instead of continued markdown
#         with st.container():
#             st.markdown(f"<div class='quiz-card'>", unsafe_allow_html=True)
#             st.markdown(f"<h3>Question {i+1}: {question['question']}</h3>", unsafe_allow_html=True)
            
#             is_correct = False
            
#             if question["type"] == "multiple_choice":
#                 user_answer = answer
#                 correct = question["correct_answer"]
                
#                 if user_answer == correct:
#                     is_correct = True
#                     correct_answers += 1
                
#                 st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer or 'No answer'}</span></p>", unsafe_allow_html=True)
#                 st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
#             elif question["type"] == "true_false":
#                 user_answer = "True" if answer else "False" if answer is not None else "No answer"
#                 correct = "True" if question["correct_answer"] else "False"
                
#                 if (answer is True and question["correct_answer"] is True) or (answer is False and question["correct_answer"] is False):
#                     is_correct = True
#                     correct_answers += 1
                
#                 st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer}</span></p>", unsafe_allow_html=True)
#                 st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
#             elif question["type"] == "fill_in_blank":
#                 user_answer = answer
#                 correct = question["correct_answer"]
                
#                 # Case-insensitive comparison for fill-in-the-blank
#                 if user_answer.lower().strip() == correct.lower().strip():
#                     is_correct = True
#                     correct_answers += 1
                
#                 st.markdown(f"<p>Your answer: <span class=\"{'correct-answer' if is_correct else 'incorrect-answer'}\">{user_answer or 'No answer'}</span></p>", unsafe_allow_html=True)
#                 st.markdown(f"<p>Correct answer: <span class=\"correct-answer\">{correct}</span></p>", unsafe_allow_html=True)
            
#             st.markdown(f"<p><strong>Explanation:</strong> {question['explanation']}</p>", unsafe_allow_html=True)
#             st.markdown("</div>", unsafe_allow_html=True)
    
#     # Display final score
#     percentage = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
    
#     st.markdown(f"""
#     <div class="score-display">
#         Your Score: {correct_answers}/{total_questions} ({percentage:.1f}%)
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Feedback based on score
#     if percentage >= 90:
#         st.success("Excellent work! You've mastered this material.")
#     elif percentage >= 70:
#         st.success("Good job! You understand most of the content.")
#     elif percentage >= 50:
#         st.warning("You're on the right track, but might need to review some concepts.")
#     else:
#         st.error("You might need to spend more time with this material. Don't give up!")
    
#     # Button to restart the quiz
#     if st.button("Generate New Quiz"):
#         restart_quiz()

# else:
#     # Quiz taking page
#     current_q = st.session_state.current_question
#     question = st.session_state.questions[current_q]
    
#     progress = f"{current_q + 1}/{len(st.session_state.questions)}"
#     st.header(f"Question {progress}")
    
#     with st.form(f"question_form_{current_q}"):
#         st.markdown(f"### {question['question']}")
        
#         if question["type"] == "multiple_choice":
#             options = question["options"]
#             st.radio(
#                 "Select your answer:",
#                 options=options,
#                 key=f"question_{current_q}",
#                 index=None
#             )
        
#         elif question["type"] == "true_false":
#             st.radio(
#                 "Select your answer:",
#                 options=["True", "False"],
#                 key=f"question_{current_q}",
#                 index=None,
#                 format_func=lambda x: x
#             )
        
#         elif question["type"] == "fill_in_blank":
#             st.text_input(
#                 "Enter your answer:",
#                 key=f"question_{current_q}"
#             )
        
#         col1, col2 = st.columns([1, 1])
        
#         with col1:
#             if current_q > 0:
#                 st.form_submit_button("Previous", on_click=go_to_prev_question)
#             else:
#                 st.form_submit_button("Previous", disabled=True)
        
#         with col2:
#             next_button_text = "Next" if current_q < len(st.session_state.questions) - 1 else "Finish"
#             st.form_submit_button(next_button_text, on_click=go_to_next_question)

# # Footer
# st.markdown("---")
# st.markdown("Made with ❤️ for Effe , using LangChain, and Groq")

import streamlit as st
import os
import tempfile
import random
from PyPDF2 import PdfReader
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.schema import StrOutputParser
from pydantic import BaseModel, Field
from typing import List, Optional, Union

from dotenv import load_dotenv
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="PDF Quiz Generator",
    page_icon="📚",
    layout="wide"
)

# Add custom CSS
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



# Initialize session states
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = 0
if 'answers' not in st.session_state:
    st.session_state.answers = []
if 'show_results' not in st.session_state:
    st.session_state.show_results = False
if 'randomized_options' not in st.session_state:
    st.session_state.randomized_options = {}


def find_closest_match(target, options):
    if not options:
        return None
    
    # First check for exact match
    if target in options:
        return target
    
    # Try case-insensitive match
    target_lower = target.lower().strip()
    for option in options:
        if option.lower().strip() == target_lower:
            return option
    
    # Check if the target is a substring of any option
    for option in options:
        if target_lower in option.lower().strip():
            return option
        
    # Default to first option if no match found
    return options[0]

# Define the expected schema for the LLM output
class QuizQuestion(BaseModel):
    question: str = Field(description="The text of the question")
    type: str = Field(description="Type of question: 'multiple_choice', 'true_false', or 'fill_in_blank'")
    options: Optional[List[str]] = Field(description="Answer options for multiple choice questions", default=None)
    correct_answer: Union[str, bool] = Field(description="The correct answer to the question")
    explanation: str = Field(description="Explanation of why the answer is correct")

class QuizQuestions(BaseModel):
    questions: List[QuizQuestion] = Field(description="List of quiz questions")

# Function to extract text from a PDF file
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

# Function to generate questions using Groq
def generate_questions(pdf_text, num_questions, question_types, difficulty):
    # Initialize the Groq LLM
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("GROQ API key not found. Please set the GROQ_API_KEY environment variable.")
        return None
    
    # Select the model based on document size
    model_name = "llama3-70b-8192"  # Default model
    
    # For very large documents, offer the option to use a model with a larger context window
    if len(pdf_text) > 8000:
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
    - Include 4 options (labeled as "Option A", "Option B", "Option C", "Option D")
    - Make sure there is exactly one correct answer
    - VERY IMPORTANT: Make sure the correct_answer field exactly matches one of the options in the options array
    - VERY IMPORTANT: Vary which option (A, B, C, or D) is correct across different questions
    - Ensure wrong answers are plausible but clearly incorrect
    
    For true/false questions:
    - Create a balanced mix of true and false statements (roughly 50% true and 50% false)
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
                "correct_answer": "Option B",
                "explanation": "Explanation of why Option B is correct"
            }},
            {{
                "question": "True or false: Statement here",
                "type": "true_false",
                "options": null,
                "correct_answer": false,
                "explanation": "Explanation of why this is false"
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
    
    IMPORTANT: Double-check that for each multiple-choice question, the correct_answer EXACTLY matches one of the strings in the options array. Do not add extra text or formatting to the correct_answer that isn't in the options.
    
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
    max_tokens = 16000  # Increased from 8000 to support longer documents
    if len(pdf_text) > max_tokens:
        pdf_text = pdf_text[:max_tokens]
        st.warning("The PDF content was truncated due to length constraints. The quiz will be based on the first part of the document.")
        
    # For very large documents, offer chunking option
    if len(pdf_text) > max_tokens:
        if st.checkbox("My document is very large. Use document chunking for better coverage", value=False):
            # Simple chunking strategy - divide the document into multiple chunks with 20% overlap
            chunk_size = max_tokens
            overlap = int(chunk_size * 0.2)
            chunks = []
            
            # Create overlapping chunks
            for i in range(0, len(pdf_text), chunk_size - overlap):
                end = min(i + chunk_size, len(pdf_text))
                chunks.append(pdf_text[i:end])
                if end == len(pdf_text):
                    break
            
            # Select random chunks to generate questions from - this will give broader coverage
            if len(chunks) > 3:
                selected_chunks = random.sample(chunks, 3)  # Select 3 random chunks
                # Join the selected chunks
                pdf_text = " ".join(selected_chunks)
                # Truncate again if needed
                if len(pdf_text) > max_tokens:
                    pdf_text = pdf_text[:max_tokens]
                st.info("Using document chunking to generate questions from different parts of your document.")
    
    # Generate the questions
    try:
        result = chain.invoke({
            "text": pdf_text,
            "num_questions": num_questions,
            "question_types": question_types,
            "difficulty": difficulty
        })
        
        # Post-process the questions for additional randomization
        questions = result["questions"]
        
        # Further randomize multiple-choice options
        for i, question in enumerate(questions):
            if question["type"] == "multiple_choice" and question["options"]:
                try:
                    # Get the correct answer
                    correct_answer = question["correct_answer"]
                    
                    # Make sure the correct answer is in the options list
                    if correct_answer not in question["options"]:
                        # If not, we need to find what might be the issue
                        st.warning(f"Question {i+1}: Correct answer not found in options. Fixing...")
                        # Get the closest match using our utility function
                        closest_match = find_closest_match(correct_answer, question["options"])
                        correct_answer = closest_match
                        question["correct_answer"] = closest_match
                    
                    # Randomize the options
                    options = question["options"].copy()
                    
                    # Shuffle the options
                    random.shuffle(options)
                    
                    # Update the question with the shuffled options
                    question["options"] = options
                    
                    # Make sure the correct answer is still in options after shuffle
                    if correct_answer in options:
                        # Update the correct answer reference
                        question["correct_answer"] = correct_answer
                    else:
                        # If somehow the correct answer is lost during shuffle (shouldn't happen)
                        # default to the first option
                        question["correct_answer"] = options[0]
                        st.warning(f"Question {i+1}: Correct answer missing after shuffle. Fixed.")
                except Exception as e:
                    st.warning(f"Error processing question {i+1}: {str(e)}. Using first option as correct answer.")
                    # Default to the first option
                    if question["options"] and len(question["options"]) > 0:
                        question["correct_answer"] = question["options"][0]
            
            # Randomize true/false answers to ensure balance
            if question["type"] == "true_false":
                # If we have an imbalance in the questions, let's try to balance them
                if i > 0 and i % 2 == 1:  # For odd-indexed questions
                    # Make half the questions true and half false
                    if (i // 2) % 2 == 0:
                        # No change needed - keep as LLM generated
                        pass
                    else:
                        # Flip the answer for balance
                        question["correct_answer"] = not question["correct_answer"]
                        
                        # Update explanation accordingly
                        if question["correct_answer"]:
                            question["explanation"] = "This statement is actually true: " + question["explanation"].replace("This is false", "This is true")
                        else:
                            question["explanation"] = "This statement is actually false: " + question["explanation"].replace("This is true", "This is false")
                            
                        # Update question text if needed
                        question_text = question["question"]
                        if question_text.startswith("True or false:"):
                            # Get the statement part
                            statement = question_text[len("True or false:"):].strip()
                            
                            # Modify the statement to flip its meaning, if simple negation possible
                            if "not" in statement.lower():
                                modified_statement = statement.replace(" not ", " ").replace(" Not ", " ")
                            else:
                                # Add negation if possible, otherwise keep as is
                                words = statement.split()
                                if len(words) > 3:  # Only try to negate if statement is long enough
                                    if words[0].lower() in ["the", "a", "an"]:
                                        modified_statement = f"{words[0]} {words[1]} is not {' '.join(words[2:])}"
                                    else:
                                        modified_statement = f"{words[0]} does not {' '.join(words[1:])}"
                                else:
                                    modified_statement = statement
                            
                            question["question"] = f"True or false: {modified_statement}"
        
        result["questions"] = questions
        return result
    except Exception as e:
        st.error(f"Error generating questions: {str(e)}")
        return None

# Function to handle question navigation
def go_to_next_question():
    # Save the current answer
    question_type = st.session_state.questions[st.session_state.current_question]["type"]
    
    if question_type == "multiple_choice":
        selected_option = st.session_state.get(f"question_{st.session_state.current_question}", None)
        st.session_state.answers.append(selected_option)
    elif question_type == "true_false":
        selected_answer = st.session_state.get(f"question_{st.session_state.current_question}", None)
        # Convert string "True"/"False" to boolean True/False
        if selected_answer == "True":
            selected_bool = True
        elif selected_answer == "False":
            selected_bool = False
        else:
            selected_bool = None
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
    st.session_state.randomized_options = {}

# Main app layout
st.title("📚 PDF Quiz Generator")

if not st.session_state.questions:
    # Initial setup page
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
                            
                            # Process and validate questions before displaying
                            for i, question in enumerate(st.session_state.questions):
                                if question["type"] == "multiple_choice" and question["options"]:
                                    try:
                                        correct_answer = question["correct_answer"]
                                        
                                        # Verify the correct answer is in the options
                                        if correct_answer not in question["options"]:
                                            # Try to find the closest match
                                            closest_match = find_closest_match(correct_answer, question["options"])
                                            st.warning(f"Question {i+1}: Fixing incorrect options-answer match. Using '{closest_match}' as correct answer.")
                                            question["correct_answer"] = closest_match
                                            correct_answer = closest_match
                                        
                                        # Randomize options order
                                        randomized_options = question["options"].copy()
                                        random.shuffle(randomized_options)
                                        question["options"] = randomized_options
                                    except Exception as e:
                                        st.warning(f"Error processing Question {i+1}: {str(e)}")
                                        # Make sure we have a valid state
                                        if question["options"] and len(question["options"]) > 0:
                                            question["correct_answer"] = question["options"][0]
                            
                            st.rerun()
                        else:
                            st.error("Failed to generate questions. Please try again.")

elif st.session_state.show_results:
    # Results page
    st.header("Quiz Results")
    
    correct_answers = 0
    total_questions = len(st.session_state.questions)
    
    for i, (question, answer) in enumerate(zip(st.session_state.questions, st.session_state.answers)):
        # Use a cleaner approach with individual elements instead of continued markdown
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
                
                # Case-insensitive comparison for fill-in-the-blank
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