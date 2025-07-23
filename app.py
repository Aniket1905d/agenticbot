import os
import logging
import json
from typing import Dict, Any, List, Optional

from flask import Flask, render_template, request, jsonify
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, ValidationError
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("support_classifier.log")
    ]
)
logger = logging.getLogger(__name__)

# ------------------- Data Models -------------------
class ProcedureResult(BaseModel):
    query_type: str  # "problem" or "guidance"
    procedure_name: str
    original_message: str

class Problem(BaseModel):
    problem_text: str
    context: Optional[str] = ""  # default to empty string
    problem_id: str
    current_user_problem: str

class ProblemDetectionResult(BaseModel):
    has_multiple_problems: bool
    needs_clarification: bool
    clarification_question: Optional[str] = None
    problems: List[Problem]
    separation_reasoning: str
    original_message: str

class ClassificationResult(BaseModel):
    classification: Optional[str]  # One of: hardware, software, etc.
    confidence: float
    reasoning: str
    need_question: bool
    question: Optional[str]

# --- MODIFIED: ConversationState Model ---
class ConversationState(BaseModel):
    current_problems: List[Dict] = []
    current_problem_index: int = 0
    conversation_history: List[Dict] = []
    is_awaiting_followup: bool = False
    is_awaiting_problem_selection: bool = False # NEW state

# ------------------- Prompt Manager -------------------
class PromptManager:
    """Centralized management of all system prompts"""

    @staticmethod
    def get_procedure_detection_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
            """You are a technical support intent classifier. Determine if the user is:
            A) Describing a technical PROBLEM (needs classification)
            B) Requesting step-by-step GUIDANCE for a procedure
            C) Asking something ambiguous (needs clarification) than treat it as a problem and return problem in query_type

            **PROCEDURAL REQUEST INDICATORS:**
            - "how to", "step by step", "guide me", "show me", "procedure for", "instructions for"
            - Asking for instructions rather than describing symptoms
            - Imperative forms: "tell me...", "explain..."

            **PROBLEM REPORT INDICATORS:**
            - Describes symptoms (crashes, errors, slowness, failures)
            - Uses words like: "problem", "issue", "not working", "broken", "error"
            - Reports unexpected behavior

            **RESPONSE FORMAT:**
            {{
                "query_type": "problem" | "guidance" ,
                "procedure_name": "procedure name if guidance"| "NULL name if problem",
                "original_message": "user input message"
            }}"""), 
            ("human", "User message: {user_message}")
        ])

    @staticmethod
    def get_problem_detection_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system",
            """
            You are an expert in identifying no of problem in user input. Your task is to systematically separate multiple distinct technical problems from user messages.

            **PROBLEM IDENTIFICATION :**
            1. Count and separate clearly distinct issues affecting different systems, applications, or components.
            2. Only separate if two or more issues have different root causes or contexts.
            4. Separate only if Clearly Distinct: Separate problems only if they affect unrelated systems (e.g., "my printer is jammed and I can't access my email") or have clearly different symptoms and contexts.
            5. If the user input clearly describes _one_ symptom (e.g. “my pc is slow”, “printer won't print”), assume _one_ problem and do NOT ask any clarification question about that symptom.

            **RULES**
            1. Do NOT ask about the nature or details of a symptom — your job is ONLY to determine number of distinct problems, not to analyze what the problem is.
            2. Treat simple, single-clause inputs (like "my pc not working", "email not opening", "printer is jammed") as a single, clearly defined problem. Do NOT request clarification in such cases.
            3. Consolidate Symptoms: If one symptom is clearly a result of another (e.g. "The app crashes and I lose my work"), treat as a single problem.
            4. CRITICAL SEPARATION RULE: Separate problems that suggest different technical categories:
               - Performance issues (slowness, lag, loading times) vs Hardware issues (overheating, physical problems)
               - Software crashes vs Hardware failures  
               - App-specific problems vs System-wide problems
            6. Ask for clarification *only* when you cannot determine if the described situation involves 1 or multiple issues — e.g., if the syntax is ambiguous or two clauses blur together.
            7. Never ask for clarification based just on lack of technical detail in a single clause — this is out of scope.

            If need to clarify, set "needs_clarification": true, leave "problems" empty, and provide a targeted question in "clarification_question". Otherwise, set "needs_clarification": false and proceed.

            **ENHANCED RESPONSE FORMAT:**
            {{
                "has_multiple_problems": true/false,
                "needs_clarification": true/false,
                "clarification_question": "A single, targeted question to resolve ambiguity." OR null,
                "problems": [
                    {{
                        "problem_text": "A clear, specific, and consolidated problem description.",
                        "context": "All relevant context, symptoms, and conditions.",
                        "problem_id": "problem_1/2/...",
                        "current_user_problem": "The part of the original message describing this consolidated problem."
                    }}
                ],
                "separation_reasoning": "A clear explanation of why problems were either separated or consolidated into one.",
                "original_message": "The full original user message."
            }}
            """),
            ("human", "User message: {user_message}\nConversation history: {conversation_history2}")
        ])

    @staticmethod
    def get_guidance_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert technical support assistant. The user has requested step-by-step instructions for a procedure. 
            Generate comprehensive, accurate guidance with the following structure:

            **RESPONSE GUIDELINES:**
            1. Start with a clear title
            2. Provide step-by-step instructions in a clean numbered list format
            3. Include platform variations (Windows/Mac/Linux/iOS/Android) where applicable
            4. Add important warnings/cautions where necessary
            5. Mention prerequisites if needed
            6. End with troubleshooting tips or next steps
            7. Use PLAIN TEXT FORMAT only - NO MARKDOWN, NO ASTERISKS, NO HASHTAGS

            **CRITICAL FORMATTING RULES:**
            - DO NOT use markdown symbols like *, #, **, etc.
            - DO NOT use bullet points (•) or asterisks
            - Use simple numbered lists with clear line breaks
            - Separate sections with clear spacing
            - Keep each step concise (1 line per step)
            - Use colons (:) for platform-specific instructions
            Generate instructions for: {procedure_topic}"""),
            ("human", "User request: {user_message}")
        ])

    @staticmethod
    def get_classification_prompt() -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert technical support classifier. Analyze user messages and classify them into these categories:

            1. **Hardware** - Physical device problems (screen, battery, keyboard, ports, overheating, won't turn on, fell and broke, physical damage)
            2. **Software/OS** - Operating system, boot, driver, update issues (won't boot, Windows update failed, driver missing, system-wide software issues)
            3. **Performance** - System or app slowness, resource usage, apps running slowly or poorly (computer slow, startup takes forever, app is slow, not responding well)
            4. **App Crash** - Application crashes, freezes, error dialogs, won't start (specific app crashes, program hangs, error messages)
            5. **System Crash** - System-wide crashes, BSOD, random reboots (blue screen, system freezes, computer shuts down)

            **DIAGNOSTIC APPROACH:**
            You must distinguish between:
            - **Surface symptoms** vs **root causes**
            - **App-specific** vs **system-wide** issues
            - **Hardware** vs **software** problems
            - **Performance slowness** vs **crash issues**

            CRITICAL UNDERSTANDING:
            - "App not working" could mean: Performance (slow/laggy), App Crash (won't start/crashes), or Software/OS (system-wide issue)
            - "Computer won't start" could mean: Hardware (power/physical) OR Software/OS (boot failure)
            - "Slow" typically means Performance, but could be Hardware (failing components) or Software/OS (malware/corruption)
            - Only classify as "App Crash" if user mentions crashing, freezing, error messages, or won't start
            - "Not working properly" often indicates Performance issues unless crash symptoms are described

            ANALYSIS APPROACH:
            1. Consider ALL possible interpretations of vague language
            2. Ask targeted questions to distinguish between similar categories
            3. Don't assume "not working" means "crashing" - could be performance
            4. Use precise diagnostic questions based on symptoms

            Respond with JSON in this exact format:
            {{
                "classification": "final_category_or_null", 
                "confidence": 0.0_to_1.0,
                "reasoning": "internal_step_by_step_analysis",
                "need_question": true_or_false,
                "question": "targeted_diagnostic_question_if_needed"
            }}

            Decision Logic:
            - If CLEARLY only 1 category possible (high confidence): Classify immediately
            - If multiple interpretations possible: Set needs_question=true and ask to differentiate
            - Focus on what the user ACTUALLY said, not assumptions about what they meant"""),
            ("human", "Problem to classify: {problem_text}\n Conversation: {conversation_history}")
        ])
class TechnicalSupportClassifier:
    """Enhanced technical support classifier with state management"""

    def __init__(self, model: str = "llama3-70b-8192",model1: str = "gemma2-9b-it"):
        self.model_name = model
        self.model_name1= model1
        self.llm = self._initialize_llm()
        self.llm1 = self._initialize_llm1()
        self.prompt_manager = PromptManager()
        self.parser = JsonOutputParser()

        # Initialize processing chains
        self.procedure_chain = self._build_chain(
            self.prompt_manager.get_procedure_detection_prompt(),
            pydantic_model=ProcedureResult
        )
        self.problem_detection_chain = self._build_chain1(
            self.prompt_manager.get_problem_detection_prompt(),
            pydantic_model=ProblemDetectionResult
        )
        self.guidance_chain = self._build_chain(
            self.prompt_manager.get_guidance_prompt()
        )
        self.classification_chain = self._build_chain(
            self.prompt_manager.get_classification_prompt(),
            pydantic_model=ClassificationResult
        )

    def _initialize_llm(self) -> ChatGroq:
        """Initialize Groq LLM with API key validation"""
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY environment variable is required")
            raise ValueError("Missing GROQ_API_KEY environment variable")

        return ChatGroq(
            model_name=self.model_name,
            temperature=0.9,
            api_key=groq_api_key
        )
    def _initialize_llm1(self) -> ChatGroq:
        """Initialize Groq LLM with API key validation"""
        groq_api_key = os.environ.get("GROQ_API_KEY")
        if not groq_api_key:
            logger.error("GROQ_API_KEY environment variable is required")
            raise ValueError("Missing GROQ_API_KEY environment variable")

        return ChatGroq(
            model_name=self.model_name1,
            temperature=0.0,
            api_key=groq_api_key
        )

    def _build_chain(self, prompt: ChatPromptTemplate, pydantic_model: BaseModel = None):
        """Build processing chain with optional validation"""
        chain = prompt | self.llm
        if pydantic_model:
            chain = chain | self.parser | self._create_validator(pydantic_model)
        return chain

    def _build_chain1(self, prompt: ChatPromptTemplate, pydantic_model: BaseModel = None):
        """Build processing chain with optional validation"""
        chain = prompt | self.llm1
        if pydantic_model:
            chain = chain | self.parser | self._create_validator(pydantic_model)
        return chain

    def _create_validator(self, model: BaseModel):
        """Create validator for output parsing"""
        def validate_output(data: Dict) -> Dict:
            try:
                return model(**data).model_dump()
            except ValidationError as e:
                logger.error(f"Validation error: {e}")
                raise ValueError("LLM returned invalid response format") from e
        return validate_output

    def detect_request_type(self, user_message: str) -> Dict:
        """Determine if request is problem report or guidance request"""
        return self.procedure_chain.invoke({"user_message": user_message})

    def detect_problems(self, user_message: str, conversation_history2: str = "") -> Dict:
        return self.problem_detection_chain.invoke({
            "user_message": user_message,
            "conversation_history2": conversation_history2
        })
    def generate_guidance(self, procedure_topic: str, user_message: str) -> str:
        """Generate step-by-step guidance for procedures"""
        response = self.guidance_chain.invoke({
            "procedure_topic": procedure_topic,
            "user_message": user_message
        })
        return response.content if hasattr(response, 'content') else str(response)

    def classify_problem(
        self, 
        problem_text: str,
        conversation_history: List[Dict]
    ) -> Dict:
        """Classify a technical problem with conversation context"""
        return self.classification_chain.invoke({
            "problem_text": problem_text,
            "conversation_history": json.dumps(conversation_history)
        })
# --- MODIFIED: ConversationManager ---
class ConversationManager:
    def __init__(self, classifier: TechnicalSupportClassifier):
        self.classifier = classifier
        self.state = ConversationState()

    def reset_state(self):
        self.state = ConversationState()

    def handle_message(self, request_data: Dict) -> Dict[str, Any]:
        """Processes a user message request (now a dict) and returns a structured response."""

        # Check for special message types first
        if self.state.is_awaiting_problem_selection and request_data.get("type") == "problem_selection":
            return self._handle_problem_choice(request_data)

        user_message = request_data.get("message", "")
        if not user_message:
            return {"type": "error", "content": "No message content received."}

        if self.state.is_awaiting_followup:
            self.state.conversation_history.append({'role': 'user', 'content': user_message})
            self.state.is_awaiting_followup = False
            return self._classify_current_problem()

        # Handle as a new request
        self.reset_state()
        try:
            request_type = self.classifier.detect_request_type(user_message)
            if request_type["query_type"] == "guidance":
                guidance = self.classifier.generate_guidance(request_type["procedure_name"], user_message)
                return {"type": "guidance", "content": guidance}
            else:
                return self._process_problem_report(user_message)
        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            return {"type": "error", "content": "Sorry, I encountered an error. Please try again."}

    def _process_problem_report(self, user_message: str) -> Dict[str, Any]:
        """Identifies problems and either classifies or asks user to choose."""
        problem_detection = self.classifier.detect_problems(user_message, "")
        self.state.current_problems = problem_detection["problems"]

        if not self.state.current_problems:
            return {"type": "error", "content": "I couldn't identify a specific problem. Could you rephrase?"}

        if problem_detection["has_multiple_problems"]:
            self.state.is_awaiting_problem_selection = True
            return {
                "type": "multiple_problems_selection",
                "content": "I've found a few issues. Which one would you like to address first?",
                "problems": self.state.current_problems
            }
        else:
            self.state.current_problem_index = 0
            return self._classify_current_problem()

    def _handle_problem_choice(self, request_data: Dict) -> Dict[str, Any]:
        """Handles the user's choice of which problem to tackle first."""
        selected_id = request_data.get("selected_problem_id")

        # Reorder self.state.current_problems to put the selected one first
        selected_problem = next((p for p in self.state.current_problems if p['problem_id'] == selected_id), None)
        if selected_problem:
            other_problems = [p for p in self.state.current_problems if p['problem_id'] != selected_id]
            self.state.current_problems = [selected_problem] + other_problems

        self.state.is_awaiting_problem_selection = False
        self.state.current_problem_index = 0
        return self._classify_current_problem()

    def _classify_current_problem(self) -> Dict[str, Any]:
        """Classifies the problem at the current index, handling follow-ups and chaining."""
        if self.state.current_problem_index >= len(self.state.current_problems):
            self.reset_state()
            return {"type": "final", "content": "All issues have been addressed! How can I help you further?"}

        problem = self.state.current_problems[self.state.current_problem_index]

        problem_text = problem["current_user_problem"]
        if not self.state.conversation_history:
            self.state.conversation_history = [{"role": "user", "content": problem_text}]

        try:
            classification = self.classifier.classify_problem(problem_text, self.state.conversation_history)

            if classification['need_question']:
                question = classification['question']
                self.state.is_awaiting_followup = True
                self.state.conversation_history.append({'role': 'assistant', 'content': question})
                return {"type": "question", "content": question}
            else:
                category = classification['classification']
                response_text = f"For the issue '{problem_text}', I've classified it as: <b>{category.replace('_', ' ').title()}</b>."

                # Move to the next problem
                self.state.current_problem_index += 1
                self.state.conversation_history = []

                if self.state.current_problem_index < len(self.state.current_problems):
                    next_problem_message = self._classify_current_problem()
                    # Combine the classification result with the next step's message
                    response_text += "<br><br>" + next_problem_message['content']
                    # The type should reflect the final action (e.g., another question or final result)
                    return {"type": next_problem_message['type'], "content": response_text}
                else:
                    self.reset_state()
                    return {"type": "final", "content": response_text + "<br><br>All issues have been addressed. Feel free to ask anything else!"}

        except Exception as e:
            logger.error(f"Classification error: {e}", exc_info=True)
            return {"type": "error", "content": "I had trouble classifying the problem. Please try rephrasing."}

# --- Flask Application Setup ---
app = Flask(__name__)
try:
    classifier = TechnicalSupportClassifier(model="llama3-70b-8192")
    conversation_manager = ConversationManager(classifier)
except ValueError as e:
    logger.critical(f"Failed to initialize: {e}")
    classifier, conversation_manager = None, None

@app.route("/")
def index(): return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    if not conversation_manager: return jsonify({"type": "error", "content": "Chatbot not configured."}), 500
    request_data = request.json
    if not request_data: return jsonify({"type": "error", "content": "Invalid request."}), 400
    response = conversation_manager.handle_message(request_data)
    return jsonify(response)

@app.route("/reset", methods=["POST"])
def reset_chat():
    if conversation_manager:
        conversation_manager.reset_state()
        logger.info("Conversation state reset.")
        return jsonify({"status": "success"})
    return jsonify({"status": "error"}), 500



