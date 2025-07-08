import os
from typing import List, Dict, Any, Optional
from enum import Enum
import json
import re
import collections.abc

# Optional imports with graceful fallbacks
try:
    import openai
    # Check if we have the new OpenAI client (v1.0+)
    if hasattr(openai, 'OpenAI'):
        OPENAI_AVAILABLE = True
        print("✅ OpenAI v1.0+ detected")
    else:
        OPENAI_AVAILABLE = False
        print("⚠️ OpenAI version too old - please upgrade to v1.0+")
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not available - OpenAI provider disabled")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Generative AI not available - Gemini provider disabled")

try:
    import anthropic
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False
    print("⚠️ Anthropic not available - Claude provider disabled")

try:
    from llama_cpp import Llama
    LLAMA_AVAILABLE = True
except ImportError:
    LLAMA_AVAILABLE = False
    print("⚠️ llama-cpp not available - Local provider disabled")

# Prompt Templates for consistent, structured responses
PROMPT_TEMPLATES = {
    "question_answering": """
You are OmniDoc AI, an expert document analysis assistant. Your role is to provide accurate, well-referenced answers based on the provided document context.

CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Analyze the context carefully to find relevant information
2. Provide a clear, concise answer
3. Include specific document references (e.g., "Page 3, Section 2.1")
4. Explain your reasoning step-by-step
5. If the user requests, provide a layman/simple explanation or analogy in addition to the technical answer
6. If the context doesn't contain enough information, say so clearly

RESPONSE FORMAT:
Answer: [Your direct answer here]

References: [List specific citations like "Page X, Section Y"]

Reasoning: [Step-by-step explanation of how you arrived at your answer]

Confidence: [High/Medium/Low based on available context]
""",

    "challenge_generation": """
You are OmniDoc AI, an expert educator creating challenging questions based on document content.

CONTEXT:
{context}

KEY TERMS: {key_terms}

INSTRUCTIONS:
Generate 3 challenging questions that test:
1. Comprehension of key concepts
2. Critical thinking and analysis
3. Synthesis of information

Each question should:
- Be based on specific content from the document
- Require reasoning beyond simple recall
- Have a clear, correct answer
- Include an explanation of why the answer is correct

RESPONSE FORMAT:
Question 1: [Question text]
Correct Answer: [Detailed answer]
Explanation: [Why this answer is correct]
Reference: [Document location]

Question 2: [Question text]
Correct Answer: [Detailed answer]
Explanation: [Why this answer is correct]
Reference: [Document location]

Question 3: [Question text]
Correct Answer: [Detailed answer]
Explanation: [Why this answer is correct]
Reference: [Document location]
""",

    "answer_evaluation": """
You are OmniDoc AI, an expert evaluator assessing the quality of answers to challenge questions.

QUESTION: {question}
CORRECT ANSWER: {correct_answer}
USER ANSWER: {user_answer}
CONTEXT: {context}

INSTRUCTIONS:
Evaluate the user's answer based on:
1. Accuracy of information
2. Completeness of response
3. Understanding of concepts
4. Use of relevant terminology

RESPONSE FORMAT:
Is Correct: [Yes/No/Partial]
Score: [0-100]
Feedback: [Detailed feedback explaining strengths and areas for improvement]
Reasoning: [Step-by-step analysis of the evaluation]
""",

    "summary_generation": """
You are OmniDoc AI, an expert summarizer creating concise, informative summaries.

DOCUMENT CONTENT:
{content}

INSTRUCTIONS:
Create a comprehensive summary that includes:
1. Main topics and themes
2. Key findings or conclusions
3. Important methodologies or approaches
4. Significant data or results

RESPONSE FORMAT:
Summary: [Comprehensive summary in 2-3 paragraphs]

Key Points:
- [Point 1]
- [Point 2]
- [Point 3]

Document Type: [Research Paper/Technical Document/Report/etc.]
""",

    "key_concept_extraction": """
You are OmniDoc AI, an expert at identifying and extracting key concepts from documents.

DOCUMENT CONTENT:
{content}

INSTRUCTIONS:
Identify the most important concepts, terms, and ideas in this document. Focus on:
1. Technical terms and definitions
2. Key methodologies or approaches
3. Important findings or conclusions
4. Central themes or topics

RESPONSE FORMAT:
Key Concepts:
1. [Concept 1]: [Brief definition or explanation]
2. [Concept 2]: [Brief definition or explanation]
3. [Concept 3]: [Brief definition or explanation]

Technical Terms: [List of important technical terms]

Main Themes: [List of central themes or topics]
"""
}

class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    LOCAL = "local"

class LLMService:
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        if api_keys is None:
            api_keys = {}
        self.openai_client = None
        self.gemini_model = None
        self.claude_client = None
        self.local_model = None
        
        # Initialize providers with provided API keys
        self._initialize_providers(api_keys)
    
    def _initialize_providers(self, api_keys: Optional[Dict[str, str]] = None):
        """Initialize LLM providers with API keys from request or environment"""
        if api_keys is None:
            api_keys = {}
        # Use provided API keys or fall back to environment variables
        openai_api_key = api_keys.get('openai') if api_keys else os.getenv("OPENAI_API_KEY")
        google_api_key = api_keys.get('gemini') if api_keys else os.getenv("GOOGLE_API_KEY")
        anthropic_api_key = api_keys.get('claude') if api_keys else os.getenv("ANTHROPIC_API_KEY")
        llama_model_path = api_keys.get('local') if api_keys else os.getenv("LLAMA_MODEL_PATH")
        
        # OpenAI
        if openai_api_key and OPENAI_AVAILABLE:
            try:
                self.openai_client = openai.OpenAI(api_key=openai_api_key)
                print("✅ OpenAI initialized")
            except Exception as e:
                print(f"❌ OpenAI initialization failed: {e}")
        elif openai_api_key and not OPENAI_AVAILABLE:
            print("⚠️ OpenAI API key provided but OpenAI library not available")
        
        # Google Gemini
        if google_api_key and GEMINI_AVAILABLE:
            try:
                genai.configure(api_key=google_api_key)  # type: ignore
                # Use Gemini 1.5 Flash (or change to 'gemini-1.5-flash-lite' if needed)
                self.gemini_model = genai.GenerativeModel('gemini-1.5-flash')  # type: ignore
                print("✅ Google Gemini initialized")
            except Exception as e:
                print(f"❌ Google Gemini initialization failed: {e}")
        
        # Anthropic Claude
        if anthropic_api_key and CLAUDE_AVAILABLE:
            try:
                self.claude_client = anthropic.Anthropic(api_key=anthropic_api_key)
                print("✅ Anthropic Claude initialized")
            except Exception as e:
                print(f"❌ Anthropic Claude initialization failed: {e}")
        
        # Local LLM (llama-cpp)
        if llama_model_path and LLAMA_AVAILABLE and os.path.exists(llama_model_path):
            try:
                self.local_model = Llama(
                    model_path=llama_model_path,
                    n_ctx=2048,
                    n_threads=4
                )
                print("✅ Local LLM initialized")
            except Exception as e:
                print(f"❌ Local LLM initialization failed: {e}")

    def _parse_structured_response(self, response_text: str, response_type: str) -> Dict[str, Any]:
        """Parse structured response based on template format"""
        try:
            print(f"[Parser Debug] Parsing response_type: {response_type}")
            print(f"[Parser Debug] Raw response_text: {response_text}")
            if response_type == "question_answering":
                return self._parse_qa_response(response_text)
            elif response_type == "challenge_generation":
                return {"questions": self._parse_challenge_response(response_text)}
            elif response_type == "answer_evaluation":
                return self._parse_evaluation_response(response_text)
            elif response_type == "summary_generation":
                return self._parse_summary_response(response_text)
            elif response_type == "key_concept_extraction":
                return self._parse_concept_response(response_text)
            else:
                return {"raw_response": response_text}
        except Exception as e:
            import traceback
            print(f"[Parser Debug] Exception: {e}")
            traceback.print_exc()
            return {"answer": response_text, "reasoning": "(Could not parse response)", "references": "", "confidence": "Medium"}

    def _parse_qa_response(self, response_text: str) -> Dict[str, Any]:
        """Parse question-answering response"""
        try:
            print(f"[Parser Debug] _parse_qa_response input: {response_text}")
            answer_match = re.search(r'Answer:\s*(.+?)(?=\nReferences:|$)', response_text, re.DOTALL)
            references_match = re.search(r'References:\s*(.+?)(?=\nReasoning:|$)', response_text, re.DOTALL)
            reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\nConfidence:|$)', response_text, re.DOTALL)
            confidence_match = re.search(r'Confidence:\s*(.+?)(?=\n|$)', response_text, re.DOTALL)
            confidence_str = confidence_match.group(1).strip() if confidence_match else "Medium"
            # Map confidence string to float
            confidence_map = {"high": 0.9, "medium": 0.7, "low": 0.4}
            confidence_val = confidence_map.get(confidence_str.lower(), 0.7)
            return {
                "answer": answer_match.group(1).strip() if answer_match else response_text,
                "references": references_match.group(1).strip() if references_match else "",
                "reasoning": reasoning_match.group(1).strip() if reasoning_match else "",
                "confidence": confidence_val
            }
        except Exception as e:
            import traceback
            print(f"[Parser Debug] Exception in _parse_qa_response: {e}")
            traceback.print_exc()
            return {"answer": response_text, "reasoning": "(Could not parse QA response)", "references": "", "confidence": 0.7}

    def _parse_challenge_response(self, response_text: str) -> List[Dict[str, str]]:
        """Parse challenge generation response"""
        questions = []
        question_blocks = re.split(r'Question \d+:', response_text)[1:]  # Skip first empty element
        
        for block in question_blocks:
            question_match = re.search(r'(.+?)(?=Correct Answer:|$)', block, re.DOTALL)
            answer_match = re.search(r'Correct Answer:\s*(.+?)(?=Explanation:|$)', block, re.DOTALL)
            explanation_match = re.search(r'Explanation:\s*(.+?)(?=Reference:|$)', block, re.DOTALL)
            reference_match = re.search(r'Reference:\s*(.+?)(?=\n|$)', block, re.DOTALL)
            
            if question_match:
                questions.append({
                    "question": question_match.group(1).strip(),
                    "correct_answer": answer_match.group(1).strip() if answer_match else "",
                    "explanation": explanation_match.group(1).strip() if explanation_match else "",
                    "reference": reference_match.group(1).strip() if reference_match else ""
                })
        
        return questions

    def _parse_evaluation_response(self, response_text: str) -> Dict[str, Any]:
        """Parse answer evaluation response"""
        is_correct_match = re.search(r'Is Correct:\s*(.+?)(?=\nScore:|$)', response_text, re.DOTALL)
        score_match = re.search(r'Score:\s*(\d+)', response_text)
        feedback_match = re.search(r'Feedback:\s*(.+?)(?=\nReasoning:|$)', response_text, re.DOTALL)
        reasoning_match = re.search(r'Reasoning:\s*(.+?)(?=\n|$)', response_text, re.DOTALL)
        
        is_correct = is_correct_match.group(1).strip() if is_correct_match else "No"
        score = int(score_match.group(1)) if score_match else 0
        
        return {
            "is_correct": is_correct.lower() in ["yes", "correct", "true"],
            "score": score,
            "feedback": feedback_match.group(1).strip() if feedback_match else "",
            "reasoning": reasoning_match.group(1).strip() if reasoning_match else ""
        }

    def _parse_summary_response(self, response_text: str) -> Dict[str, Any]:
        """Parse summary generation response"""
        summary_match = re.search(r'Summary:\s*(.+?)(?=\nKey Points:|$)', response_text, re.DOTALL)
        key_points_match = re.search(r'Key Points:\s*(.+?)(?=\nDocument Type:|$)', response_text, re.DOTALL)
        doc_type_match = re.search(r'Document Type:\s*(.+?)(?=\n|$)', response_text, re.DOTALL)
        
        return {
            "summary": summary_match.group(1).strip() if summary_match else response_text,
            "key_points": key_points_match.group(1).strip() if key_points_match else "",
            "document_type": doc_type_match.group(1).strip() if doc_type_match else "Document"
        }

    def _parse_concept_response(self, response_text: str) -> Dict[str, Any]:
        """Parse key concept extraction response"""
        concepts_match = re.search(r'Key Concepts:\s*(.+?)(?=\nTechnical Terms:|$)', response_text, re.DOTALL)
        terms_match = re.search(r'Technical Terms:\s*(.+?)(?=\nMain Themes:|$)', response_text, re.DOTALL)
        themes_match = re.search(r'Main Themes:\s*(.+?)(?=\n|$)', response_text, re.DOTALL)
        
        return {
            "key_concepts": concepts_match.group(1).strip() if concepts_match else "",
            "technical_terms": terms_match.group(1).strip() if terms_match else "",
            "main_themes": themes_match.group(1).strip() if themes_match else ""
        }

    def _generate_response(self, prompt: str, provider: str, max_tokens: int = 1000) -> str:
        """Generate response using specified provider"""
        try:
            if provider == LLMProvider.OPENAI and self.openai_client:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.7
                )
                return response.choices[0].message.content or ""
            
            elif provider == LLMProvider.GEMINI and self.gemini_model:
                import traceback
                print(f"[Gemini Debug] Provider: {provider}")
                print(f"[Gemini Debug] Prompt: {prompt}")
                try:
                    response = self.gemini_model.generate_content([prompt])
                    print(f"[Gemini Debug] Raw response: {response}")
                    # Try to extract text from the new response structure
                    if hasattr(response, 'text') and response.text:
                        return response.text
                    # Newer API: candidates[0].content.parts[0].text
                    if hasattr(response, 'candidates') and response.candidates:
                        parts = response.candidates[0].content.parts
                        if parts and hasattr(parts[0], 'text'):
                            return parts[0].text
                    # Fallback: return string representation
                    return str(response)
                except Exception as gemini_exc:
                    print(f"[Gemini Debug] Exception: {gemini_exc}")
                    traceback.print_exc()
                    print(f"[Gemini Debug] Gemini raw response: {locals().get('response', None)}")
                    return f"Error: Unable to generate response with gemini. Please try again."
            
            elif provider == LLMProvider.CLAUDE and self.claude_client:
                response = self.claude_client.messages.create(
                    model="claude-3-sonnet-20240229",
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}]
                )
                return getattr(response.content[0], "text", "")  # type: ignore
            
            elif provider == LLMProvider.LOCAL and self.local_model:
                response = self.local_model(
                    prompt,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    stop=["\n\n", "Human:", "Assistant:"]
                )
                choices = response["choices"]  # type: ignore
                if isinstance(choices, list):
                    first_choice = choices[0] if choices else None
                else:
                    first_choice = next(iter(choices), None)
                return first_choice.get("text", "") if isinstance(first_choice, dict) else ""
            
            else:
                raise ValueError(f"Provider {provider} not available or not initialized")
                
        except Exception as e:
            print(f"Error generating response with {provider}: {e}")
            return f"Error: Unable to generate response with {provider}. Please try again."

    def ask_question(self, question: str, context: List[str], provider: str = "openai") -> Dict[str, Any]:
        """Ask a question with structured response"""
        # Prepare context
        context_text = "\n\n".join(context[:5])  # Limit to 5 chunks
        
        # Use structured prompt template
        prompt = PROMPT_TEMPLATES["question_answering"].format(
            context=context_text,
            question=question
        )
        
        # Generate response
        response_text = self._generate_response(prompt, provider, max_tokens=1500)
        
        # Parse structured response
        parsed_response = self._parse_structured_response(response_text, "question_answering")
        
        return {
            "answer": parsed_response.get("answer", response_text),
            "reasoning": parsed_response.get("reasoning", ""),
            "confidence": self._calculate_confidence(parsed_response.get("confidence", "Medium")),
            "references": parsed_response.get("references", ""),
            "raw_response": response_text
        }

    def generate_challenge_questions(self, context: List[str], provider: str = "openai", key_terms: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """Generate challenge questions with structured response"""
        # Prepare context
        context_text = "\n\n".join(context[:10])  # Use more chunks for question generation
        key_terms_text = ", ".join(key_terms[:10]) if key_terms else "key concepts from the document"
        
        # Use structured prompt template
        prompt = PROMPT_TEMPLATES["challenge_generation"].format(
            context=context_text,
            key_terms=key_terms_text
        )
        
        # Generate response
        response_text = self._generate_response(prompt, provider, max_tokens=2000)
        
        # Parse structured response
        questions = self._parse_structured_response(response_text, "challenge_generation")
        
        return questions if isinstance(questions, list) else []

    def evaluate_challenge_answer(self, question: str, correct_answer: str, user_answer: str, 
                                context: str, provider: str = "openai") -> Dict[str, Any]:
        """Evaluate challenge answer with structured response"""
        # Use structured prompt template
        prompt = PROMPT_TEMPLATES["answer_evaluation"].format(
            question=question,
            correct_answer=correct_answer,
            user_answer=user_answer,
            context=context
        )
        
        # Generate response
        response_text = self._generate_response(prompt, provider, max_tokens=1000)
        
        # Parse structured response
        evaluation = self._parse_structured_response(response_text, "answer_evaluation")
        
        return evaluation

    def generate_summary(self, content: str, provider: str = "openai") -> Dict[str, Any]:
        """Generate document summary with structured response"""
        # Use structured prompt template
        prompt = PROMPT_TEMPLATES["summary_generation"].format(content=content)
        
        # Generate response
        response_text = self._generate_response(prompt, provider, max_tokens=1000)
        
        # Parse structured response
        summary = self._parse_structured_response(response_text, "summary_generation")
        
        return summary

    def extract_key_concepts(self, content: str, provider: str = "openai") -> Dict[str, Any]:
        """Extract key concepts with structured response"""
        # Use structured prompt template
        prompt = PROMPT_TEMPLATES["key_concept_extraction"].format(content=content)
        
        # Generate response
        response_text = self._generate_response(prompt, provider, max_tokens=1000)
        
        # Parse structured response
        concepts = self._parse_structured_response(response_text, "key_concept_extraction")
        
        return concepts

    def _calculate_confidence(self, confidence_text) -> float:
        """Convert confidence text to numerical score"""
        confidence_lower = str(confidence_text).lower()
        if "high" in confidence_lower:
            return 0.9
        elif "medium" in confidence_lower:
            return 0.7
        elif "low" in confidence_lower:
            return 0.4
        else:
            return 0.7  # Default to medium confidence

    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        providers = []
        if self.openai_client:
            providers.append("openai")
        if self.gemini_model:
            providers.append("gemini")
        if self.claude_client:
            providers.append("claude")
        if self.local_model:
            providers.append("local")
        return providers

    def test_provider(self, provider: str) -> Dict[str, Any]:
        """Test a specific provider"""
        try:
            test_prompt = "Please respond with 'Hello from OmniDoc AI' to test the connection."
            response = self._generate_response(test_prompt, provider, max_tokens=50)
            
            return {
                "provider": provider,
                "status": "success",
                "response": response,
                "available": True
            }
        except Exception as e:
            return {
                "provider": provider,
                "status": "error",
                "error": str(e),
                "available": False
            } 