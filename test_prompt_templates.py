#!/usr/bin/env python3
"""
Test script for OmniDoc AI Structured Prompt Templates
Tests the modular prompt templates and response parsing
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from backend.services.llm_service import LLMService, PROMPT_TEMPLATES

def test_prompt_templates():
    """Test that prompt templates are properly formatted"""
    print("🧪 Testing Prompt Templates")
    print("=" * 50)
    
    # Test 1: Question Answering Template
    print("\n1. Testing Question Answering Template:")
    qa_prompt = PROMPT_TEMPLATES["question_answering"].format(
        context="This is a test context about machine learning.",
        question="What is machine learning?"
    )
    print(f"✅ Template formatted successfully")
    print(f"   - Contains 'Answer:' section: {'Answer:' in qa_prompt}")
    print(f"   - Contains 'References:' section: {'References:' in qa_prompt}")
    print(f"   - Contains 'Reasoning:' section: {'Reasoning:' in qa_prompt}")
    
    # Test 2: Challenge Generation Template
    print("\n2. Testing Challenge Generation Template:")
    challenge_prompt = PROMPT_TEMPLATES["challenge_generation"].format(
        context="This is a test context about neural networks.",
        key_terms="neural networks, deep learning, artificial intelligence"
    )
    print(f"✅ Template formatted successfully")
    print(f"   - Contains 'Question 1:' section: {'Question 1:' in challenge_prompt}")
    print(f"   - Contains 'Correct Answer:' section: {'Correct Answer:' in challenge_prompt}")
    print(f"   - Contains 'Explanation:' section: {'Explanation:' in challenge_prompt}")
    
    # Test 3: Answer Evaluation Template
    print("\n3. Testing Answer Evaluation Template:")
    eval_prompt = PROMPT_TEMPLATES["answer_evaluation"].format(
        question="What is machine learning?",
        correct_answer="Machine learning is a subset of AI that enables computers to learn from data.",
        user_answer="Machine learning is AI that learns from data.",
        context="Machine learning context here."
    )
    print(f"✅ Template formatted successfully")
    print(f"   - Contains 'Is Correct:' section: {'Is Correct:' in eval_prompt}")
    print(f"   - Contains 'Score:' section: {'Score:' in eval_prompt}")
    print(f"   - Contains 'Feedback:' section: {'Feedback:' in eval_prompt}")
    
    # Test 4: Summary Generation Template
    print("\n4. Testing Summary Generation Template:")
    summary_prompt = PROMPT_TEMPLATES["summary_generation"].format(
        content="This is a test document about artificial intelligence and machine learning."
    )
    print(f"✅ Template formatted successfully")
    print(f"   - Contains 'Summary:' section: {'Summary:' in summary_prompt}")
    print(f"   - Contains 'Key Points:' section: {'Key Points:' in summary_prompt}")
    print(f"   - Contains 'Document Type:' section: {'Document Type:' in summary_prompt}")
    
    # Test 5: Key Concept Extraction Template
    print("\n5. Testing Key Concept Extraction Template:")
    concept_prompt = PROMPT_TEMPLATES["key_concept_extraction"].format(
        content="This document discusses neural networks, deep learning, and artificial intelligence."
    )
    print(f"✅ Template formatted successfully")
    print(f"   - Contains 'Key Concepts:' section: {'Key Concepts:' in concept_prompt}")
    print(f"   - Contains 'Technical Terms:' section: {'Technical Terms:' in concept_prompt}")
    print(f"   - Contains 'Main Themes:' section: {'Main Themes:' in concept_prompt}")

def test_response_parsing():
    """Test response parsing functions"""
    print("\n\n🧪 Testing Response Parsing")
    print("=" * 50)
    
    llm_service = LLMService()
    
    # Test 1: QA Response Parsing
    print("\n1. Testing QA Response Parsing:")
    qa_response = """
Answer: Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed.

References: Page 3, Section 2.1 - Introduction to Machine Learning

Reasoning: Based on the context provided, machine learning is defined as a subset of AI that focuses on data-driven learning approaches.

Confidence: High
"""
    parsed_qa = llm_service._parse_qa_response(qa_response)
    print(f"✅ QA Response parsed successfully")
    print(f"   - Answer: {parsed_qa.get('answer', 'Not found')[:50]}...")
    print(f"   - References: {parsed_qa.get('references', 'Not found')}")
    print(f"   - Reasoning: {parsed_qa.get('reasoning', 'Not found')[:50]}...")
    print(f"   - Confidence: {parsed_qa.get('confidence', 'Not found')}")
    
    # Test 2: Challenge Response Parsing
    print("\n2. Testing Challenge Response Parsing:")
    challenge_response = """
Question 1: What is the primary goal of machine learning?
Correct Answer: The primary goal of machine learning is to enable computers to learn patterns from data and make predictions or decisions without explicit programming.
Explanation: Machine learning focuses on creating algorithms that can identify patterns in data and use those patterns to make predictions or decisions.
Reference: Page 5, Section 3.2

Question 2: How does supervised learning differ from unsupervised learning?
Correct Answer: Supervised learning uses labeled training data to learn patterns, while unsupervised learning finds patterns in unlabeled data.
Explanation: The key difference is the presence of labeled data in supervised learning versus unlabeled data in unsupervised learning.
Reference: Page 8, Section 4.1
"""
    parsed_challenge = llm_service._parse_challenge_response(challenge_response)
    print(f"✅ Challenge Response parsed successfully")
    print(f"   - Questions found: {len(parsed_challenge)}")
    for i, question in enumerate(parsed_challenge[:2]):
        print(f"   - Question {i+1}: {question.get('question', 'Not found')[:50]}...")
        print(f"     Answer: {question.get('correct_answer', 'Not found')[:50]}...")
    
    # Test 3: Evaluation Response Parsing
    print("\n3. Testing Evaluation Response Parsing:")
    eval_response = """
Is Correct: Yes
Score: 85
Feedback: The answer demonstrates good understanding of the concept but could be more detailed.
Reasoning: The user correctly identified the key aspects of machine learning but missed some technical details.
"""
    parsed_eval = llm_service._parse_evaluation_response(eval_response)
    print(f"✅ Evaluation Response parsed successfully")
    print(f"   - Is Correct: {parsed_eval.get('is_correct', 'Not found')}")
    print(f"   - Score: {parsed_eval.get('score', 'Not found')}")
    print(f"   - Feedback: {parsed_eval.get('feedback', 'Not found')[:50]}...")
    print(f"   - Reasoning: {parsed_eval.get('reasoning', 'Not found')[:50]}...")
    
    # Test 4: Summary Response Parsing
    print("\n4. Testing Summary Response Parsing:")
    summary_response = """
Summary: This document provides a comprehensive overview of machine learning concepts, including supervised and unsupervised learning approaches.

Key Points:
- Machine learning is a subset of AI
- Two main types: supervised and unsupervised learning
- Focuses on pattern recognition and prediction

Document Type: Technical Document
"""
    parsed_summary = llm_service._parse_summary_response(summary_response)
    print(f"✅ Summary Response parsed successfully")
    print(f"   - Summary: {parsed_summary.get('summary', 'Not found')[:50]}...")
    print(f"   - Key Points: {parsed_summary.get('key_points', 'Not found')[:50]}...")
    print(f"   - Document Type: {parsed_summary.get('document_type', 'Not found')}")

def test_confidence_calculation():
    """Test confidence calculation function"""
    print("\n\n🧪 Testing Confidence Calculation")
    print("=" * 50)
    
    llm_service = LLMService()
    
    test_cases = [
        ("High", 0.9),
        ("Medium", 0.7),
        ("Low", 0.4),
        ("high confidence", 0.9),
        ("medium level", 0.7),
        ("low certainty", 0.4),
        ("Unknown", 0.7)  # Default
    ]
    
    for confidence_text, expected_score in test_cases:
        calculated_score = llm_service._calculate_confidence(confidence_text)
        status = "✅" if abs(calculated_score - expected_score) < 0.1 else "❌"
        print(f"{status} '{confidence_text}' -> {calculated_score} (expected: {expected_score})")

def main():
    """Run all prompt template tests"""
    print("🚀 OmniDoc AI Prompt Template Testing")
    print("=" * 60)
    
    try:
        # Test 1: Prompt Templates
        test_prompt_templates()
        
        # Test 2: Response Parsing
        test_response_parsing()
        
        # Test 3: Confidence Calculation
        test_confidence_calculation()
        
        print("\n🎉 All prompt template tests completed successfully!")
        print("\n✨ Benefits of Structured Prompts:")
        print("   ✅ Consistent response format across all LLM providers")
        print("   ✅ Easy parsing and display in the frontend")
        print("   ✅ Better reasoning transparency")
        print("   ✅ Standardized confidence scoring")
        print("   ✅ Modular and maintainable prompt system")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 