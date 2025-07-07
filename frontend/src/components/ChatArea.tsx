import React, { useState, useRef, useEffect } from 'react';
import { Send, Upload, ToggleLeft, ToggleRight, Eye, EyeOff } from 'lucide-react';
import { useApp } from '../context/AppContext';
import FileUpload from './FileUpload';
import Message from './Message';
import { apiService, DocumentReference } from '../services/api';

interface ReasoningStep {
  step_id: string;
  step_type: string;
  description: string;
  input_data: any;
  output_data: any;
  confidence: number;
  timestamp: string;
}

const CHAT_HISTORY_KEY = 'omnidoc_chat_history';
const CHAT_SESSION_KEY = 'omnidoc_chat_session_id';

function uuidv4() {
  return ("10000000-1000-4000-8000-100000000000").replace(/[018]/g, (c: string) =>
    (
      parseInt(c, 10) ^ (crypto.getRandomValues(new Uint8Array(1))[0] & (15 >> (parseInt(c, 10) / 4)))
    ).toString(16)
  );
}

const ChatArea: React.FC = () => {
  const [input, setInput] = useState('');
  const [isDragOver, setIsDragOver] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const {
    messages,
    addMessage,
    chatMode,
    setChatMode,
    selectedDocument,
    documents,
    isUploading,
    selectedProvider,
    apiKeys
  } = useApp();

  // Store document references and reasoning chains for each message
  const [messageReferences, setMessageReferences] = useState<Record<string, DocumentReference[]>>({});
  const [messageReasoningChains, setMessageReasoningChains] = useState<Record<string, ReasoningStep[]>>({});
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Interactive challenge mode state
  const [currentChallengeQuestion, setCurrentChallengeQuestion] = useState<any>(null);
  const [challengeQuestionCount, setChallengeQuestionCount] = useState(0);
  const [waitingForAnswer, setWaitingForAnswer] = useState(false);
  const [selectedChallengeDocument, setSelectedChallengeDocument] = useState<string | null>(null);

  const [lastMessageId, setLastMessageId] = useState<string | null>(null);

  const [sessionId, setSessionId] = useState<string>(() => {
    const saved = localStorage.getItem(CHAT_SESSION_KEY);
    if (saved) return saved;
    const newId = uuidv4();
    localStorage.setItem(CHAT_SESSION_KEY, newId);
    return newId;
  });

  // Restore chat history and challenge state on mount
  useEffect(() => {
    const saved = localStorage.getItem(CHAT_HISTORY_KEY);
    if (saved) {
      const parsed = JSON.parse(saved);
      if (parsed[sessionId]) {
        // Restore messages and challenge state
        if (parsed[sessionId].messages) {
          parsed[sessionId].messages.forEach((msg: any) => addMessage(msg));
        }
        if (parsed[sessionId].challengeQuestionCount) {
          setChallengeQuestionCount(parsed[sessionId].challengeQuestionCount);
        }
        if (parsed[sessionId].currentChallengeQuestion) {
          setCurrentChallengeQuestion(parsed[sessionId].currentChallengeQuestion);
        }
        if (parsed[sessionId].waitingForAnswer) {
          setWaitingForAnswer(parsed[sessionId].waitingForAnswer);
        }
        if (parsed[sessionId].selectedChallengeDocument) {
          setSelectedChallengeDocument(parsed[sessionId].selectedChallengeDocument);
        }
      }
    }
  }, [sessionId]);

  // Save chat history and challenge state on change
  useEffect(() => {
    const saved = localStorage.getItem(CHAT_HISTORY_KEY);
    const parsed = saved ? JSON.parse(saved) : {};
    parsed[sessionId] = {
      messages,
      challengeQuestionCount,
      currentChallengeQuestion,
      waitingForAnswer,
      selectedChallengeDocument
    };
    localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(parsed));
  }, [messages, challengeQuestionCount, currentChallengeQuestion, waitingForAnswer, selectedChallengeDocument, sessionId]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  // Only scroll to bottom when a new message is added (last message ID changes)
  useEffect(() => {
    if (messages.length === 0) return;
    const newLastId = messages[messages.length - 1]?.id;
    if (lastMessageId !== newLastId) {
      scrollToBottom();
      setLastMessageId(newLastId);
    }
  }, [messages.length]);

  const handleSendMessage = async () => {
    if (!input.trim() || isLoading) {
      console.log('Message send blocked:', { input: input.trim(), isLoading });
      return;
    }

    // Validate setup before proceeding
    const selectedApiKey = apiKeys[selectedProvider];
    if (!selectedApiKey || selectedApiKey.trim() === '') {
    addMessage({
      type: 'user',
        content: input.trim()
    });
      addMessage({
        type: 'ai',
        content: 'API key is required. Please select your LLM provider and enter your API key in the sidebar.',
        reasoning: 'API key validation failed.'
      });
      setInput('');
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setIsLoading(true);

    try {
      if (chatMode.type === 'challenge') {
        // Interactive Challenge Mode
        if (waitingForAnswer && currentChallengeQuestion) {
          // User is answering a question
          setWaitingForAnswer(false);
          
          // Evaluate the user's answer first
          const evaluation = await apiService.evaluateChallengeAnswerInteractive(
            currentChallengeQuestion.id,
            userMessage,
            currentChallengeQuestion.correct_answer,
            currentChallengeQuestion.question,
            currentChallengeQuestion.document_references?.[0]?.chunk_preview || '',
            selectedProvider,
            apiKeys
          );

          // Add user message and evaluation together
          addMessage({
            type: 'user',
            content: userMessage
          });

          // Add evaluation message
          const evaluationMessage = `**Answer Evaluation:**

**Your Answer:** ${userMessage}

**Correct Answer:** ${currentChallengeQuestion.correct_answer}

**Score:** ${evaluation.score}/100
**Similarity Score:** ${(evaluation.similarity_score * 100).toFixed(1)}%
**Keyword Overlap:** ${(evaluation.keyword_overlap * 100).toFixed(1)}%
**Overall Score:** ${(evaluation.comprehensive_score * 100).toFixed(1)}%

**Feedback:** ${evaluation.feedback}

**Explanation:** ${evaluation.reasoning}

**Correct Answer Details:** ${currentChallengeQuestion.explanation}`;

          const messageId = addMessage({
            type: 'ai',
            content: evaluationMessage,
            reasoning: evaluation.reasoning
          });

          // Store document references
          if (messageId) {
            setMessageReferences(prev => ({
              ...prev,
              [messageId]: currentChallengeQuestion.document_references || []
            }));
          }

          // Ask if user wants another question
          const nextQuestionMessage = `Great! You've completed question ${challengeQuestionCount}. Would you like another challenge question? Just say "yes" or "next" to continue, or "no" to end the challenge.`;

          addMessage({
            type: 'ai',
            content: nextQuestionMessage,
            reasoning: 'Asking user if they want to continue with more challenge questions.'
          });

          setCurrentChallengeQuestion(null);
          // Do NOT increment here - wait for the next question

        } else if (userMessage.toLowerCase().includes('yes') || userMessage.toLowerCase().includes('next') || userMessage.toLowerCase().includes('continue')) {
          // User wants another question
          if (documents.length === 0) {
            addMessage({
              type: 'user',
              content: userMessage
            });
            addMessage({
              type: 'ai',
              content: 'Please upload at least one document to generate challenge questions.',
              reasoning: 'No documents available for challenge generation.'
            });
            return;
          }

          // Generate next question
          const documentIds = selectedChallengeDocument 
            ? [selectedChallengeDocument] 
            : documents.map(doc => doc.id);
          const nextQuestion = await apiService.getNextChallengeQuestion(
            documentIds,
            selectedProvider,
            apiKeys
          );
          setCurrentChallengeQuestion(nextQuestion);
          setWaitingForAnswer(true);
          setChallengeQuestionCount(prev => prev + 1); // Increment here
          // Add user message and question together
          addMessage({
            type: 'user',
            content: userMessage
          });
          // Only show the challenge mode activated message if this is the first question
          let questionMessage;
          if (challengeQuestionCount === 0) {
            questionMessage = `🎯 **Interactive Challenge Mode Activated!**\n\nI'll ask you questions one by one based on "${selectedChallengeDocument ? documents.find(doc => doc.id === selectedChallengeDocument)?.name : documents[0]?.name}", and after each answer, I'll provide detailed feedback and evaluation.\n\n**Challenge Question 1:**\n\n${nextQuestion.question}\n\nPlease provide your answer below. I'll evaluate it and give you detailed feedback!`;
          } else {
            questionMessage = `**Challenge Question ${challengeQuestionCount + 1}:**\n\n${nextQuestion.question}\n\nPlease provide your answer below. I'll evaluate it and give you detailed feedback!`;
          }
          const messageId = addMessage({
            type: 'ai',
            content: questionMessage,
            reasoning: 'Generated a new challenge question for interactive learning.'
          });
          if (messageId) {
            setMessageReferences(prev => ({
              ...prev,
              [messageId]: nextQuestion.document_references || []
            }));
          }
        } else if (userMessage.toLowerCase().includes('no') || userMessage.toLowerCase().includes('stop') || userMessage.toLowerCase().includes('end')) {
          // User wants to end the challenge
          addMessage({
            type: 'user',
            content: userMessage
          });
          
          const endMessage = `Challenge completed! You answered ${challengeQuestionCount} question(s). 

**Challenge Summary:**
- Questions attempted: ${challengeQuestionCount}
- Interactive learning mode: ✅
- Detailed feedback provided: ✅

Great job engaging with the material! You can switch back to "Ask Anything" mode or start a new challenge anytime.`;

          addMessage({
            type: 'ai',
            content: endMessage,
            reasoning: 'User ended the challenge session.'
          });

          // Reset challenge state
          setCurrentChallengeQuestion(null);
          setChallengeQuestionCount(0);
          setWaitingForAnswer(false);

        } else {
          // Check if this is a document selection response
          if (documents.length > 1 && !selectedChallengeDocument && !waitingForAnswer) {
            // Try to match document selection
            const userInput = userMessage.toLowerCase().trim();
            let selectedDoc = null;
            
            // Check for number selection
            const numberMatch = userInput.match(/^(\d+)$/);
            if (numberMatch) {
              const docIndex = parseInt(numberMatch[1]) - 1;
              if (docIndex >= 0 && docIndex < documents.length) {
                selectedDoc = documents[docIndex];
              }
            } else {
              // Check for name selection
              selectedDoc = documents.find(doc => 
                doc.name.toLowerCase().includes(userInput) || 
                userInput.includes(doc.name.toLowerCase())
              );
            }
            
            if (selectedDoc) {
              setSelectedChallengeDocument(selectedDoc.id);
              addMessage({
                type: 'user',
                content: userMessage
              });
              
              const confirmMessage = `Perfect! I'll generate challenge questions from "${selectedDoc.name}". Let me start the interactive challenge mode...`;
              addMessage({
                type: 'ai',
                content: confirmMessage,
                reasoning: 'User selected a specific document for challenge questions.'
              });
              
              // Now start the challenge with the selected document
              const firstQuestion = await apiService.getNextChallengeQuestion(
                [selectedDoc.id],
                selectedProvider,
                apiKeys
              );

              setCurrentChallengeQuestion(firstQuestion);
              setWaitingForAnswer(true);
              setChallengeQuestionCount(1);

              // Add welcome message and first question
              const welcomeMessage = `🎯 **Interactive Challenge Mode Activated!**\n\nI'll ask you questions one by one based on "${selectedDoc.name}", and after each answer, I'll provide detailed feedback and evaluation.\n\n**Challenge Question 1:**\n\n${firstQuestion.question}\n\nPlease provide your answer below. I'll evaluate it and give you comprehensive feedback!`;
              const messageId = addMessage({
                type: 'ai',
                content: welcomeMessage,
                reasoning: 'Started interactive challenge mode with first question from selected document.'
              });

              // Store document references
              if (messageId) {
                setMessageReferences(prev => ({
                  ...prev,
                  [messageId]: firstQuestion.document_references || []
                }));
              }
              return;
            } else {
              addMessage({
                type: 'user',
                content: userMessage
              });
              addMessage({
                type: 'ai',
                content: 'I couldn\'t find that document. Please try again with the document number or name.',
                reasoning: 'Invalid document selection.'
              });
              return;
    }
          }

          // First time entering challenge mode
          if (documents.length === 0) {
            addMessage({
              type: 'user',
              content: userMessage
            });
            addMessage({
              type: 'ai',
              content: 'Please upload at least one document to generate challenge questions.',
              reasoning: 'No documents available for challenge generation.'
            });
            return;
          }

          // Check if multiple documents and no specific document selected
          if (documents.length > 1 && !selectedChallengeDocument) {
            addMessage({
              type: 'user',
              content: userMessage
            });
            
            const documentList = documents.map((doc, index) => `${index + 1}. ${doc.name}`).join('\n');
            const selectionMessage = `I see you have multiple documents uploaded. Which document would you like me to generate challenge questions from?

**Available Documents:**
${documentList}

Please respond with the document number (1, 2, 3, etc.) or the document name.`;

            addMessage({
              type: 'ai',
              content: selectionMessage,
              reasoning: 'Prompting user to select a specific document for challenge questions.'
            });
            return;
          }

          // Start interactive challenge first
          const documentIds = selectedChallengeDocument 
            ? [selectedChallengeDocument] 
            : documents.map(doc => doc.id);
          const firstQuestion = await apiService.getNextChallengeQuestion(
            documentIds,
            selectedProvider,
            apiKeys
          );

          setCurrentChallengeQuestion(firstQuestion);
          setWaitingForAnswer(true);
          setChallengeQuestionCount(1);

          // Add user message and welcome message together
          addMessage({
            type: 'user',
            content: userMessage
          });

          // Only show the challenge mode activated message if this is the first question
          const welcomeMessage = `🎯 **Interactive Challenge Mode Activated!**\n\nI'll ask you questions one by one, and after each answer, I'll provide detailed feedback and evaluation.\n\n**Challenge Question 1:**\n\n${firstQuestion.question}\n\nPlease provide your answer below. I'll evaluate it and give you comprehensive feedback!`;
          const messageId = addMessage({
            type: 'ai',
            content: welcomeMessage,
            reasoning: 'Started interactive challenge mode with first question.'
          });

          // Store document references
          if (messageId) {
            setMessageReferences(prev => ({
              ...prev,
              [messageId]: firstQuestion.document_references || []
            }));
          }
        }
      } else {
        // Ask question about documents
        if (documents.length === 0) {
          addMessage({
            type: 'user',
            content: userMessage
          });
          addMessage({
            type: 'ai',
            content: 'Please upload a document first to ask questions about it.',
            reasoning: 'No documents available for analysis.'
          });
          return;
        }

        // Make API call first
        const response = await apiService.askQuestion(
          userMessage,
          'ask',
          selectedDocument || undefined,
          selectedProvider,
          apiKeys
        );

        // Add user message and AI response together
        addMessage({
          type: 'user',
          content: userMessage
        });

        // Add the message and get its ID
        const messageId = addMessage({
          type: 'ai',
          content: response.message.content,
          documentRef: response.document_references.length > 0 ? 
            response.document_references[0].citation : undefined,
          reasoning: response.reasoning
        });

        // Store document references
        if (messageId) {
          setMessageReferences(prev => ({
            ...prev,
            [messageId]: response.document_references
          }));

          // Store reasoning chain if available
          setMessageReasoningChains(prev => ({
            ...prev,
            [messageId]: response.reasoning_chain ?? []
          }));
        }
      }
    } catch (error) {
      console.error('Error getting AI response:', error);
      addMessage({
        type: 'ai',
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`,
        reasoning: 'An error occurred while processing your request.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => 
      file.type === 'application/pdf' || file.type === 'text/plain'
    );
    
    if (validFiles.length > 0) {
      // Handle file upload
      console.log('Files dropped:', validFiles);
    }
  };

  const handleModeToggle = () => {
    // Reset challenge state when switching modes
    if (chatMode.type === 'challenge') {
      setCurrentChallengeQuestion(null);
      setChallengeQuestionCount(0);
      setWaitingForAnswer(false);
      setSelectedChallengeDocument(null);
    }
    
    setChatMode(
      chatMode.type === 'ask' 
        ? { type: 'challenge', label: 'Challenge Me' }
        : { type: 'ask', label: 'Ask Anything' }
    );
  };

  return (
    <div className="flex flex-col h-full bg-[#263238] rounded-lg border border-[#37474F] overflow-hidden">
      {/* Chat Messages */}
      <div 
        className={`flex-1 overflow-y-auto p-4 space-y-4 scrollbar-thin scrollbar-thumb-[#4FC3F7] scrollbar-track-[#1A2C32] min-h-0 ${isDragOver ? 'border-2 border-dashed border-[#4FC3F7] bg-[#1A2C32]' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        style={{
          scrollbarWidth: 'thin',
          scrollbarColor: '#4FC3F7 #1A2C32'
        }}
      >
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center space-y-4">
            <div className="w-16 h-16 bg-[#4FC3F7] rounded-full flex items-center justify-center">
              <Upload size={32} className="text-white" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-[#ECEFF1] mb-2">Welcome to OmniDoc AI</h3>
              <p className="text-[#90A4AE] max-w-md">
                Upload a PDF or TXT document to get started. I'll analyze it with advanced semantic processing and provide intelligent Q&A with detailed reasoning chains.
              </p>
            </div>
            
            {/* Setup Warning */}
            {(!apiKeys[selectedProvider] || apiKeys[selectedProvider].trim() === '') && (
              <div className="w-full max-w-md p-3 bg-yellow-900/20 border border-yellow-600/50 rounded-lg">
                <div className="flex items-center gap-2 mb-2">
                  <div className="w-2 h-2 bg-yellow-400 rounded-full" />
                  <span className="text-sm font-medium text-yellow-400">Setup Required</span>
                </div>
                <p className="text-xs text-yellow-300">
                  Please select your LLM provider and enter your API key in the sidebar to upload files and ask questions.
                </p>
              </div>
            )}
            
            <FileUpload />
          </div>
        ) : (
          messages.map((message) => (
            <Message 
              key={message.id} 
              message={message}
              documentReferences={messageReferences[message.id] || []}
              reasoningChain={messageReasoningChains[message.id] || []}
            />
          ))
        )}
        
        {isUploading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#4FC3F7]"></div>
            <span className="ml-3 text-[#90A4AE]">Processing document with semantic analysis...</span>
          </div>
        )}

        {isLoading && (
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-[#4FC3F7]"></div>
            <span className="ml-3 text-[#90A4AE]">Analyzing with hybrid search and reasoning chain...</span>
          </div>
        )}
        
        {/* Scroll anchor */}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div className="border-t border-[#37474F] p-4 space-y-3">
        {/* Mode Toggle */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-sm text-[#90A4AE]">Mode:</span>
            <button
              onClick={handleModeToggle}
              className="flex items-center gap-2 px-3 py-1 bg-[#37474F] rounded-lg border border-[#455A64] hover:bg-[#455A64] transition-colors"
              disabled={isLoading}
            >
              {chatMode.type === 'ask' ? (
                <ToggleLeft size={16} className="text-[#4FC3F7]" />
              ) : (
                <ToggleRight size={16} className="text-[#4FC3F7]" />
              )}
              <span className="text-sm text-[#ECEFF1]">{chatMode.label}</span>
            </button>
          </div>
          
          <FileUpload />
        </div>

        {/* Message Input */}
        <div className="flex gap-2">
          <div className="flex-1 relative">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder={
                chatMode.type === 'ask' 
                  ? "Ask anything about your document with detailed reasoning chains..." 
                  : waitingForAnswer 
                    ? "Type your answer to the challenge question..."
                    : "Start an interactive challenge! I'll ask questions one by one and evaluate your answers."
              }
              className="w-full px-4 py-3 bg-[#37474F] text-[#ECEFF1] rounded-lg border border-[#455A64] focus:outline-none focus:ring-2 focus:ring-[#4FC3F7] focus:border-transparent resize-none"
              rows={2}
              disabled={isUploading || isLoading}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!input.trim() || isUploading || isLoading}
            className={`px-4 py-2 rounded-lg transition-colors ${
              !input.trim() || isUploading || isLoading
                ? 'bg-[#4FC3F7] text-white opacity-50 cursor-not-allowed'
                : !apiKeys[selectedProvider] || apiKeys[selectedProvider].trim() === ''
                  ? 'bg-yellow-600 text-white hover:bg-yellow-700'
                  : 'bg-[#4FC3F7] text-white hover:bg-[#29B6F6]'
            }`}
            title={
              !input.trim() || isUploading || isLoading
                ? "Cannot send empty message"
                : !apiKeys[selectedProvider] || apiKeys[selectedProvider].trim() === ''
                  ? "API key is required. Please select your LLM provider and enter your API key in the sidebar."
                  : "Send message"
            }
          >
            <Send size={20} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default ChatArea;