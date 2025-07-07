import React, { useState } from 'react';
import { User, Bot, Eye, EyeOff, FileText, ExternalLink, Quote, Brain, ChevronDown, ChevronRight, Activity } from 'lucide-react';
import { useApp } from '../context/AppContext';
import { Message as MessageType } from '../context/AppContext';
import { DocumentReference } from '../services/api';

interface ReasoningStep {
  step_id: string;
  step_type: string;
  description: string;
  input_data: any;
  output_data: any;
  confidence: number;
  timestamp: string;
}

interface MessageProps {
  message: MessageType;
  documentReferences?: DocumentReference[];
  reasoningChain?: ReasoningStep[];
}

const Message: React.FC<MessageProps> = ({ message, documentReferences = [], reasoningChain = [] }) => {
  const { toggleMessageReasoning, showReasoning } = useApp();
  const [showReasoningChain, setShowReasoningChain] = useState(false);
  const [expandedSteps, setExpandedSteps] = useState<Set<string>>(new Set());

  const handleToggleReasoning = () => {
    toggleMessageReasoning(message.id);
  };

  const handleToggleReasoningChain = () => {
    setShowReasoningChain(!showReasoningChain);
  };

  const toggleStepExpansion = (stepId: string) => {
    const newExpanded = new Set(expandedSteps);
    if (newExpanded.has(stepId)) {
      newExpanded.delete(stepId);
    } else {
      newExpanded.add(stepId);
    }
    setExpandedSteps(newExpanded);
  };

  const formatCitation = (citation: string) => {
    try {
      return citation.replace(/, /g, ' • ');
    } catch (error) {
      console.warn('Error formatting citation:', error);
      return citation || '';
    }
  };

  const getStepIcon = (stepType: string) => {
    switch (stepType) {
      case 'analysis': return <Brain size={14} className="text-[#4FC3F7]" />;
      case 'retrieval': return <FileText size={14} className="text-[#4FC3F7]" />;
      case 'synthesis': return <Activity size={14} className="text-[#4FC3F7]" />;
      case 'evaluation': return <Bot size={14} className="text-[#4FC3F7]" />;
      case 'generation': return <Brain size={14} className="text-[#4FC3F7]" />;
      case 'error': return <Activity size={14} className="text-[#FF5722]" />;
      default: return <Activity size={14} className="text-[#90A4AE]" />;
    }
  };

  const getStepColor = (stepType: string) => {
    switch (stepType) {
      case 'analysis': return 'border-[#4FC3F7] bg-[#1A2C32]';
      case 'retrieval': return 'border-[#4FC3F7] bg-[#1A2C32]';
      case 'synthesis': return 'border-[#4FC3F7] bg-[#1A2C32]';
      case 'evaluation': return 'border-[#4FC3F7] bg-[#1A2C32]';
      case 'generation': return 'border-[#4FC3F7] bg-[#1A2C32]';
      case 'error': return 'border-[#FF5722] bg-[#2C1A1A]';
      default: return 'border-[#90A4AE] bg-[#1A2C32]';
    }
  };

  const formatTimestamp = (timestamp: string) => {
    try {
      return new Date(timestamp).toLocaleTimeString();
    } catch (error) {
      console.warn('Invalid timestamp:', timestamp);
      return 'Unknown time';
    }
  };

  const formatStructuredResponse = (content: string) => {
    try {
      if (content && content.includes('Answer:') && content.includes('References:') && content.includes('Reasoning:')) {
        const answerMatch = content.match(/Answer:\s*(.+?)(?=\nReferences:|$)/s);
        const referencesMatch = content.match(/References:\s*(.+?)(?=\nReasoning:|$)/s);
        const reasoningMatch = content.match(/Reasoning:\s*(.+?)(?=\nConfidence:|$)/s);
        
        if (answerMatch) {
          return {
            isStructured: true,
            answer: answerMatch[1].trim(),
            references: referencesMatch ? referencesMatch[1].trim() : '',
            reasoning: reasoningMatch ? reasoningMatch[1].trim() : ''
          };
        }
      }
      
      return { isStructured: false, content: content || '' };
    } catch (error) {
      console.warn('Error formatting structured response:', error);
      return { isStructured: false, content: content || '' };
    }
  };

  const responseData = formatStructuredResponse(message?.content || '');

  try {
  return (
      <div className={`flex gap-3 ${message?.type === 'user' ? 'justify-end' : 'justify-start'}`}>
        {message?.type === 'ai' && (
        <div className="w-8 h-8 bg-[#4FC3F7] rounded-full flex items-center justify-center flex-shrink-0">
          <Bot size={16} className="text-white" />
        </div>
      )}
      
        <div className={`max-w-[80%] ${message?.type === 'user' ? 'order-2' : ''}`}>
        <div className={`
          p-4 rounded-lg 
            ${message?.type === 'user' 
            ? 'bg-[#4FC3F7] text-white' 
            : 'bg-[#37474F] text-[#ECEFF1]'
          }
        `}>
            {/* Display structured response or regular content */}
            {responseData.isStructured ? (
              <div className="space-y-3">
                <div>
                  <h4 className="text-sm font-medium text-[#4FC3F7] mb-2">Answer:</h4>
                  <p className="whitespace-pre-wrap">{responseData.answer}</p>
                </div>
                
                {responseData.references && (
                  <div>
                    <h4 className="text-sm font-medium text-[#4FC3F7] mb-2">References:</h4>
                    <p className="text-sm text-[#B0BEC5]">{responseData.references}</p>
                  </div>
                )}
                
                {responseData.reasoning && (
                  <div>
                    <h4 className="text-sm font-medium text-[#4FC3F7] mb-2">Reasoning:</h4>
                    <p className="text-sm text-[#B0BEC5] font-mono leading-relaxed">{responseData.reasoning}</p>
                  </div>
                )}
              </div>
            ) : (
              <p className="whitespace-pre-wrap">{message?.content || ''}</p>
            )}
            
            {/* Enhanced Document References */}
            {message?.type === 'ai' && documentReferences && documentReferences.length > 0 && (
              <div className="mt-3 pt-3 border-t border-[#455A64]">
                <div className="flex items-center gap-2 text-[#90A4AE] mb-2">
                  <FileText size={14} />
                  <span className="text-sm font-medium">Document References:</span>
                </div>
                
                <div className="space-y-2">
                  {documentReferences.slice(0, 3).map((ref, index) => {
                    try {
                      return (
                        <div key={index} className="bg-[#1A2C32] rounded p-3 border border-[#455A64]">
                          <div className="flex items-start justify-between gap-2">
                            <div className="flex-1">
                              <div className="flex items-center gap-2 mb-1">
                                <Quote size={12} className="text-[#4FC3F7]" />
                                <span className="text-xs font-medium text-[#4FC3F7]">
                                  {formatCitation(ref?.citation || '')}
                                </span>
                                <span className="text-xs text-[#90A4AE]">
                                  ({((ref?.relevance_score || 0) * 100).toFixed(0)}% relevant)
                                </span>
                              </div>
                              <p className="text-xs text-[#B0BEC5] leading-relaxed">
                                {ref?.chunk_preview || ''}
                              </p>
                              
                              {/* Enhanced metadata */}
                              {ref?.key_terms && ref.key_terms.length > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {ref.key_terms.slice(0, 3).map((term, termIndex) => (
                                    <span key={termIndex} className="text-xs bg-[#263238] px-2 py-1 rounded text-[#90A4AE]">
                                      {term}
                                    </span>
                                  ))}
                                </div>
                              )}
                              
                              {/* Semantic density indicator */}
                              {ref?.semantic_density && (
                                <div className="mt-1 flex items-center gap-1">
                                  <div className="text-xs text-[#90A4AE]">Density:</div>
                                  <div className="flex-1 bg-[#263238] rounded-full h-1">
                                    <div 
                                      className="bg-[#4FC3F7] h-1 rounded-full transition-all"
                                      style={{ width: `${(ref.semantic_density || 0) * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-xs text-[#90A4AE]">
                                    {((ref.semantic_density || 0) * 100).toFixed(0)}%
                                  </span>
                                </div>
                              )}
                            </div>
                            <ExternalLink size={12} className="text-[#90A4AE] flex-shrink-0 mt-1" />
                          </div>
                        </div>
                      );
                    } catch (error) {
                      console.warn('Error rendering document reference:', error);
                      return null;
                    }
                  })}
                </div>
              </div>
            )}
            
            {/* Legacy document reference (fallback) */}
            {message?.type === 'ai' && message?.documentRef && documentReferences.length === 0 && (
            <div className="mt-3 pt-3 border-t border-[#455A64] text-sm">
              <div className="flex items-center gap-2 text-[#90A4AE]">
                <FileText size={14} />
                <span>Reference: {message.documentRef}</span>
              </div>
            </div>
          )}
        </div>
        
          {/* Enhanced Reasoning Display */}
          {message?.type === 'ai' && (message?.reasoning || reasoningChain.length > 0) && (
            <div className="mt-2 space-y-2">
              {/* Basic reasoning toggle */}
              {message?.reasoning && (showReasoning || message?.showReasoning) && (
                <div>
            <button
              onClick={handleToggleReasoning}
              className="flex items-center gap-2 px-3 py-1 bg-[#37474F] rounded-lg border border-[#455A64] hover:bg-[#455A64] transition-colors text-sm"
            >
                    {message?.showReasoning ? (
                <EyeOff size={14} className="text-[#4FC3F7]" />
              ) : (
                <Eye size={14} className="text-[#4FC3F7]" />
              )}
              <span className="text-[#ECEFF1]">
                      {message?.showReasoning ? 'Hide Reasoning' : 'Show Reasoning'}
              </span>
            </button>
            
                  {message?.showReasoning && (
              <div className="mt-2 p-4 bg-[#1A2C32] rounded-lg border border-[#37474F] animate-fadeIn">
                      <h4 className="text-sm font-medium text-[#4FC3F7] mb-2 flex items-center gap-2">
                        <Bot size={14} />
                        AI Reasoning Process:
                      </h4>
                      <div className="bg-[#263238] rounded p-3 mb-3">
                <p className="text-sm text-[#ECEFF1] font-mono leading-relaxed">
                          {message?.reasoning}
                </p>
                      </div>
                      
                      {/* Confidence indicator */}
                      <div className="flex items-center gap-2 text-xs text-[#90A4AE]">
                        <div className="w-2 h-2 bg-[#4FC3F7] rounded-full"></div>
                        <span>High confidence response based on document analysis</span>
                      </div>
                    </div>
                  )}
                </div>
              )}
              
              {/* Advanced reasoning chain */}
              {reasoningChain.length > 0 && (
                <div>
                  <button
                    onClick={handleToggleReasoningChain}
                    className="flex items-center gap-2 px-3 py-1 bg-[#37474F] rounded-lg border border-[#455A64] hover:bg-[#455A64] transition-colors text-sm"
                  >
                    {showReasoningChain ? (
                      <ChevronDown size={14} className="text-[#4FC3F7]" />
                    ) : (
                      <ChevronRight size={14} className="text-[#4FC3F7]" />
                    )}
                    <Brain size={14} className="text-[#4FC3F7]" />
                    <span className="text-[#ECEFF1]">
                      {showReasoningChain ? 'Hide Reasoning Chain' : 'Show Reasoning Chain'}
                    </span>
                    <span className="text-xs text-[#90A4AE]">({reasoningChain.length} steps)</span>
                  </button>
                  
                  {showReasoningChain && (
                    <div className="mt-2 p-4 bg-[#1A2C32] rounded-lg border border-[#37474F] animate-fadeIn">
                      <h4 className="text-sm font-medium text-[#4FC3F7] mb-3 flex items-center gap-2">
                        <Brain size={14} />
                        Detailed Reasoning Chain:
                      </h4>
                      
                      <div className="space-y-3">
                        {reasoningChain.map((step, index) => {
                          try {
                            return (
                              <div key={step?.step_id || index} className={`border-l-2 p-3 ${getStepColor(step?.step_type || 'unknown')}`}>
                                <div className="flex items-center justify-between mb-2">
                                  <div className="flex items-center gap-2">
                                    {getStepIcon(step?.step_type || 'unknown')}
                                    <span className="text-sm font-medium text-[#ECEFF1] capitalize">
                                      {step?.step_type || 'unknown'}
                                    </span>
                                    <span className="text-xs text-[#90A4AE]">
                                      Step {index + 1}
                                    </span>
                                  </div>
                                  <div className="flex items-center gap-2">
                                    <span className="text-xs text-[#90A4AE]">
                                      {formatTimestamp(step?.timestamp || '')}
                                    </span>
                                    <button
                                      onClick={() => toggleStepExpansion(step?.step_id || '')}
                                      className="text-[#90A4AE] hover:text-[#ECEFF1]"
                                    >
                                      {expandedSteps.has(step?.step_id || '') ? (
                                        <ChevronDown size={12} />
                                      ) : (
                                        <ChevronRight size={12} />
                                      )}
                                    </button>
                                  </div>
                                </div>
                                
                                <p className="text-sm text-[#B0BEC5] mb-2">
                                  {step?.description || 'No description available'}
                                </p>
                                
                                {/* Confidence bar */}
                                <div className="flex items-center gap-2 mb-2">
                                  <span className="text-xs text-[#90A4AE]">Confidence:</span>
                                  <div className="flex-1 bg-[#263238] rounded-full h-1">
                                    <div 
                                      className="bg-[#4FC3F7] h-1 rounded-full transition-all"
                                      style={{ width: `${(step?.confidence || 0) * 100}%` }}
                                    ></div>
                                  </div>
                                  <span className="text-xs text-[#90A4AE]">
                                    {((step?.confidence || 0) * 100).toFixed(0)}%
                                  </span>
                                </div>
                                
                                {/* Expandable details */}
                                {expandedSteps.has(step?.step_id || '') && (
                                  <div className="mt-3 space-y-2">
                                    {step?.input_data && (
                                      <div>
                                        <h5 className="text-xs font-medium text-[#4FC3F7] mb-1">Input:</h5>
                                        <pre className="text-xs text-[#B0BEC5] bg-[#263238] p-2 rounded overflow-x-auto">
                                          {JSON.stringify(step.input_data, null, 2)}
                                        </pre>
                                      </div>
                                    )}
                                    
                                    {step?.output_data && (
                                      <div>
                                        <h5 className="text-xs font-medium text-[#4FC3F7] mb-1">Output:</h5>
                                        <pre className="text-xs text-[#B0BEC5] bg-[#263238] p-2 rounded overflow-x-auto">
                                          {JSON.stringify(step.output_data, null, 2)}
                                        </pre>
                                      </div>
                                    )}
                                  </div>
                                )}
                              </div>
                            );
                          } catch (error) {
                            console.warn('Error rendering reasoning step:', error);
                            return null;
                          }
                        })}
                      </div>
                    </div>
                  )}
              </div>
            )}
          </div>
        )}
      </div>
      
        {message?.type === 'user' && (
        <div className="w-8 h-8 bg-[#37474F] rounded-full flex items-center justify-center flex-shrink-0">
          <User size={16} className="text-[#90A4AE]" />
        </div>
      )}
    </div>
  );
  } catch (error) {
    console.error('Error rendering Message component:', error);
    return (
      <div className="flex gap-3 justify-start">
        <div className="w-8 h-8 bg-[#FF5722] rounded-full flex items-center justify-center flex-shrink-0">
          <Bot size={16} className="text-white" />
        </div>
        <div className="max-w-[80%]">
          <div className="p-4 rounded-lg bg-[#37474F] text-[#ECEFF1]">
            <p className="text-sm text-[#FF5722]">Error rendering message. Please try again.</p>
          </div>
        </div>
      </div>
    );
  }
};

export default Message;