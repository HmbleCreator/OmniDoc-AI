const API_BASE_URL = 'http://localhost:8000/api';

export interface Document {
  id: string;
  name: string;
  type: string;
  content: string;
  summary: string;
  upload_time: string;
  chunks: string[];
  chunk_metadata?: ChunkMetadata[];
  headers?: DocumentHeader[];
}

export interface ChunkMetadata {
  chunk_id: string;
  page_number: number;
  section_header?: string;
  subsection_header?: string;
  paragraph_number: number;
  chunk_type: string;
  word_count: number;
  confidence: number;
}

export interface DocumentHeader {
  text: string;
  page: number;
}

export interface ChatMessage {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: string;
  document_ref?: string;
  reasoning?: string;
  show_reasoning?: boolean;
}

export interface DocumentReference {
  citation: string;
  relevance_score: number;
  chunk_preview: string;
  key_terms?: string[];
  semantic_density?: number;
  readability_score?: number;
}

export interface ChatResponse {
  message: ChatMessage;
  reasoning: string;
  document_references: DocumentReference[];
  confidence: number;
  reasoning_chain?: ReasoningStep[];
}

export interface ReasoningStep {
  step_id: string;
  step_type: string;
  description: string;
  input_data: any;
  output_data: any;
  confidence: number;
  timestamp: string;
}

export interface ChallengeQuestion {
  id: string;
  question: string;
  correct_answer: string;
  explanation: string;
  document_references: DocumentReference[];
}

export interface ChallengeResponse {
  questions: ChallengeQuestion[];
}

export interface LLMProvider {
  id: string;
  name: string;
  description: string;
  requires_api_key: boolean;
  models: string[];
}

export interface SearchResult {
  chunk: string;
  metadata: ChunkMetadata;
  distance: number;
  citation: string;
  relevance_score: number;
}

export interface SearchResponse {
  query: string;
  results: SearchResult[];
  result_count: number;
  reranking_used: boolean;
  embedding_model: string;
}

export interface ChatSuggestion {
  type: string;
  text: string;
  category?: string;
  source?: string;
}

class ApiService {
  private async request<T>(endpoint: string, options: RequestInit = {}, apiKeys?: Record<string, string>): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    
    // Add API keys to headers if provided
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    // Merge existing headers
    if (options.headers) {
      if (typeof options.headers === 'object' && !Array.isArray(options.headers)) {
        Object.assign(headers, options.headers);
      }
    }
    
    // Add API keys
    if (apiKeys) {
      headers['X-OpenAI-Key'] = apiKeys.openai || '';
      headers['X-Gemini-Key'] = apiKeys.gemini || '';
      headers['X-Claude-Key'] = apiKeys.claude || '';
      headers['X-Local-Path'] = apiKeys.local || '';
    }
    
    const response = await fetch(url, {
      ...options,
      headers,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Unknown error' }));
      throw new Error(error.error || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // Document APIs
  async uploadDocument(file: File, apiKeys?: Record<string, string>, provider: string = 'openai'): Promise<Document> {
    const formData = new FormData();
    formData.append('file', file);

    // Add API keys to headers if provided
    const headers: Record<string, string> = {};
    if (apiKeys) {
      headers['X-OpenAI-Key'] = apiKeys.openai || '';
      headers['X-Gemini-Key'] = apiKeys.gemini || '';
      headers['X-Claude-Key'] = apiKeys.claude || '';
      headers['X-Local-Path'] = apiKeys.local || '';
    }

    const response = await fetch(`${API_BASE_URL}/documents/upload?provider=${encodeURIComponent(provider)}`, {
      method: 'POST',
      headers,
      body: formData,
    });

    console.log('Upload response status:', response.status);
    console.log('Upload response headers:', response.headers);

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Upload failed' }));
      console.error('Upload error response:', error);
      throw new Error(error.error || 'Upload failed');
    }

    const responseData = await response.json();
    console.log('Upload success response:', responseData);
    return responseData;
  }

  async getDocuments(): Promise<Document[]> {
    return this.request<Document[]>('/documents');
  }

  async getDocument(id: string): Promise<Document> {
    return this.request<Document>(`/documents/${id}`);
  }

  async getDocumentMetadata(id: string): Promise<{
    document_id: string;
    chunk_metadata: ChunkMetadata[];
    headers: DocumentHeader[];
    chunk_count: number;
    total_words: number;
  }> {
    return this.request(`/documents/${id}/metadata`);
  }

  async deleteDocument(id: string): Promise<void> {
    await this.request(`/documents/${id}`, { method: 'DELETE' });
  }

  async searchDocuments(
    query: string, 
    documentIds?: string[], 
    topK: number = 5,
    useReranking: boolean = true
  ): Promise<SearchResponse> {
    const params = new URLSearchParams({ 
      query, 
      top_k: topK.toString(),
      use_reranking: useReranking.toString()
    });
    
    if (documentIds) {
      documentIds.forEach(id => params.append('document_ids', id));
    }
    
    return this.request<SearchResponse>(`/documents/search?${params}`);
  }

  async getSearchSuggestions(documentId?: string): Promise<{
    suggestions: ChatSuggestion[];
    total_documents: number;
  }> {
    const params = documentId ? `?document_id=${documentId}` : '';
    return this.request(`/documents/search/suggestions${params}`);
  }

  // Chat APIs
  async askQuestion(
    message: string,
    mode: 'ask' | 'challenge' = 'ask',
    documentId?: string,
    provider: string = 'openai',
    apiKeys?: Record<string, string>
  ): Promise<ChatResponse> {
    return this.request<ChatResponse>('/chat/ask', {
      method: 'POST',
      body: JSON.stringify({
        message,
        mode,
        document_id: documentId,
        provider,
      }),
    }, apiKeys);
  }

  async generateChallengeQuestions(
    documentIds: string[],
    provider: string = 'openai',
    apiKeys?: Record<string, string>
  ): Promise<ChallengeResponse> {
    return this.request<ChallengeResponse>('/chat/challenge/generate', {
      method: 'POST',
      body: JSON.stringify({
        document_ids: documentIds,
        provider,
      }),
    }, apiKeys);
  }

  async getNextChallengeQuestion(
    documentIds: string[],
    provider: string = 'openai',
    apiKeys?: Record<string, string>
  ): Promise<ChallengeQuestion> {
    return this.request<ChallengeQuestion>('/chat/challenge/next-question', {
      method: 'POST',
      body: JSON.stringify({
        document_ids: documentIds,
        provider,
      }),
    }, apiKeys);
  }

  async evaluateChallengeAnswer(
    questionId: string,
    userAnswer: string,
    correctAnswer: string,
    questionText: string,
    documentContext: string,
    provider: string = 'openai',
    apiKeys?: Record<string, string>
  ): Promise<any> {
    return this.request('/chat/challenge/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        question_id: questionId,
        user_answer: userAnswer,
        correct_answer: correctAnswer,
        question_text: questionText,
        document_context: documentContext,
        provider,
      }),
    }, apiKeys);
  }

  async evaluateChallengeAnswerInteractive(
    questionId: string,
    userAnswer: string,
    correctAnswer: string,
    questionText: string,
    documentContext: string,
    provider: string = 'openai',
    apiKeys?: Record<string, string>
  ): Promise<any> {
    return this.request('/chat/challenge/evaluate-answer', {
      method: 'POST',
      body: JSON.stringify({
        question_id: questionId,
        user_answer: userAnswer,
        correct_answer: correctAnswer,
        question_text: questionText,
        document_context: documentContext,
        provider,
      }),
    }, apiKeys);
  }

  async getChatSuggestions(documentId?: string): Promise<{
    suggestions: ChatSuggestion[];
    document_id?: string;
  }> {
    const params = documentId ? `?document_id=${documentId}` : '';
    return this.request(`/chat/suggestions${params}`);
  }

  // Session APIs
  async startSession(): Promise<{ session_id: string; message: string }> {
    return this.request('/chat/session/start', { method: 'POST' });
  }

  async getSession(sessionId: string): Promise<any> {
    return this.request(`/chat/session/${sessionId}`);
  }

  async addMessageToSession(sessionId: string, message: ChatMessage): Promise<any> {
    return this.request(`/chat/session/${sessionId}/message`, {
      method: 'POST',
      body: JSON.stringify(message),
    });
  }

  async deleteSession(sessionId: string): Promise<any> {
    return this.request(`/chat/session/${sessionId}`, { method: 'DELETE' });
  }

  async getSessions(): Promise<any[]> {
    return this.request('/chat/sessions');
  }

  // LLM APIs
  async getProviders(): Promise<{ providers: LLMProvider[] }> {
    return this.request('/llm/providers');
  }

  async configureProvider(provider: string, apiKey: string): Promise<any> {
    return this.request('/llm/configure', {
      method: 'POST',
      body: JSON.stringify({
        provider,
        api_key: apiKey,
      }),
    });
  }

  async getLLMStatus(): Promise<any> {
    return this.request('/llm/status');
  }

  async testProvider(provider: string, apiKey?: string): Promise<any> {
    const params = new URLSearchParams({ provider });
    if (apiKey) {
      params.append('api_key', apiKey);
    }
    return this.request(`/llm/test?${params}`, { method: 'POST' });
  }

  async getProviderModels(provider: string): Promise<any> {
    return this.request(`/llm/models/${provider}`);
  }
}

export const apiService = new ApiService(); 