import React, { createContext, useContext, useState, ReactNode } from 'react';

export type LLMProvider = 'openai' | 'gemini' | 'claude' | 'local';

export interface Document {
  id: string;
  name: string;
  type: 'pdf' | 'txt';
  content: string;
  summary: string;
  uploadTime: Date;
}

export interface Message {
  id: string;
  type: 'user' | 'ai';
  content: string;
  timestamp: Date;
  documentRef?: string;
  reasoning?: string;
  showReasoning?: boolean;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  lastActivity: Date;
}

export interface ChatMode {
  type: 'ask' | 'challenge';
  label: string;
}

interface AppContextType {
  // Settings
  selectedProvider: LLMProvider;
  setSelectedProvider: (provider: LLMProvider) => void;
  apiKeys: Record<LLMProvider, string>;
  setApiKey: (provider: LLMProvider, key: string) => void;
  showReasoning: boolean;
  setShowReasoning: (show: boolean) => void;
  
  // Documents
  documents: Document[];
  addDocument: (doc: Document) => void;
  removeDocument: (id: string) => void;
  selectedDocument: string | null;
  setSelectedDocument: (id: string | null) => void;
  
  // Chat Sessions
  chatSessions: ChatSession[];
  currentSessionId: string | null;
  currentSession: ChatSession | null;
  messages: Message[];
  addMessage: (message: Omit<Message, 'id' | 'timestamp'>) => string;
  startNewSession: () => void;
  switchToSession: (sessionId: string) => void;
  deleteSession: (sessionId: string) => void;
  toggleMessageReasoning: (id: string) => void;
  
  // Chat mode
  chatMode: ChatMode;
  setChatMode: (mode: ChatMode) => void;
  
  // UI
  isUploading: boolean;
  setIsUploading: (uploading: boolean) => void;
}

const AppContext = createContext<AppContextType | undefined>(undefined);

export const useApp = () => {
  const context = useContext(AppContext);
  if (!context) {
    throw new Error('useApp must be used within an AppProvider');
  }
  return context;
};

export const AppProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [selectedProvider, setSelectedProvider] = useState<LLMProvider>('openai');
  const [apiKeys, setApiKeys] = useState<Record<LLMProvider, string>>({
    openai: '',
    gemini: '',
    claude: '',
    local: ''
  });
  const [showReasoning, setShowReasoning] = useState(true);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [selectedDocument, setSelectedDocument] = useState<string | null>(null);
  const [chatSessions, setChatSessions] = useState<ChatSession[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [chatMode, setChatMode] = useState<ChatMode>({ type: 'ask', label: 'Ask Anything' });
  const [isUploading, setIsUploading] = useState(false);

  // Get current session and its messages
  const currentSession = chatSessions.find(session => session.id === currentSessionId) || null;
  const messages = currentSession?.messages || [];

  const setApiKey = (provider: LLMProvider, key: string) => {
    setApiKeys(prev => ({ ...prev, [provider]: key }));
  };

  const addDocument = (doc: Document) => {
    setDocuments(prev => [...prev, doc]);
  };

  const removeDocument = (id: string) => {
    setDocuments(prev => prev.filter(doc => doc.id !== id));
    if (selectedDocument === id) {
      setSelectedDocument(null);
    }
  };

  const generateSessionTitle = (firstMessage: string): string => {
    // Generate a title from the first message (max 50 chars)
    const title = firstMessage.length > 50 
      ? firstMessage.substring(0, 47) + '...' 
      : firstMessage;
    return title;
  };

  const startNewSession = () => {
    const newSession: ChatSession = {
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      lastActivity: new Date()
    };
    
    setChatSessions(prev => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
  };

  const switchToSession = (sessionId: string) => {
    setCurrentSessionId(sessionId);
  };

  const deleteSession = (sessionId: string) => {
    setChatSessions(prev => prev.filter(session => session.id !== sessionId));
    if (currentSessionId === sessionId) {
      const remainingSessions = chatSessions.filter(session => session.id !== sessionId);
      setCurrentSessionId(remainingSessions.length > 0 ? remainingSessions[0].id : null);
    }
  };

  const addMessage = (message: Omit<Message, 'id' | 'timestamp'>) => {
    const newMessage: Message = {
      ...message,
      id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      timestamp: new Date()
    };

    // Debug logging
    console.log('Adding message:', {
      id: newMessage.id,
      type: newMessage.type,
      content: newMessage.content.substring(0, 50) + '...',
      currentSessionId,
      existingMessagesCount: currentSession?.messages?.length || 0
    });

    setChatSessions(prev => prev.map(session => {
      if (session.id === currentSessionId) {
        // Check for duplicate messages (same content and type within last 5 seconds)
        const recentMessages = session.messages.filter(msg => 
          msg.type === newMessage.type && 
          msg.content === newMessage.content &&
          Date.now() - msg.timestamp.getTime() < 5000
        );
        
        if (recentMessages.length > 0) {
          console.log('Duplicate message detected, skipping:', newMessage.content.substring(0, 50));
          return session;
        }
        
        const updatedMessages = [...session.messages, newMessage];
        
        // Update session title if this is the first user message
        let updatedTitle = session.title;
        if (session.messages.length === 0 && message.type === 'user') {
          updatedTitle = generateSessionTitle(message.content);
        }
        
        return {
          ...session,
          title: updatedTitle,
          messages: updatedMessages,
          lastActivity: new Date()
        };
      }
      return session;
    }));

    // If no current session exists, create one
    if (!currentSessionId) {
      const newSession: ChatSession = {
        id: `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        title: message.type === 'user' ? generateSessionTitle(message.content) : 'New Chat',
        messages: [newMessage],
        createdAt: new Date(),
        lastActivity: new Date()
      };
      
      setChatSessions(prev => [newSession, ...prev]);
      setCurrentSessionId(newSession.id);
    }

    return newMessage.id;
  };

  const toggleMessageReasoning = (id: string) => {
    setChatSessions(prev => prev.map(session => {
      if (session.id === currentSessionId) {
        return {
          ...session,
          messages: session.messages.map(msg => 
            msg.id === id ? { ...msg, showReasoning: !msg.showReasoning } : msg
          )
        };
      }
      return session;
    }));
  };

  return (
    <AppContext.Provider value={{
      selectedProvider,
      setSelectedProvider,
      apiKeys,
      setApiKey,
      showReasoning,
      setShowReasoning,
      documents,
      addDocument,
      removeDocument,
      selectedDocument,
      setSelectedDocument,
      chatSessions,
      currentSessionId,
      currentSession,
      messages,
      addMessage,
      startNewSession,
      switchToSession,
      deleteSession,
      toggleMessageReasoning,
      chatMode,
      setChatMode,
      isUploading,
      setIsUploading
    }}>
      {children}
    </AppContext.Provider>
  );
};