import React from 'react';
import { X, Key, Trash2, MessageSquare, ChevronDown, Plus } from 'lucide-react';
import { useApp } from '../context/AppContext';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  const { 
    selectedProvider, 
    setSelectedProvider,
    apiKeys, 
    setApiKey, 
    showReasoning, 
    setShowReasoning, 
    chatSessions,
    currentSessionId,
    startNewSession,
    switchToSession,
    deleteSession
  } = useApp();

  const providers = [
    { value: 'openai', label: 'OpenAI' },
    { value: 'gemini', label: 'Google Gemini' },
    { value: 'claude', label: 'Anthropic Claude' },
    { value: 'local', label: 'Local LLM' }
  ];

  const providerLabels = {
    openai: 'OpenAI API Key',
    gemini: 'Gemini API Key',
    claude: 'Claude API Key',
    local: 'Local LLM Path'
  };

  const handleNewChat = () => {
    startNewSession();
  };

  const handleSessionClick = (sessionId: string) => {
    switchToSession(sessionId);
  };

  const handleDeleteSession = (e: React.MouseEvent, sessionId: string) => {
    e.stopPropagation();
    deleteSession(sessionId);
  };

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
          onClick={onClose}
        />
      )}
      
      <aside className={`
        fixed lg:relative top-0 left-0 h-full w-80 bg-[#263238] border-r border-[#37474F] z-50 transform transition-transform duration-300 ease-in-out flex flex-col
        ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
      `}>
        <div className="p-4 border-b border-[#37474F] flex items-center justify-between lg:hidden">
          <h2 className="text-lg font-semibold text-[#ECEFF1]">Settings</h2>
          <button
            onClick={onClose}
            className="p-1 hover:bg-[#37474F] rounded-lg transition-colors"
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* LLM Provider Selection */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-[#90A4AE] uppercase tracking-wider">
              LLM Provider
            </h3>
            <div className="relative">
              <select
                value={selectedProvider}
                onChange={(e) => setSelectedProvider(e.target.value as any)}
                className="w-full bg-[#37474F] text-[#ECEFF1] px-3 py-2 pr-8 rounded-lg border border-[#455A64] focus:outline-none focus:ring-2 focus:ring-[#4FC3F7] focus:border-transparent appearance-none cursor-pointer"
              >
                {providers.map(provider => (
                  <option key={provider.value} value={provider.value}>
                    {provider.label}
                  </option>
                ))}
              </select>
              <ChevronDown size={16} className="absolute right-2 top-1/2 transform -translate-y-1/2 text-[#90A4AE] pointer-events-none" />
            </div>
          </div>

          {/* API Key Section */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-[#90A4AE] uppercase tracking-wider">
              API Configuration
            </h3>
            <div className="space-y-2">
              <label className="block text-sm text-[#ECEFF1]">
                {providerLabels[selectedProvider]}
              </label>
              <div className="relative">
                <Key size={16} className="absolute left-3 top-1/2 transform -translate-y-1/2 text-[#90A4AE]" />
                <input
                  type="password"
                  value={apiKeys[selectedProvider]}
                  onChange={(e) => setApiKey(selectedProvider, e.target.value)}
                  placeholder="Enter API key..."
                  className="w-full pl-10 pr-3 py-2 bg-[#37474F] text-[#ECEFF1] rounded-lg border border-[#455A64] focus:outline-none focus:ring-2 focus:ring-[#4FC3F7] focus:border-transparent"
                />
              </div>
            </div>
          </div>

          {/* Settings */}
          <div className="space-y-3">
            <h3 className="text-sm font-medium text-[#90A4AE] uppercase tracking-wider">
              Settings
            </h3>
            <label className="flex items-center gap-3 cursor-pointer">
              <input
                type="checkbox"
                checked={showReasoning}
                onChange={(e) => setShowReasoning(e.target.checked)}
                className="w-4 h-4 text-[#4FC3F7] bg-[#37474F] border-[#455A64] rounded focus:ring-[#4FC3F7] focus:ring-2"
              />
              <span className="text-sm text-[#ECEFF1]">Always show reasoning</span>
            </label>
          </div>

          {/* Chat Sessions */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-medium text-[#90A4AE] uppercase tracking-wider">
                Chat Sessions
              </h3>
              <button
                onClick={handleNewChat}
                className="flex items-center gap-2 px-3 py-1 bg-[#4FC3F7] text-white rounded-lg hover:bg-[#29B6F6] transition-colors text-sm"
                title="Start new chat"
              >
                <Plus size={14} />
                <span>New Chat</span>
              </button>
            </div>
            
            <div className="space-y-2 max-h-60 overflow-y-auto">
              {chatSessions.length === 0 ? (
                <p className="text-sm text-[#90A4AE] italic">No chat sessions yet</p>
              ) : (
                chatSessions.map((session) => (
                  <div
                    key={session.id}
                    onClick={() => handleSessionClick(session.id)}
                    className={`
                      p-3 rounded-lg border cursor-pointer transition-colors group
                      ${currentSessionId === session.id 
                        ? 'bg-[#4FC3F7] bg-opacity-20 border-[#4FC3F7]' 
                        : 'bg-[#37474F] border-[#455A64] hover:bg-[#455A64]'
                      }
                    `}
                  >
                    <div className="flex items-start justify-between">
                      <div className="flex items-start gap-2 flex-1 min-w-0">
                        <MessageSquare size={14} className="text-[#4FC3F7] mt-1 flex-shrink-0" />
                        <div className="min-w-0 flex-1">
                          <p className="text-sm text-[#ECEFF1] line-clamp-2 font-medium">
                            {session.title}
                          </p>
                          <div className="flex items-center gap-2 mt-1">
                            <p className="text-xs text-[#90A4AE]">
                              {session.messages.length} messages
                            </p>
                            <span className="text-xs text-[#90A4AE]">•</span>
                            <p className="text-xs text-[#90A4AE]">
                              {session.lastActivity.toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                      </div>
                      <button
                        onClick={(e) => handleDeleteSession(e, session.id)}
                        className="p-1 hover:bg-[#37474F] rounded-lg transition-colors opacity-0 group-hover:opacity-100"
                        title="Delete session"
                      >
                        <Trash2 size={12} className="text-[#90A4AE]" />
                      </button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;