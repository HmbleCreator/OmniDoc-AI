import React, { useState, useEffect } from 'react';
import { Upload, Settings, MessageSquare, FileText, Brain, Eye, EyeOff, Trash2, Download, ChevronRight, ChevronDown } from 'lucide-react';
import TopAppBar from './components/TopAppBar';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import SummaryCards from './components/SummaryCards';
import { AppProvider } from './context/AppContext';

// Error Boundary Component
class ErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('App Error:', error, errorInfo);
    console.error('Error Stack:', error.stack);
    console.error('Error Info:', errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="min-h-screen bg-[#121212] text-[#ECEFF1] flex items-center justify-center">
          <div className="text-center p-8">
            <h1 className="text-2xl font-bold mb-4">Something went wrong</h1>
            <p className="mb-4">The application encountered an error. Please refresh the page.</p>
            <button 
              onClick={() => window.location.reload()} 
              className="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isLoaded, setIsLoaded] = useState(false);

  useEffect(() => {
    // Ensure the app is properly loaded
    setIsLoaded(true);
  }, []);

  if (!isLoaded) {
    return (
      <div className="min-h-screen bg-[#121212] text-[#ECEFF1] flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-xl font-semibold mb-2">Loading OmniDoc AI...</h2>
          <p>Please wait while the application initializes.</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
    <AppProvider>
        <div className="min-h-screen bg-[#121212] text-[#ECEFF1] flex flex-col overflow-y-auto">
        <TopAppBar onToggleSidebar={() => setSidebarOpen(!sidebarOpen)} />
        
        <div className="flex-1 flex relative">
          <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
          
            <main className="flex-1 flex flex-col lg:flex-row gap-4 p-4">
              <div className="flex-1 flex flex-col min-w-0" style={{ height: 'calc(100vh - 120px)' }}>
                <div className="h-full">
              <ChatArea />
                </div>
            </div>
            
              <div className="w-full lg:w-80 flex-shrink-0 overflow-y-auto">
              <SummaryCards />
            </div>
          </main>
        </div>
      </div>
    </AppProvider>
    </ErrorBoundary>
  );
}

export default App;