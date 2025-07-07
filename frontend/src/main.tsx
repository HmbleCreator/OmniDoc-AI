import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';

// Add error handling for Edge compatibility
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}

try {
  const root = createRoot(rootElement);
  root.render(<App />);
} catch (error) {
  console.error('Failed to render app:', error);
  // Fallback for older browsers
  if (rootElement) {
    rootElement.innerHTML = '<div style="padding: 20px; color: white; background: #121212;">Loading application...</div>';
  }
}
