import { useState, useEffect } from 'react';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Evaluate from './pages/Evaluate';
import Training from './pages/Training';
import Borrowers from './pages/Borrowers';
import Profile from './pages/Profile';
import Home from './pages/Home';
import Decisions from './pages/Decisions';
import GraphPage from './pages/GraphPage';
import Setup from './pages/Setup';
import { setAuthToken } from './api/client';
import { FullPageLoading } from './components/Loading';

type View = 'login' | 'register' | 'dashboard' | 'evaluate' | 'decisions' | 'train' | 'borrowers' | 'profile' | 'graph' | 'setup';

function App() {
  const [view, setView] = useState<View>('login');
  const [isInitializing, setIsInitializing] = useState(true);

  useEffect(() => {
    const isSetupDone = localStorage.getItem('setup_done');
    if (!isSetupDone) {
      setView('setup');
      setIsInitializing(false);
      return;
    }

    const token = localStorage.getItem('auth_token');
    if (token) {
      setAuthToken(token);
      setView('dashboard');
    }
    // Simulate a brief initialization for premium feel or real check if needed
    setTimeout(() => {
      setIsInitializing(false);
    }, 1000);
  }, []);

  const handleSetupFinish = () => {
    localStorage.setItem('setup_done', 'true');
    setView('login');
  };

  const handleLoginSuccess = () => {
    setView('dashboard');
  };

  const handleLogout = () => {
    setAuthToken('');
    localStorage.removeItem('auth_user_name'); // Clear cached user name if any
    localStorage.removeItem('auth_user_email'); // Clear cached email if any
    setView('login');
  };

  if (isInitializing) {
    return <FullPageLoading />;
  }

  const renderDashboardContent = () => {
    switch (view) {
      case 'dashboard': return <Home onNavigate={setView} />;
      case 'evaluate': return <Evaluate />;
      case 'decisions': return <Decisions />;
      case 'train': return <Training />;
      case 'borrowers': return <Borrowers />;
      case 'profile': return <Profile />;
      case 'graph': return <GraphPage />;
      default: return <Home onNavigate={setView} />;
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-foreground">
      {view === 'setup' && (
        <Setup onFinish={handleSetupFinish} />
      )}
      {view === 'login' && (
        <Login
          onLoginSuccess={handleLoginSuccess}
          onNavigateRegister={() => setView('register')}
        />
      )}
      {view === 'register' && (
        <Register
          onNavigateLogin={() => setView('login')}
        />
      )}
      {(view === 'dashboard' || view === 'evaluate' || view === 'decisions' || view === 'train' || view === 'borrowers' || view === 'profile' || view === 'graph') && (
        <Dashboard
          onLogout={handleLogout}
          onNavigate={(v: any) => setView(v)}
          currentView={view}
        >
          {renderDashboardContent()}
        </Dashboard>
      )}
    </div>
  );
}

export default App;
