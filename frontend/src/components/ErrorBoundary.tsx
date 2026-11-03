import { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw, Home } from 'lucide-react';

interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null,
    };
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo): void {
    this.setState({ errorInfo });
    // Log to console for debugging
    console.error('[FedSecureAI ErrorBoundary Caught Error]:', error, errorInfo);
  }

  handleReset = (): void => {
    this.setState({ hasError: false, error: null, errorInfo: null });
    if (this.props.onReset) {
      this.props.onReset();
    }
  };

  handleReload = (): void => {
    window.location.reload();
  };

  render(): ReactNode {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center p-6 font-sans antialiased">
          <div className="max-w-2xl w-full bg-slate-900 border border-rose-500/40 rounded-3xl p-8 shadow-2xl space-y-6">
            <div className="flex items-center space-x-4">
              <div className="w-14 h-14 rounded-2xl bg-rose-950/80 border border-rose-500/60 text-rose-400 flex items-center justify-center shrink-0">
                <AlertTriangle className="w-7 h-7" />
              </div>
              <div>
                <h2 className="text-xl font-bold text-white tracking-tight">Application Render Error</h2>
                <p className="text-xs text-rose-300 font-mono mt-0.5">
                  An unexpected exception was caught by the FedSecureAI error boundary.
                </p>
              </div>
            </div>

            <div className="p-4 rounded-2xl bg-slate-950 border border-slate-800 text-xs font-mono space-y-2 overflow-hidden">
              <div className="text-rose-400 font-semibold break-words">
                {this.state.error?.name}: {this.state.error?.message || 'Unknown runtime error'}
              </div>
              {this.state.error?.stack && (
                <div className="text-[11px] text-slate-400 max-h-48 overflow-y-auto whitespace-pre-wrap font-mono pt-2 border-t border-slate-900">
                  {this.state.error.stack}
                </div>
              )}
            </div>

            <div className="flex flex-wrap items-center justify-between gap-3 pt-2 border-t border-slate-800">
              <span className="text-xs text-slate-400 font-mono">
                FedSecureAI Enterprise Analyst Dashboard
              </span>
              <div className="flex items-center space-x-3">
                <button
                  onClick={this.handleReset}
                  className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-slate-700 transition cursor-pointer"
                >
                  <Home className="w-3.5 h-3.5" />
                  <span>Reset View</span>
                </button>
                <button
                  onClick={this.handleReload}
                  className="flex items-center space-x-1.5 px-4 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shadow-lg shadow-indigo-600/30 transition cursor-pointer"
                >
                  <RefreshCw className="w-3.5 h-3.5" />
                  <span>Reload Page</span>
                </button>
              </div>
            </div>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
