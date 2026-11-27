import React from 'react';
import {
  siPython,
  siPytorch,
  siFastapi,
  siScikitlearn,
  siPandas,
  siNumpy,
  siReact,
  siTypescript,
  siVite,
  siTailwindcss,
  siDocker,
  siHuggingface,
  siVercel,
  siRender,
  siGithub,
  siPytest,
  siRuff,
  type SimpleIcon,
} from 'simple-icons';
import {
  Upload,
  FileSpreadsheet,
  Layers,
  Network,
  Gauge,
  FileText,
  ArrowRight,
  ArrowDown,
  Server,
  BarChart3,
  Trees,
  Sparkles,
  GitMerge,
  ShieldCheck,
} from 'lucide-react';

interface TechCardItem {
  name: string;
  summary: string;
  simpleIcon?: SimpleIcon;
  lucideIcon?: React.ReactNode;
}

interface TechCategory {
  title: string;
  items: TechCardItem[];
}

const BrandLogo: React.FC<{ icon: SimpleIcon }> = ({ icon }) => (
  <svg
    viewBox="0 0 24 24"
    className="w-6 h-6 shrink-0"
    role="img"
    aria-label={icon.title}
  >
    <path d={icon.path} fill={`#${icon.hex}`} />
  </svg>
);

const TECH_CATEGORIES: TechCategory[] = [
  {
    title: 'Machine Learning',
    items: [
      {
        name: 'PyTorch',
        summary: 'Trains and serves recurrent LSTM, GRU, and BiLSTM AML sequence models.',
        simpleIcon: siPytorch,
      },
      {
        name: 'FedAvg & FedProx',
        summary: 'Aggregates client weights across simulated banks without sharing raw rows.',
        lucideIcon: <GitMerge className="w-6 h-6 text-indigo-400 shrink-0" />,
      },
      {
        name: 'LightGBM',
        summary: 'Provides fast tabular gradient-boosted fallback and custom CSV training.',
        lucideIcon: <Trees className="w-6 h-6 text-emerald-400 shrink-0" />,
      },
      {
        name: 'scikit-learn',
        summary: 'Computes PR-AUC, ROC-AUC, F1, calibration, and feature scaling.',
        simpleIcon: siScikitlearn,
      },
      {
        name: 'Integrated Gradients',
        summary: 'Attributes per-transaction risk scores back to input sequence features.',
        lucideIcon: <Sparkles className="w-6 h-6 text-amber-400 shrink-0" />,
      },
      {
        name: 'Pandas',
        summary: 'Parses uploaded CSVs, aligns schemas, and builds account sequences.',
        simpleIcon: siPandas,
      },
      {
        name: 'NumPy',
        summary: 'Vectorizes sequence tensors, probability arrays, and threshold metrics.',
        simpleIcon: siNumpy,
      },
      {
        name: 'Matplotlib',
        summary: 'Renders benchmark plots and visual diagnostics during evaluation.',
        lucideIcon: <BarChart3 className="w-6 h-6 text-sky-400 shrink-0" />,
      },
    ],
  },
  {
    title: 'Backend',
    items: [
      {
        name: 'Python',
        summary: 'Core runtime powering data preprocessing, training, and API services.',
        simpleIcon: siPython,
      },
      {
        name: 'FastAPI',
        summary: 'Serves typed REST endpoints for CSV inspection, jobs, and explanations.',
        simpleIcon: siFastapi,
      },
      {
        name: 'Uvicorn',
        summary: 'ASGI server running asynchronous prediction jobs and health checks.',
        lucideIcon: <Server className="w-6 h-6 text-violet-400 shrink-0" />,
      },
      {
        name: 'ReportLab',
        summary: 'Generates downloadable PDF evaluation reports with AML metrics.',
        lucideIcon: <FileText className="w-6 h-6 text-rose-400 shrink-0" />,
      },
    ],
  },
  {
    title: 'Frontend',
    items: [
      {
        name: 'React',
        summary: 'Renders interactive analytics, triage queues, and model comparisons.',
        simpleIcon: siReact,
      },
      {
        name: 'TypeScript',
        summary: 'Enforces strict API contracts across evaluation and history views.',
        simpleIcon: siTypescript,
      },
      {
        name: 'Vite',
        summary: 'Bundles the frontend and runs the fast local development server.',
        simpleIcon: siVite,
      },
      {
        name: 'Tailwind CSS',
        summary: 'Styles responsive dark and light layouts across desktop and mobile.',
        simpleIcon: siTailwindcss,
      },
    ],
  },
  {
    title: 'Deployment & Testing',
    items: [
      {
        name: 'Docker',
        summary: 'Packages the FastAPI backend and model artifacts for portable hosting.',
        simpleIcon: siDocker,
      },
      {
        name: 'Hugging Face Spaces',
        summary: 'Hosts the containerized backend with 16 GB RAM for large CSVs.',
        simpleIcon: siHuggingface,
      },
      {
        name: 'Vercel',
        summary: 'Deploys and serves the production React frontend globally.',
        simpleIcon: siVercel,
      },
      {
        name: 'Render',
        summary: 'Alternative cloud deployment target configured via render.yaml.',
        simpleIcon: siRender,
      },
      {
        name: 'GitHub',
        summary: 'Manages source control and deployment branches for the repository.',
        simpleIcon: siGithub,
      },
      {
        name: 'pytest',
        summary: 'Verifies CSV evaluation, inference jobs, and API contracts automatically.',
        simpleIcon: siPytest,
      },
      {
        name: 'Ruff',
        summary: 'Lints Python backend code for fast style and correctness checks.',
        simpleIcon: siRuff,
      },
    ],
  },
];

const WORKFLOW_STEPS = [
  {
    title: 'CSV Upload',
    detail: 'Any delimiter & size',
    icon: <Upload className="w-5 h-5 text-indigo-400" />,
  },
  {
    title: 'Schema Adapter',
    detail: 'Auto-maps columns',
    icon: <FileSpreadsheet className="w-5 h-5 text-cyan-400" />,
  },
  {
    title: 'Sequence Builder',
    detail: '10-step account windows',
    icon: <Layers className="w-5 h-5 text-violet-400" />,
  },
  {
    title: 'Federated Model',
    detail: 'LSTM / LightGBM score',
    icon: <Network className="w-5 h-5 text-emerald-400" />,
  },
  {
    title: 'Thresholding',
    detail: 'θ = 0.3378 decision',
    icon: <Gauge className="w-5 h-5 text-amber-400" />,
  },
  {
    title: 'Reports & History',
    detail: 'PDF & CSV exports',
    icon: <FileText className="w-5 h-5 text-rose-400" />,
  },
];

export const TechnologyStackPage: React.FC = () => {
  return (
    <div className="space-y-8 animate-fade-in">
      {/* Page Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white tracking-tight">
          Technology Stack &amp; System Walkthrough
        </h1>
        <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
          Core libraries, federated learning workflow, and deployment stack powering FedSecureAI.
        </p>
      </div>

      {/* End-to-End System Workflow */}
      <section className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-5 sm:p-6 shadow-sm space-y-5">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-3">
          <h2 className="text-base font-semibold text-slate-900 dark:text-white">
            End-to-End Prediction Pipeline
          </h2>
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
            Client-side CSV &rarr; FastAPI &rarr; Sequence Inference &rarr; Triage Outputs
          </span>
        </div>

        {/* Desktop horizontal flow */}
        <div className="hidden md:flex items-center justify-between gap-2">
          {WORKFLOW_STEPS.map((step, idx) => (
            <React.Fragment key={step.title}>
              <div className="flex-1 min-w-0 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/70 p-3.5 text-center space-y-1.5">
                <div className="w-9 h-9 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center justify-center mx-auto">
                  {step.icon}
                </div>
                <div className="text-xs font-semibold text-slate-900 dark:text-white truncate">
                  {step.title}
                </div>
                <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400 truncate">
                  {step.detail}
                </div>
              </div>
              {idx < WORKFLOW_STEPS.length - 1 && (
                <ArrowRight className="w-4 h-4 text-slate-400 shrink-0" />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Mobile vertical stepper */}
        <div className="md:hidden space-y-2">
          {WORKFLOW_STEPS.map((step, idx) => (
            <React.Fragment key={step.title}>
              <div className="flex items-center gap-3 rounded-xl border border-slate-200 dark:border-slate-800 bg-slate-50 dark:bg-slate-950/70 p-3">
                <div className="w-9 h-9 rounded-lg bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 flex items-center justify-center shrink-0">
                  {step.icon}
                </div>
                <div className="min-w-0">
                  <div className="text-xs font-semibold text-slate-900 dark:text-white">
                    {idx + 1}. {step.title}
                  </div>
                  <div className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                    {step.detail}
                  </div>
                </div>
              </div>
              {idx < WORKFLOW_STEPS.length - 1 && (
                <div className="flex justify-center">
                  <ArrowDown className="w-4 h-4 text-slate-500" />
                </div>
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Mini Federated Learning Illustration */}
        <div className="rounded-xl border border-indigo-200 dark:border-indigo-900/50 bg-indigo-50/60 dark:bg-indigo-950/20 p-4 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-start gap-3">
            <ShieldCheck className="w-5 h-5 text-indigo-500 shrink-0 mt-0.5" />
            <div>
              <h3 className="text-sm font-semibold text-slate-900 dark:text-white">
                How Federated Learning Protects Bank Data
              </h3>
              <p className="text-xs text-slate-600 dark:text-slate-300 mt-0.5">
                Each bank trains locally on its own transactions and shares only model weights (FedAvg / FedProx), never raw customer records.
              </p>
            </div>
          </div>

          <svg
            viewBox="0 0 320 72"
            className="w-full max-w-[320px] h-18 shrink-0 mx-auto md:mx-0"
            role="img"
            aria-label="Federated learning weight aggregation diagram"
          >
            <rect x="4" y="6" width="76" height="26" rx="6" fill="#1e293b" stroke="#475569" />
            <text x="42" y="23" fill="#e2e8f0" fontSize="10" fontFamily="monospace" textAnchor="middle">Bank A</text>

            <rect x="4" y="40" width="76" height="26" rx="6" fill="#1e293b" stroke="#475569" />
            <text x="42" y="57" fill="#e2e8f0" fontSize="10" fontFamily="monospace" textAnchor="middle">Bank B</text>

            <line x1="80" y1="19" x2="128" y2="33" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="3 2" />
            <line x1="80" y1="53" x2="128" y2="39" stroke="#6366f1" strokeWidth="1.5" strokeDasharray="3 2" />
            <text x="103" y="16" fill="#818cf8" fontSize="8" fontFamily="monospace" textAnchor="middle">Δw₁</text>
            <text x="103" y="62" fill="#818cf8" fontSize="8" fontFamily="monospace" textAnchor="middle">Δw₂</text>

            <rect x="128" y="18" width="84" height="36" rx="8" fill="#312e81" stroke="#6366f1" strokeWidth="1.5" />
            <text x="170" y="34" fill="#e0e7ff" fontSize="9.5" fontFamily="monospace" fontWeight="bold" textAnchor="middle">FedAvg</text>
            <text x="170" y="46" fill="#a5b4fc" fontSize="8" fontFamily="monospace" textAnchor="middle">Aggregator</text>

            <line x1="212" y1="36" x2="238" y2="36" stroke="#10b981" strokeWidth="1.5" />
            <rect x="238" y="18" width="78" height="36" rx="8" fill="#064e3b" stroke="#10b981" strokeWidth="1.5" />
            <text x="277" y="34" fill="#d1fae5" fontSize="9.5" fontFamily="monospace" fontWeight="bold" textAnchor="middle">Global</text>
            <text x="277" y="46" fill="#6ee7b7" fontSize="8" fontFamily="monospace" textAnchor="middle">LSTM</text>
          </svg>
        </div>
      </section>

      {/* Categorized Technology Grid */}
      <div className="space-y-6">
        {TECH_CATEGORIES.map((category) => (
          <section key={category.title} className="space-y-3">
            <h2 className="text-base font-semibold text-slate-900 dark:text-white">
              {category.title}
            </h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3.5">
              {category.items.map((item) => (
                <div
                  key={item.name}
                  className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 shadow-sm flex items-start gap-3"
                >
                  <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 flex items-center justify-center shrink-0">
                    {item.simpleIcon ? <BrandLogo icon={item.simpleIcon} /> : item.lucideIcon}
                  </div>
                  <div className="min-w-0">
                    <h3 className="text-sm font-semibold text-slate-900 dark:text-white truncate">
                      {item.name}
                    </h3>
                    <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5 leading-snug">
                      {item.summary}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </section>
        ))}
      </div>

      {/* Concise Future Work Note */}
      <p className="text-xs text-slate-500 dark:text-slate-400 text-center pt-2">
        Future work: differential privacy, secure aggregation.
      </p>
    </div>
  );
};
