import React, { useState } from 'react';
import {
  Cpu,
  ArrowRight,
  CheckCircle2,
  AlertCircle,
  Package,
  Clock,
  Layers,
  Terminal,
  ExternalLink,
} from 'lucide-react';

type TechStatus =
  | 'Implemented and active'
  | 'Implemented but optional'
  | 'Present as a dependency but not confirmed in use'
  | 'Planned or not implemented';

interface TechItem {
  id: string;
  name: string;
  category: string;
  status: TechStatus;
  pipelineStep: string;
  description: string;
  usedIn: string;
  howToVerify: string;
  limitation: string;
  badgeText: string;
  accentColor: string;
  officialUrl: string;
  logoSvg: React.ReactNode;
}

const PIPELINE_STEPS = [
  'All Steps',
  '1. CSV Upload',
  '2. Validation & Preprocessing',
  '3. Temporal Sequence Features',
  '4. Model Inference & Federated Components',
  '5. Threshold Decision',
  '6. Explanations',
  '7. Reporting & History',
  '8. Deployment & Testing',
];

const TechLogo: React.FC<{ label: string; bg: string; fg?: string }> = ({ label, bg, fg = '#ffffff' }) => (
  <svg
    viewBox="0 0 40 40"
    className="w-10 h-10 rounded-xl shrink-0 shadow-sm"
    role="img"
    aria-label={`${label} technology emblem`}
  >
    <rect width="40" height="40" rx="10" fill={bg} />
    <text
      x="50%"
      y="54%"
      dominantBaseline="middle"
      textAnchor="middle"
      fill={fg}
      fontSize="12"
      fontWeight="800"
      fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
    >
      {label}
    </text>
  </svg>
);

const TECH_ITEMS: TechItem[] = [
  {
    id: 'react-vite-ts',
    name: 'React 18 + TypeScript + Vite + Tailwind CSS',
    category: 'Frontend UI & Build',
    status: 'Implemented and active',
    pipelineStep: '1. CSV Upload',
    description:
      'Builds the responsive single-page dashboard for CSV inspection, async job progress polling, threshold visualization, and downloads.',
    usedIn: 'frontend/src/App.tsx, frontend/src/pages/AmlEvaluationPage.tsx, frontend/src/services/apiService.ts, frontend/package.json',
    howToVerify: 'Open the AML Evaluation tab, click "Load Demo Sample (20 Rows)", and inspect network calls in Browser DevTools.',
    limitation: 'Requires VITE_API_URL / VITE_API_BASE_URL to be baked in at Vercel build time.',
    badgeText: 'TS/React',
    accentColor: '#38bdf8',
    officialUrl: 'https://react.dev',
    logoSvg: <TechLogo label="TS" bg="#0284c7" />,
  },
  {
    id: 'fastapi-uvicorn',
    name: 'FastAPI + Uvicorn (ASGI)',
    category: 'Backend Web API',
    status: 'Implemented and active',
    pipelineStep: '1. CSV Upload',
    description:
      'Serves REST endpoints for schema inspection, multipart/JSON CSV upload, async job submission/polling, and health/readiness probes.',
    usedIn: 'asgi.py, backend/interfaces/api/app.py, backend/interfaces/api/routes/{health,evaluation,inference,research}.py',
    howToVerify: 'Open /health or /ready on the backend URL and observe HTTP 200 JSON responses.',
    limitation: 'Async job state is stored in single-worker memory (--workers 1) plus disk JSON persistence.',
    badgeText: 'API',
    accentColor: '#10b981',
    officialUrl: 'https://fastapi.tiangolo.com',
    logoSvg: <TechLogo label="API" bg="#059669" />,
  },
  {
    id: 'pandas-numpy',
    name: 'pandas + NumPy',
    category: 'Data Parsing & Feature Engineering',
    status: 'Implemented and active',
    pipelineStep: '2. Validation & Preprocessing',
    description:
      'Detects delimiters/encodings, adapts column schemas, sorts chronologically, applies log/standardization transforms, and exports filtered CSVs.',
    usedIn: 'backend/infrastructure/data/readers/universal_csv_reader.py, backend/interfaces/api/services/csv_evaluation_service.py',
    howToVerify: 'Upload any CSV on AML Evaluation to see detected delimiter, encoding, and column mapping badges.',
    limitation: 'Large CSVs must stay within host RAM; chunked inference avoids allocating full 3D arrays at once.',
    badgeText: 'PD/NP',
    accentColor: '#6366f1',
    officialUrl: 'https://pandas.pydata.org',
    logoSvg: <TechLogo label="PD" bg="#4f46e5" />,
  },
  {
    id: 'sequence-features',
    name: 'AccountSequenceDataset (10x57 Temporal Windows)',
    category: 'Temporal Feature Pipeline',
    status: 'Implemented and active',
    pipelineStep: '3. Temporal Sequence Features',
    description:
      'Constructs up to 10-step causal per-account transaction sequences (57 one-hot + continuous features) in 4,096-row float32 chunks.',
    usedIn: 'backend/infrastructure/data/sequences/account_sequence_dataset.py, _construct_sequences_and_infer in csv_evaluation_service.py',
    howToVerify: 'Run evaluation and inspect sequence_context_diagnostics (mean history length & padding ratio) on Results & Analytics.',
    limitation: 'Single-transaction slices without repeated account history receive padded sequences and conservative baseline scores.',
    badgeText: 'SEQ',
    accentColor: '#8b5cf6',
    officialUrl: 'https://pytorch.org',
    logoSvg: <TechLogo label="SEQ" bg="#7c3aed" />,
  },
  {
    id: 'pytorch-lstm',
    name: 'PyTorch (LSTM / BiLSTM / GRU Sequence Models)',
    category: 'Deep Learning Inference',
    status: 'Implemented and active',
    pipelineStep: '4. Model Inference & Federated Components',
    description:
      'Loads SHA-256 verified .pt checkpoints (HI-Small LSTM FedAvg IID, LI-Medium BiLSTM FedAvg IID) and executes batched torch.inference_mode() scoring.',
    usedIn: 'backend/infrastructure/models/pytorch/sequence_classifier.py, backend/interfaces/api/services/model_service.py',
    howToVerify: 'Check /ready loaded_models and run Execute AML Prediction to score transactions.',
    limitation: 'ModelService caps cached models in RAM (max 2) to stay memory-safe on free hosting.',
    badgeText: 'PT',
    accentColor: '#f97316',
    officialUrl: 'https://pytorch.org',
    logoSvg: <TechLogo label="PT" bg="#ea580c" />,
  },
  {
    id: 'fedavg-fedprox',
    name: 'FedAvg & FedProx (Simulated Federated Training)',
    category: 'Federated Learning Optimization',
    status: 'Implemented and active',
    pipelineStep: '4. Model Inference & Federated Components',
    description:
      'Offline federated training harness comparing FedAvg and FedProx (mu=0.0001) across IID and Non-IID bank client partitions.',
    usedIn: 'backend/infrastructure/federated/, backend/results/phase8/phase8_summary.csv, backend/results/phase7/',
    howToVerify: 'Open the Model Comparison and Ensemble Analytics tabs to inspect all 24 federated + 3 centralized benchmark runs.',
    limitation: 'Trained in offline multi-client simulation rather than live cross-institution network orchestration.',
    badgeText: 'FL',
    accentColor: '#ec4899',
    officialUrl: 'https://arxiv.org/abs/1602.05629',
    logoSvg: <TechLogo label="FL" bg="#db2777" />,
  },
  {
    id: 'scikit-learn',
    name: 'scikit-learn',
    category: 'Metrics & Thresholding',
    status: 'Implemented and active',
    pipelineStep: '5. Threshold Decision',
    description:
      'Computes PR-AUC, ROC-AUC, precision/recall/F1, and applies validation-calibrated frozen thresholds (e.g. θ=0.3378 for HI-Small).',
    usedIn: 'backend/evaluation/metrics.py, backend/interfaces/api/services/csv_evaluation_service.py',
    howToVerify: 'Upload a labeled CSV (like the 20-row demo sample) and view PR-AUC, ROC-AUC, and confusion counts.',
    limitation: 'Small samples (<30 positives) trigger statistical variance warnings.',
    badgeText: 'SK',
    accentColor: '#f59e0b',
    officialUrl: 'https://scikit-learn.org',
    logoSvg: <TechLogo label="SK" bg="#d97706" />,
  },
  {
    id: 'integrated-gradients',
    name: 'Sequence Integrated Gradients',
    category: 'Explainability (XAI)',
    status: 'Implemented and active',
    pipelineStep: '6. Explanations',
    description:
      'Computes per-feature attributions on PyTorch sequence predictions for any inspected transaction row.',
    usedIn: 'backend/infrastructure/explainability/integrated_gradients.py, backend/interfaces/api/services/explain_service.py',
    howToVerify: 'In Prediction History, click the Eye ("Inspect") button on any transaction row to view positive/negative feature drivers.',
    limitation: 'Requires PyTorch sequence inputs; tabular fallback models use a separate attribution path.',
    badgeText: 'IG',
    accentColor: '#14b8a6',
    officialUrl: 'https://arxiv.org/abs/1703.01365',
    logoSvg: <TechLogo label="IG" bg="#0d9488" />,
  },
  {
    id: 'reportlab-matplotlib',
    name: 'ReportLab + Matplotlib',
    category: 'PDF Reporting & Charts',
    status: 'Implemented and active',
    pipelineStep: '7. Reporting & History',
    description:
      'Generates multi-page metrics-only PDF audit reports with PR/ROC curves, calibration plots, and transaction risk heatmaps.',
    usedIn: 'backend/infrastructure/evaluation/pdf_report_generator.py, backend/interfaces/api/routes/evaluation.py',
    howToVerify: 'Open Prediction History after running an evaluation and click "Report (Metrics PDF)".',
    limitation: 'Itemized transaction rows are exported in separate Flagged / Under Review / Legitimate CSVs rather than inside the PDF.',
    badgeText: 'PDF',
    accentColor: '#e11d48',
    officialUrl: 'https://www.reportlab.com',
    logoSvg: <TechLogo label="PDF" bg="#be123c" />,
  },
  {
    id: 'pytest',
    name: 'pytest',
    category: 'Automated Testing',
    status: 'Implemented and active',
    pipelineStep: '8. Deployment & Testing',
    description:
      'Verifies CSV evaluation, schema adaptation, threshold parity, and API contracts across unit and system test suites.',
    usedIn: 'pytest.ini, backend/tests/unit/, backend/tests/system/test_phase17_csv_evaluation.py',
    howToVerify: 'Run pytest backend/tests/system/test_phase17_csv_evaluation.py -q (17 tests passing).',
    limitation: 'Some legacy integration tests reference pre-monorepo relative paths.',
    badgeText: 'TEST',
    accentColor: '#22c55e',
    officialUrl: 'https://docs.pytest.org',
    logoSvg: <TechLogo label="PYT" bg="#16a34a" />,
  },
  {
    id: 'tabular-fallback',
    name: 'Tabular Fallback & Mode B Custom Trainer',
    category: 'Optional Classification Mode',
    status: 'Implemented but optional',
    pipelineStep: '4. Model Inference & Federated Components',
    description:
      'Optional HistGradientBoosting / RandomForest / LightGBM fallback for single-transaction CSVs lacking account history.',
    usedIn: 'backend/infrastructure/models/tabular_fallback.py, backend/infrastructure/models/dynamic_model_trainer.py',
    howToVerify: 'Use Mode B custom training on the AML Evaluation page with a labeled CSV.',
    limitation: 'Lazy-loaded on demand so startup avoids unnecessary memory or blocked DLLs.',
    badgeText: 'OPT',
    accentColor: '#64748b',
    officialUrl: 'https://scikit-learn.org',
    logoSvg: <TechLogo label="OPT" bg="#475569" />,
  },
  {
    id: 'docker',
    name: 'Docker (Hugging Face Spaces Target)',
    category: 'Container Packaging',
    status: 'Implemented but optional',
    pipelineStep: '8. Deployment & Testing',
    description:
      'Packages Python 3.11-slim + CPU-only PyTorch + FastAPI backend on port 7860 for 16 GB free-tier Hugging Face Spaces.',
    usedIn: 'Dockerfile, .dockerignore, deploy/hf_space/README.md, scripts/push_to_hf_space.ps1',
    howToVerify: 'Inspect Dockerfile and run scripts/push_to_hf_space.ps1 to deploy the container.',
    limitation: 'Live production currently runs on Render/Vercel until pushed to your HF Space.',
    badgeText: 'DOC',
    accentColor: '#0ea5e9',
    officialUrl: 'https://www.docker.com',
    logoSvg: <TechLogo label="DKR" bg="#0369a1" />,
  },
  {
    id: 'ruff-mypy',
    name: 'Ruff & mypy',
    category: 'Developer Linting & Typing',
    status: 'Implemented but optional',
    pipelineStep: '8. Deployment & Testing',
    description:
      'Configured in pyproject.toml for Python linting (PEP8/docstrings/pathlib) and strict static type checking.',
    usedIn: 'pyproject.toml ([tool.ruff], [tool.mypy])',
    howToVerify: 'Inspect pyproject.toml in the repository root.',
    limitation: 'Developer CLI tools; not executed at API runtime.',
    badgeText: 'LINT',
    accentColor: '#64748b',
    officialUrl: 'https://docs.astral.sh/ruff/',
    logoSvg: <TechLogo label="RF" bg="#334155" />,
  },
  {
    id: 'shap-gh-actions',
    name: 'SHAP / GitHub Actions',
    category: 'Dependencies / Stubs',
    status: 'Present as a dependency but not confirmed in use',
    pipelineStep: '6. Explanations',
    description:
      'explain_service.py tags tabular fallback attributions as tree_shap, and .github/workflows/.gitkeep exists without active .yml pipelines.',
    usedIn: 'backend/interfaces/api/services/explain_service.py, .github/workflows/.gitkeep',
    howToVerify: 'Check .github/workflows/ and requirements.txt (no shap package pinned; no CI workflow files).',
    limitation: 'Sequence explanations use Integrated Gradients, not the shap library; CI runs manually/locally.',
    badgeText: 'STUB',
    accentColor: '#d97706',
    officialUrl: 'https://github.com/features/actions',
    logoSvg: <TechLogo label="CI" bg="#b45309" />,
  },
  {
    id: 'dp-secagg-mlflow-dvc',
    name: 'Differential Privacy, Secure Aggregation, MLflow, DVC',
    category: 'Not Implemented in Repo',
    status: 'Planned or not implemented',
    pipelineStep: '4. Model Inference & Federated Components',
    description:
      'Explicitly NOT active in this codebase: no formal DP noise, no cryptographic secure aggregation, and no MLflow or DVC tracking.',
    usedIn: 'Not present in requirements.txt, pyproject.toml, or runtime services',
    howToVerify: 'Search the repository for mlflow, dvc, or opacus — zero runtime integrations exist.',
    limitation: 'Privacy in FedSecureAI relies on keeping raw rows local during offline FL simulation rather than formal DP/MPC.',
    badgeText: 'NONE',
    accentColor: '#e11d48',
    officialUrl: 'https://arxiv.org/abs/2603.19314',
    logoSvg: <TechLogo label="N/A" bg="#9f1239" />,
  },
];

const STATUS_BADGES: Record<TechStatus, { cls: string; icon: React.ReactNode }> = {
  'Implemented and active': {
    cls: 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60',
    icon: <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />,
  },
  'Implemented but optional': {
    cls: 'bg-sky-950/80 text-sky-300 border-sky-700/60',
    icon: <Package className="w-3.5 h-3.5 text-sky-400 shrink-0" />,
  },
  'Present as a dependency but not confirmed in use': {
    cls: 'bg-amber-950/80 text-amber-300 border-amber-700/60',
    icon: <AlertCircle className="w-3.5 h-3.5 text-amber-400 shrink-0" />,
  },
  'Planned or not implemented': {
    cls: 'bg-rose-950/80 text-rose-300 border-rose-700/60',
    icon: <Clock className="w-3.5 h-3.5 text-rose-400 shrink-0" />,
  },
};

export const TechnologyStackPage: React.FC = () => {
  const [selectedStep, setSelectedStep] = useState<string>('All Steps');
  const [statusFilter, setStatusFilter] = useState<'All' | TechStatus>('All');

  const filteredTech = TECH_ITEMS.filter((item) => {
    if (selectedStep !== 'All Steps' && item.pipelineStep !== selectedStep) return false;
    if (statusFilter !== 'All' && item.status !== statusFilter) return false;
    return true;
  });

  return (
    <div className="space-y-8 animate-fade-in-up">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white tracking-tight">
              Technology Stack & System Walkthrough
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-1">
              Audited inventory of technologies verified in the FedSecureAI codebase, mapped to the actual end-to-end execution pipeline.
            </p>
          </div>
        </div>
      </div>

      {/* End-to-End Pipeline Walkthrough */}
      <div className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-5 sm:p-6 shadow-xl space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div className="flex items-center space-x-2">
            <Layers className="w-5 h-5 text-indigo-400" />
            <h2 className="text-base font-bold text-slate-900 dark:text-white">
              End-to-End Execution Pipeline (Click a Step to Filter)
            </h2>
          </div>
          <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
            Showing {filteredTech.length} of {TECH_ITEMS.length} audited items
          </span>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {PIPELINE_STEPS.map((step, idx) => (
            <React.Fragment key={step}>
              <button
                type="button"
                onClick={() => setSelectedStep(step)}
                className={`px-3 py-2 rounded-xl text-xs font-semibold transition cursor-pointer border ${
                  selectedStep === step
                    ? 'bg-indigo-600 text-white border-indigo-500 shadow-sm'
                    : 'bg-slate-50 dark:bg-slate-950 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-800 hover:border-indigo-500/50'
                }`}
              >
                {step}
              </button>
              {idx > 0 && idx < PIPELINE_STEPS.length - 1 && (
                <ArrowRight className="w-3.5 h-3.5 text-slate-400 hidden sm:inline shrink-0" />
              )}
            </React.Fragment>
          ))}
        </div>

        {/* Honesty Classification Filter */}
        <div className="flex flex-wrap items-center gap-1.5 pt-3 border-t border-slate-200 dark:border-slate-800">
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400 mr-1">
            Classification:
          </span>
          {(
            [
              'All',
              'Implemented and active',
              'Implemented but optional',
              'Present as a dependency but not confirmed in use',
              'Planned or not implemented',
            ] as const
          ).map((st) => (
            <button
              key={st}
              type="button"
              onClick={() => setStatusFilter(st)}
              className={`px-2.5 py-1 rounded-lg text-[11px] font-semibold transition cursor-pointer border ${
                statusFilter === st
                  ? 'bg-slate-900 dark:bg-indigo-600 text-white border-indigo-500'
                  : 'bg-slate-100 dark:bg-slate-950 text-slate-600 dark:text-slate-400 border-slate-200 dark:border-slate-800 hover:text-slate-900 dark:hover:text-white'
              }`}
            >
              {st}
            </button>
          ))}
        </div>
      </div>

      {/* Technology Cards Grid */}
      {filteredTech.length === 0 ? (
        <div className="p-8 text-center rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 text-slate-500 text-sm">
          No technologies match the selected pipeline step and classification filter.
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {filteredTech.map((tech) => {
            const stStyle = STATUS_BADGES[tech.status];
            return (
              <article
                key={tech.id}
                className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-5 shadow-lg hover:border-indigo-500/40 transition-colors flex flex-col justify-between space-y-4"
              >
                <div className="space-y-3">
                  <div className="flex items-start justify-between gap-3">
                    <div className="flex items-center space-x-3">
                      {tech.logoSvg}
                      <div>
                        <h3 className="text-base font-bold text-slate-900 dark:text-white leading-snug">
                          {tech.name}
                        </h3>
                        <p className="text-xs text-slate-500 dark:text-slate-400">
                          {tech.category} · {tech.pipelineStep}
                        </p>
                      </div>
                    </div>
                    <a
                      href={tech.officialUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      aria-label={`Official site for ${tech.name}`}
                      className="p-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:text-indigo-500 transition shrink-0"
                    >
                      <ExternalLink className="w-3.5 h-3.5" />
                    </a>
                  </div>

                  <div>
                    <span
                      className={`inline-flex items-center space-x-1.5 px-2.5 py-1 rounded-full text-[11px] font-mono border ${stStyle.cls}`}
                    >
                      {stStyle.icon}
                      <span>{tech.status}</span>
                    </span>
                  </div>

                  <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
                    {tech.description}
                  </p>
                </div>

                <div className="space-y-2 pt-3 border-t border-slate-200 dark:border-slate-800 text-xs">
                  <div>
                    <span className="font-mono font-bold text-slate-500 dark:text-slate-400 uppercase text-[10px]">
                      Where Used:{' '}
                    </span>
                    <code className="font-mono text-[11px] text-indigo-600 dark:text-indigo-300 break-all">
                      {tech.usedIn}
                    </code>
                  </div>
                  <div className="flex items-start space-x-1.5">
                    <Terminal className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                    <span className="text-slate-700 dark:text-slate-300">
                      <strong className="text-emerald-600 dark:text-emerald-400">Verify:</strong> {tech.howToVerify}
                    </span>
                  </div>
                  <div className="text-slate-600 dark:text-slate-400">
                    <strong className="text-amber-600 dark:text-amber-400">Note / Limitation:</strong> {tech.limitation}
                  </div>
                </div>
              </article>
            );
          })}
        </div>
      )}

      {/* Attribution Note */}
      <p className="text-[11px] text-slate-500 dark:text-slate-400 text-center">
        Clean vector emblems rendered inline with accessible labels; external links open official project documentation.
      </p>
    </div>
  );
};
