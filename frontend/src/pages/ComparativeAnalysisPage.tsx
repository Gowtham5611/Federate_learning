import React, { useState } from 'react';
import {
  BookOpen,
  ExternalLink,
  AlertCircle,
  GitCompare,
  ArrowRight,
  ArrowDown,
  ShieldCheck,
  FileText,
  Server,
  BarChart3,
  Maximize2,
  X,
  CheckCircle2,
} from 'lucide-react';

interface ComparisonItem {
  id: string;
  name: string;
  year?: number;
  status: string;
  limitation: string;
  connection: string;
  url: string;
  linkLabel: 'View paper' | 'View system';
  kind: 'paper' | 'system';
}

interface VisualGalleryItem {
  id: string;
  title: string;
  subtitle: string;
  imageSrc: string;
  imageAlt: string;
  badge: string;
  bullets: { heading: string; text: string }[];
}

const COMPARISON_ITEMS: ComparisonItem[] = [
  {
    id: 'pricing-cross-bank',
    name: 'Pricing Cross-Bank Context in Federated Anti-Money Laundering',
    year: 2026,
    status: 'arXiv preprint',
    limitation:
      'Relies on graph-based cross-bank context modeling in a preprint study without a CSV triage workflow.',
    connection:
      'FedSecureAI uses 10-step per-account sequence models (LSTM/BiLSTM) and CSV triage; scores are not directly comparable.',
    url: 'https://arxiv.org/abs/2610.35996',
    linkLabel: 'View paper',
    kind: 'paper',
  },
  {
    id: 'dpxfin',
    name: 'DPxFin: Adaptive Differential Privacy for Anti-Money Laundering Detection via Reputation-Weighted Federated Learning',
    year: 2026,
    status: 'arXiv preprint',
    limitation:
      'Introduces a privacy-utility trade-off through adaptive differential privacy noise and reputation weighting.',
    connection:
      'FedSecureAI does not inject active differential privacy noise in training (unnoised FedAvg/FedProx only); formal DP remains an open gap.',
    url: 'https://arxiv.org/abs/2603.19314',
    linkLabel: 'View paper',
    kind: 'paper',
  },
  {
    id: 'ssrn-4669561',
    name: 'A Federated Machine Learning Approaches For Anti-Money Laundering Detection',
    year: 2023,
    status: 'Conference paper record on SSRN',
    limitation:
      'Evaluates tree-based models on the synthetic PaySim mobile-money dataset with a different split and metric setup.',
    connection:
      'FedSecureAI evaluates sequence models on IBM AML HI-Small artifacts; metrics cannot be compared across different datasets or protocols.',
    url: 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4669561',
    linkLabel: 'View paper',
    kind: 'paper',
  },
  {
    id: 'flower-fraud',
    name: 'Flower Federated Financial Fraud Detection',
    status: 'Open-source example',
    limitation:
      'Starter federated example scoped to local Flower simulation rather than full AML CSV evaluation and reporting.',
    connection:
      'FedSecureAI provides an interactive FastAPI + React CSV inspection, thresholded triage, Integrated Gradients, and PDF/CSV export workflow.',
    url: 'https://flower.ai/apps/mainthread/federated-fraud-detection',
    linkLabel: 'View system',
    kind: 'system',
  },
  {
    id: 'consilient-core',
    name: 'Consilient Core AML/CFT Model',
    status: 'Commercial system',
    limitation:
      'Proprietary platform with limited public implementation details and no open benchmark metrics.',
    connection:
      'FedSecureAI is an open research prototype using simulated single-host FL on public IBM AML data, not a live commercial cross-bank deployment.',
    url: 'https://consilient.com/our-products/core-aml-cft-model',
    linkLabel: 'View system',
    kind: 'system',
  },
];

const VISUAL_GALLERY: VisualGalleryItem[] = [
  {
    id: 'non-iid-robustness',
    title: '1. Robustness Under Non-IID Skew (Relative Retention %)',
    subtitle: 'Comparing IID-to-Non-IID retention across simulated banking partitions and reference studies.',
    imageSrc: '/comparative/non_iid_robustness.png',
    imageAlt: 'Robustness Under Non-IID Skew (Relative Retention %)',
    badge: '94.6% PR-AUC Retention',
    bullets: [
      {
        heading: 'What the graph shows',
        text: 'FedSecureAI retains 94.6% of its IID PR-AUC (0.4505 vs. 0.4760) under typology-skewed Non-IID splits, compared with 78.7% for FedGAT (M=32) and 3.6% for edge-dropping FedAvg.',
      },
      {
        heading: 'How we solved Non-IID data',
        text: 'We built causal 10-step per-account transaction sequences (57 engineered features) so each simulated bank preserves temporal laundering context locally without cross-bank raw data pooling.',
      },
      {
        heading: 'Why FedAvg beat FedProx',
        text: 'Our experiments showed FedProx (μ = 0.0001) pulled weights toward the dominant legitimate class (-40.7% PR-AUC drop), so we selected sample-weighted FedAvg as the champion aggregator.',
      },
      {
        heading: 'No artificial oversampling bias',
        text: 'Unlike DPxFin (which uses SMOTE to balance >7.5M rows), FedSecureAI evaluates on the raw 0.177% positive prevalence of IBM HI-Small using PR-AUC and validation-tuned thresholding (θ = 0.3378).',
      },
      {
        heading: 'Exam comparison note',
        text: 'Retention percentages reflect within-study IID vs. Non-IID changes across different datasets, models, and metrics—never a head-to-head ranking across papers.',
      },
    ],
  },
  {
    id: 'pipeline-coverage',
    title: '2. End-to-End Pipeline Coverage Matrix',
    subtitle: 'Mapping implemented workflow stages from federated training through analyst triage and audit exports.',
    imageSrc: '/comparative/pipeline_coverage_matrix.png',
    imageAlt: 'End-to-End Pipeline Coverage Matrix',
    badge: 'Full Evaluation & Triage Loop',
    bullets: [
      {
        heading: 'No raw data sharing',
        text: 'Simulated FedAvg/FedProx clients train locally on partitioned bank splits and exchange only model weights, keeping raw customer transaction rows isolated.',
      },
      {
        heading: 'Leakage-free chronological split',
        text: 'Implemented a strict 60/20/20 chronological train/validation/test split on IBM AML HI-Small, freezing the validation-optimal threshold (θ = 0.3378) before held-out testing.',
      },
      {
        heading: 'Feature attribution (XAI)',
        text: 'Integrated Gradients explains 10-step LSTM/BiLSTM predictions per transaction, while Tree SHAP explains the LightGBM tabular fallback when account history is unavailable.',
      },
      {
        heading: 'Alert triage & drift checks',
        text: 'The React + FastAPI dashboard flags suspicious transactions, isolates missed laundering cases (Under Review), and runs per-upload prior-shift and sequence-depth diagnostics.',
      },
      {
        heading: 'PDF & CSV audit exports',
        text: 'Prediction History generates ReportLab metrics PDFs alongside separate Flagged, Under Review, and Legitimate CSV downloads for compliance review.',
      },
    ],
  },
  {
    id: 'capability-readiness',
    title: '3. System Capability & Compliance Readiness (1–5 Scale)',
    subtitle: 'Illustrative six-axis readiness overview connecting technical modeling with operational AML triage.',
    imageSrc: '/comparative/capability_readiness_radar.jpg',
    imageAlt: 'System Capability & Compliance Readiness (1-5 Scale)',
    badge: '6-Axis Readiness Overview',
    bullets: [
      {
        heading: 'Imbalanced metric rigor',
        text: 'Evaluated using PR-AUC (0.4760), ROC-AUC (0.9555), F1 (0.5359), precision (0.8183), recall (0.3984), Brier score (0.0012), and bounded alerts (8.62 per 10k) instead of misleading accuracy.',
      },
      {
        heading: 'Temporal sequence modeling',
        text: 'PyTorch LSTM/GRU/BiLSTM models capture multi-hop transaction sequences per sending account, with an automatic LightGBM fallback for single-row or zero-history CSV uploads.',
      },
      {
        heading: 'Non-IID resilience & ensemble audit',
        text: 'We tested IID vs. Non-IID blending via an α-sweep on validation data, confirming α* = 1.00 (IID champion alone) prevented negative transfer from typology skew.',
      },
      {
        heading: 'Explainability & drift awareness',
        text: 'Combines Integrated Gradients / Tree SHAP explanations with calibration metrics (Brier/ECE) and offline client-drift tracking modules across federated rounds.',
      },
      {
        heading: 'Honest privacy scope for viva',
        text: 'Federated training is simulated on a single host with weight-only averaging; active differential privacy noise (DP-SGD) and secure aggregation remain documented future work.',
      },
    ],
  },
];

export const ComparativeAnalysisPage: React.FC = () => {
  const [lightboxImage, setLightboxImage] = useState<VisualGalleryItem | null>(null);

  return (
    <div className="space-y-8 animate-fade-in-up motion-reduce:animate-none">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <div className="flex items-start sm:items-center gap-3">
          <div className="p-2.5 rounded-xl bg-indigo-600/15 border border-indigo-500/30 text-indigo-500 dark:text-indigo-400 shrink-0">
            <BookOpen className="w-6 h-6" />
          </div>
          <div className="min-w-0">
            <h1 className="text-2xl sm:text-3xl font-bold text-slate-900 dark:text-white tracking-tight">
              Comparative Analysis
            </h1>
            <p className="text-sm text-slate-600 dark:text-slate-400 mt-1">
              Visual comparison of FedSecureAI against five reference preprints, conference records, open-source examples, and commercial systems.
            </p>
          </div>
        </div>
      </div>

      {/* FedSecureAI Research Prototype Summary */}
      <section className="rounded-2xl border border-indigo-500/40 bg-indigo-50/60 dark:bg-indigo-950/25 p-4 sm:p-5 space-y-2">
        <div className="flex flex-wrap items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-indigo-500 dark:text-indigo-400 shrink-0" />
          <h2 className="text-sm font-semibold text-slate-900 dark:text-white">
            FedSecureAI (This Project) · Research Prototype
          </h2>
          <span className="px-2 py-0.5 rounded-full text-[11px] font-mono bg-indigo-600 text-white">
            Simulated FL
          </span>
        </div>
        <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed">
          <strong className="text-slate-900 dark:text-white">Implemented:</strong> Simulated FedAvg/FedProx sequence models on IBM AML HI-Small, CSV schema adaptation, thresholded triage (θ = 0.3378), Integrated Gradients &amp; Tree SHAP, and PDF/CSV exports.{' '}
          <strong className="text-slate-900 dark:text-white">Future work:</strong> Active differential privacy noise, secure aggregation, and live cross-bank orchestration.
        </p>
      </section>

      {/* Visual Charts Showcase with 5 Bullet Points Each */}
      <section className="space-y-6">
        <div className="flex items-center justify-between gap-2">
          <div className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5 text-indigo-500 dark:text-indigo-400 shrink-0" />
            <h2 className="text-lg font-bold text-slate-900 dark:text-white">
              Visual Comparison &amp; Project Defense Breakdown
            </h2>
          </div>
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400 hidden sm:inline">
            Click any chart to enlarge
          </span>
        </div>

        <div className="space-y-6">
          {VISUAL_GALLERY.map((item) => (
            <article
              key={item.id}
              className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 sm:p-6 shadow-sm hover:shadow-md transition-shadow motion-reduce:transition-none space-y-5"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 dark:border-slate-800 pb-3">
                <div>
                  <h3 className="text-base sm:text-lg font-semibold text-slate-900 dark:text-white">
                    {item.title}
                  </h3>
                  <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-0.5">
                    {item.subtitle}
                  </p>
                </div>
                <span className="inline-flex items-center px-2.5 py-1 rounded-full text-[11px] font-mono font-semibold bg-indigo-500/15 text-indigo-700 dark:text-indigo-300 border border-indigo-500/30 shrink-0 self-start sm:self-center">
                  {item.badge}
                </span>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-center">
                {/* Chart Image with Crisp Light Backdrop & Click-to-Zoom */}
                <div className="lg:col-span-6">
                  <button
                    type="button"
                    onClick={() => setLightboxImage(item)}
                    className="group relative w-full rounded-xl overflow-hidden bg-white border border-slate-200 dark:border-slate-700 p-2 sm:p-3 shadow-sm hover:border-indigo-500 transition-colors motion-reduce:transition-none cursor-zoom-in focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-500"
                    aria-label={`Enlarge ${item.title}`}
                  >
                    <img
                      src={item.imageSrc}
                      alt={item.imageAlt}
                      loading="lazy"
                      className="w-full h-auto rounded-lg object-contain transition-transform duration-300 group-hover:scale-[1.01] motion-reduce:transition-none motion-reduce:group-hover:scale-100"
                    />
                    <span className="absolute bottom-3 right-3 inline-flex items-center gap-1 px-2.5 py-1 rounded-lg bg-slate-900/80 text-white text-[11px] font-mono opacity-90 group-hover:opacity-100 transition-opacity">
                      <Maximize2 className="w-3 h-3" />
                      <span>Zoom</span>
                    </span>
                  </button>
                </div>

                {/* 5 Structured Bullet Points */}
                <div className="lg:col-span-6 space-y-2.5">
                  <h4 className="text-xs font-mono uppercase tracking-wider text-indigo-600 dark:text-indigo-400 font-semibold">
                    Key Takeaways &amp; How FedSecureAI Solved It
                  </h4>
                  <ul className="space-y-2.5">
                    {item.bullets.map((b, idx) => (
                      <li
                        key={idx}
                        className="flex items-start gap-2.5 rounded-xl bg-slate-50 dark:bg-slate-950/70 border border-slate-200/80 dark:border-slate-800/80 p-3 text-xs sm:text-sm text-slate-700 dark:text-slate-300 leading-relaxed"
                      >
                        <CheckCircle2 className="w-4 h-4 text-emerald-500 shrink-0 mt-0.5" />
                        <span>
                          <strong className="text-slate-900 dark:text-white">{b.heading}:</strong>{' '}
                          {b.text}
                        </span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </article>
          ))}
        </div>
      </section>

      {/* 5 Comparison Cards: Paper/System -> Main Limitation -> FedSecureAI Connection */}
      <section className="space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-white">
          Reference Literature &amp; Systems
        </h2>
        {COMPARISON_ITEMS.map((item) => (
          <article
            key={item.id}
            className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4 sm:p-5 shadow-sm transition-colors motion-reduce:transition-none space-y-4"
          >
            {/* Card Header: Name, Year/Status, Link */}
            <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-3">
              <div className="min-w-0 space-y-1.5">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-mono bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700">
                    {item.kind === 'paper' ? (
                      <FileText className="w-3 h-3 text-indigo-500 dark:text-indigo-400 shrink-0" />
                    ) : (
                      <Server className="w-3 h-3 text-emerald-500 dark:text-emerald-400 shrink-0" />
                    )}
                    <span>{item.year ? `${item.year} · ${item.status}` : item.status}</span>
                  </span>
                </div>
                <h3 className="text-base font-semibold text-slate-900 dark:text-white break-words leading-snug">
                  {item.name}
                </h3>
              </div>

              <a
                href={item.url}
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center justify-center gap-1.5 px-3.5 py-2 min-h-[44px] sm:min-h-[38px] rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white text-xs font-semibold shrink-0 transition motion-reduce:transition-none focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-indigo-400"
              >
                <span>{item.linkLabel}</span>
                <ExternalLink className="w-3.5 h-3.5 shrink-0" />
              </a>
            </div>

            {/* Visual Flow: Paper/System -> Main Limitation -> FedSecureAI Connection */}
            <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] items-stretch gap-3">
              <div className="rounded-xl border border-amber-200 dark:border-amber-900/50 bg-amber-50/70 dark:bg-amber-950/20 p-3.5 min-w-0 space-y-1">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-800 dark:text-amber-300">
                  <AlertCircle className="w-3.5 h-3.5 shrink-0" />
                  <span>Main Limitation / Context</span>
                </div>
                <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 break-words leading-relaxed">
                  {item.limitation}
                </p>
              </div>

              <div className="flex items-center justify-center text-slate-400 dark:text-slate-500">
                <ArrowRight className="hidden md:block w-4 h-4 shrink-0" aria-hidden="true" />
                <ArrowDown className="md:hidden w-4 h-4 shrink-0" aria-hidden="true" />
              </div>

              <div className="rounded-xl border border-emerald-200 dark:border-emerald-900/50 bg-emerald-50/70 dark:bg-emerald-950/20 p-3.5 min-w-0 space-y-1">
                <div className="flex items-center gap-1.5 text-xs font-semibold text-emerald-800 dark:text-emerald-300">
                  <GitCompare className="w-3.5 h-3.5 shrink-0" />
                  <span>FedSecureAI Connection</span>
                </div>
                <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300 break-words leading-relaxed">
                  {item.connection}
                </p>
              </div>
            </div>
          </article>
        ))}
      </section>

      {/* Non-Comparability Note */}
      <p className="text-xs text-slate-500 dark:text-slate-400">
        Note: Entries span different datasets, privacy assumptions, and evaluation protocols; do not compare headline metrics directly across systems.
      </p>

      {/* Lightbox Modal for Enlarging Charts During Presentation */}
      {lightboxImage && (
        <div
          className="fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-sm flex items-center justify-center p-4"
          role="dialog"
          aria-modal="true"
          aria-label={lightboxImage.title}
          onClick={() => setLightboxImage(null)}
        >
          <div
            className="bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-700 rounded-2xl max-w-5xl w-full p-4 sm:p-6 space-y-4 shadow-2xl"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between gap-4">
              <h3 className="text-base sm:text-lg font-bold text-slate-900 dark:text-white">
                {lightboxImage.title}
              </h3>
              <button
                type="button"
                onClick={() => setLightboxImage(null)}
                aria-label="Close enlarged chart"
                className="p-2 min-h-[40px] min-w-[40px] rounded-xl bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-700 flex items-center justify-center cursor-pointer"
              >
                <X className="w-5 h-5" />
              </button>
            </div>
            <div className="bg-white rounded-xl p-2 sm:p-4 border border-slate-200">
              <img
                src={lightboxImage.imageSrc}
                alt={lightboxImage.imageAlt}
                className="w-full max-h-[75vh] object-contain mx-auto"
              />
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
