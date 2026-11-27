import React, { useState, useMemo } from 'react';
import {
  BookOpen,
  ExternalLink,
  Filter,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  ShieldCheck,
  Search,
  Info,
} from 'lucide-react';

type SourceCategory =
  | 'All'
  | 'Research Paper'
  | 'Survey'
  | 'Open-Source Example'
  | 'Commercial Platform'
  | 'This Project';

interface ComparativeEntry {
  id: string;
  name: string;
  shortLabel: string;
  category: Exclude<SourceCategory, 'All'>;
  publicationStatus: string;
  authorsOrOrg: string;
  year: number;
  architecture: string;
  federatedStrategy: string;
  privacyMechanism: string;
  dataset: string;
  evaluationProtocol: string;
  explainability: string;
  calibrationDrift: string;
  limitations: string;
  methodSummary: string;
  caveats: string;
  contribution: string;
  differsFromFedSecureAI: string;
  sourceUrl?: string;
  sourceLabel: string;
  isFedSecureAI?: boolean;
}

const COMPARATIVE_ENTRIES: ComparativeEntry[] = [
  {
    id: 'fedsecureai',
    name: 'FedSecureAI (This Project)',
    shortLabel: 'HI-Small Champion (LSTM FedAvg IID)',
    category: 'This Project',
    publicationStatus: 'Verified Project Artifacts (v1.0)',
    authorsOrOrg: 'FedSecureAI Benchmark Suite',
    year: 2026,
    architecture: 'LSTM / BiLSTM / GRU (PyTorch sequence models, 10x57)',
    federatedStrategy: 'Simulated FedAvg & FedProx (IID and Non-IID bank splits)',
    privacyMechanism: 'Weight-only federated averaging in offline simulation; no DP or Secure Aggregation',
    dataset: 'IBM AML (HI-Small, LI-Medium)',
    evaluationProtocol: 'Chronological split; frozen val threshold (HI-Small θ=0.3378: PR-AUC 0.4760, ROC-AUC 0.9555, F1 0.5359)',
    explainability: 'Integrated Gradients (sequence models); tabular fallback path',
    calibrationDrift: 'Brier score (0.0012), ECE, per-upload prior-shift diagnostics',
    limitations: 'Non-IID degraded vs IID (PR-AUC 0.4505 vs 0.4760, hybrid α*=1.0); simulated single-host FL, not live cross-bank orchestration',
    methodSummary:
      'Trains recurrent sequence models over 10-step account transaction histories (57 engineered features) under simulated IID and Non-IID federated splits, freezing validation-optimal thresholds for live FastAPI CSV inference.',
    caveats:
      'Metrics come from offline IBM AML chronological splits (HI-Small test set: 716 TP, 159 FP, 1,081 FN, 1,013,608 TN). Synthetic/uploaded CSVs without multi-step account history receive conservative priors.',
    contribution:
      'End-to-end reproducible artifact pipeline with threshold-frozen inference, chunked memory-safe serving, Integrated Gradients attributions, and downloadable PDF/CSV triage reports.',
    differsFromFedSecureAI: 'Reference baseline for this repository.',
    sourceLabel: 'FedSecureAI Evaluated Benchmark Metrics',
    isFedSecureAI: true,
  },
  {
    id: 'pricing-cross-bank',
    name: 'Pricing Cross-Bank Context in Federated Anti-Money Laundering',
    shortLabel: 'Losavio et al. (2026)',
    category: 'Research Paper',
    publicationStatus: 'arXiv Preprint (Not Peer-Reviewed)',
    authorsOrOrg: 'Lidia Losavio, Dario Fenoglio, Martin Gjoreski, Marc Langheinrich',
    year: 2026,
    architecture: 'Multi-layer Graph Neural Network (GNN)',
    federatedStrategy: 'Cross-silo Federated Learning across separate bank graphs',
    privacyMechanism: 'Approximates remote account states from shared endpoint transactions without raw account exchange',
    dataset: 'Cross-bank transaction graphs (per paper setup)',
    evaluationProtocol: 'Graph-based AML evaluation comparing local vs cross-bank context',
    explainability: 'Not primary focus',
    calibrationDrift: 'Not reported as primary focus',
    limitations: 'Preprint status; focuses on cross-bank graph context rather than deployed web triage or threshold calibration',
    methodSummary:
      'Exploits transactions already recorded by both sending and receiving banks to approximate remote account states locally, enabling multi-layer GNN computation without pooling the full transaction graph.',
    caveats:
      'arXiv preprint submitted Oct 2026; uses graph-level assumptions and protocols not directly comparable to FedSecureAI sequence splits.',
    contribution:
      'Quantifies the predictive value of cross-bank neighborhood context in federated graph AML without centralizing graphs.',
    differsFromFedSecureAI:
      'Uses multi-layer GNNs over cross-bank graph edges, whereas FedSecureAI uses 10-step per-account LSTM/BiLSTM sequences.',
    sourceUrl: 'https://arxiv.org/abs/2610.35996',
    sourceLabel: 'arXiv:2610.35996',
  },
  {
    id: 'dpxfin',
    name: 'DPxFin: Adaptive Differential Privacy for AML Detection via Reputation-Weighted FL',
    shortLabel: 'Kanagavelu et al. (2026)',
    category: 'Research Paper',
    publicationStatus: 'arXiv Preprint (Not Peer-Reviewed)',
    authorsOrOrg: 'Renuga Kanagavelu, Manjil Nepal, Ning Peiyan, et al.',
    year: 2026,
    architecture: 'Tabular financial / AML classifiers',
    federatedStrategy: 'Reputation-weighted Federated Learning',
    privacyMechanism: 'Adaptive Differential Privacy (client reputation-guided noise)',
    dataset: 'Tabular financial / AML datasets',
    evaluationProtocol: 'Federated utility vs privacy-budget evaluation',
    explainability: 'Not primary focus',
    calibrationDrift: 'Client reputation weighting addresses heterogeneous client quality',
    limitations: 'Preprint status; DP noise introduces privacy-utility tradeoff not present in non-DP baselines',
    methodSummary:
      'Integrates reputation-guided adaptive differential privacy into federated AML training to mitigate privacy leakage in tabular financial data while down-weighting noisy or low-reputation clients.',
    caveats:
      'arXiv preprint submitted Mar 2026; different privacy budget settings and tabular splits prevent direct metric ranking.',
    contribution:
      'Demonstrates adaptive client-level DP coupled with reputation-weighted aggregation for tabular AML.',
    differsFromFedSecureAI:
      'Implements formal Adaptive Differential Privacy and reputation weighting; FedSecureAI currently does not implement DP noise.',
    sourceUrl: 'https://arxiv.org/abs/2603.19314',
    sourceLabel: 'arXiv:2603.19314',
  },
  {
    id: 'fedgraph-vasp',
    name: 'FedGraph-VASP: Federated Graph Learning with Post-Quantum Security',
    shortLabel: 'Commey et al. (2026)',
    category: 'Research Paper',
    publicationStatus: 'arXiv Preprint (Not Peer-Reviewed)',
    authorsOrOrg: 'Daniel Commey, Matilda Nkoom, Yousef Alsenani, Sena G. Hounsinou, Garth V. Crosby',
    year: 2026,
    architecture: 'Federated Graph Neural Network (VASP cross-institutional)',
    federatedStrategy: 'Federated graph learning with Boundary Embedding Exchange',
    privacyMechanism: 'Compressed, non-invertible boundary embeddings + post-quantum cryptographic security',
    dataset: 'Cross-institutional VASP / crypto AML flows',
    evaluationProtocol: 'Cross-chain / VASP laundering detection experiments',
    explainability: 'Not primary focus',
    calibrationDrift: 'Not primary focus',
    limitations: 'Preprint status; tailored to Virtual Asset Service Providers (VASPs) and cryptographic overhead',
    methodSummary:
      'Enables Virtual Asset Service Providers to collaboratively detect cross-institutional laundering by exchanging compressed, non-invertible boundary embeddings secured with post-quantum cryptography.',
    caveats:
      'arXiv preprint submitted Jan 2026; targets VASP/blockchain transaction graphs rather than bank fiat CSV streams.',
    contribution:
      'Combines federated graph boundary exchange with post-quantum security for cross-VASP AML.',
    differsFromFedSecureAI:
      'Addresses VASP graph topologies and post-quantum transport security; FedSecureAI focuses on fiat sequence scoring and interactive triage.',
    sourceUrl: 'https://arxiv.org/abs/2601.17935',
    sourceLabel: 'arXiv:2601.17935',
  },
  {
    id: 'fl-financial-survey',
    name: 'The Role of Federated Learning in Improving Financial Security: A Survey',
    shortLabel: 'Kennedy et al. (2025)',
    category: 'Survey',
    publicationStatus: 'arXiv Preprint Survey (Not Peer-Reviewed)',
    authorsOrOrg: 'Cade Houston Kennedy, Amr Hilal, Morteza Momeni',
    year: 2025,
    architecture: 'Taxonomy across GNN, RNN/LSTM, tree-based, and hybrid models',
    federatedStrategy: 'Cross-silo & cross-device FL taxonomy (FedAvg, FedProx, vertical/horizontal)',
    privacyMechanism: 'Surveys DP, Secure Aggregation, Homomorphic Encryption',
    dataset: 'Literature synthesis (PaySim, Elliptic, IBM AML, IEEE-CIS)',
    evaluationProtocol: 'Comparative literature survey (no single unified benchmark)',
    explainability: 'Reviews XAI gaps in financial FL',
    calibrationDrift: 'Highlights Non-IID drift and regulatory auditability challenges',
    limitations: 'Survey paper; summarizes prior work without introducing a new runnable system',
    methodSummary:
      'Reviews how cross-silo Federated Learning supports fraud and AML detection across financial endpoints (banks, ATMs, POS) while cataloguing privacy, heterogeneity, and deployment challenges.',
    caveats:
      'Survey synthesis (Oct 2025); reported metrics across surveyed papers are not directly comparable.',
    contribution:
      'Maps privacy mechanisms, Non-IID failure modes, and regulatory requirements across financial security FL.',
    differsFromFedSecureAI:
      'Provides broad taxonomy; FedSecureAI is a concrete, runnable sequence-model evaluation & reporting web application.',
    sourceUrl: 'https://arxiv.org/abs/2510.14991',
    sourceLabel: 'arXiv:2510.14991',
  },
  {
    id: 'ssrn-fate-aml',
    name: 'Federated Machine Learning Approaches for Anti-Money Laundering Detection',
    shortLabel: 'Guembe et al. (SSRN 4669561)',
    category: 'Research Paper',
    publicationStatus: 'SSRN Electronic Journal / ICISET Conference Paper',
    authorsOrOrg: 'Blessing Guembe, Ambrose Azeta, Victor Osamor, R. Ekpo',
    year: 2023,
    architecture: 'XGBoost / SecureBoost tree ensemble',
    federatedStrategy: 'FATE framework (central coordinator + institutional nodes, FedAvg comparison)',
    privacyMechanism: 'Homomorphic Encryption for model weights',
    dataset: 'PaySim synthetic mobile-money AML dataset',
    evaluationProtocol: 'Simulated federated vs centralized split on PaySim',
    explainability: 'XGBoost filter-based feature importance',
    calibrationDrift: 'Not reported',
    limitations: 'Evaluated on PaySim (synthetic mobile money) where accuracy (~99.9%) reflects class imbalance rather than PR-AUC',
    methodSummary:
      'Applies FATE SecureBoost with homomorphic encryption and XGBoost feature filtering on PaySim to match centralized detection without sharing raw customer transaction records.',
    caveats:
      'Uses PaySim dataset and accuracy-focused reporting; not directly comparable to IBM AML rare-class PR-AUC evaluation.',
    contribution:
      'Demonstrates tree-based federated AML with homomorphic encryption in FATE.',
    differsFromFedSecureAI:
      'Uses FATE + SecureBoost + HE on PaySim; FedSecureAI uses PyTorch LSTM/BiLSTM sequences on IBM AML with PR-AUC & alert-budget evaluation.',
    sourceUrl: 'https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4669561',
    sourceLabel: 'SSRN 4669561',
  },
  {
    id: 'flower-fraud',
    name: 'Flower Federated Fraud Detection Example (@mainthread)',
    shortLabel: 'Flower Hub Community Example',
    category: 'Open-Source Example',
    publicationStatus: 'Open-Source Community Example',
    authorsOrOrg: 'Flower Hub (@mainthread/federated-fraud-detection)',
    year: 2024,
    architecture: 'XGBoost gradient-boosted trees',
    federatedStrategy: 'Flower federated orchestration (multi-client simulation)',
    privacyMechanism: 'Decentralized local training; no built-in DP/HE in base example',
    dataset: 'Tabular financial fraud dataset',
    evaluationProtocol: 'Local Flower simulation example',
    explainability: 'Not bundled as interactive UI',
    calibrationDrift: 'Not bundled',
    limitations: 'Reference/starter example rather than a production AML triage system or audited benchmark',
    methodSummary:
      'Demonstrates federated financial fraud detection using XGBoost orchestrated via the open-source Flower framework.',
    caveats:
      'Community example app intended for learning and prototyping, not a complete production compliance platform.',
    contribution:
      'Clean, reproducible open-source template for running federated XGBoost in Flower.',
    differsFromFedSecureAI:
      'Focuses on Flower runtime + XGBoost trees; FedSecureAI provides an end-to-end FastAPI/React sequence evaluation, thresholding, and PDF reporting app.',
    sourceUrl: 'https://flower.ai/apps/mainthread/federated-fraud-detection',
    sourceLabel: 'Flower Hub',
  },
  {
    id: 'ibm-fl',
    name: 'IBM Federated Learning: An Enterprise Framework',
    shortLabel: 'Ludwig et al. (2020)',
    category: 'Research Paper',
    publicationStatus: 'arXiv White Paper (v0.1)',
    authorsOrOrg: 'Heiko Ludwig, Nathalie Baracaldo, Gegi Thomas, et al. (IBM Research)',
    year: 2020,
    architecture: 'Model-agnostic (NN, Keras, PyTorch, scikit-learn, RL)',
    federatedStrategy: 'Enterprise coordinator + party protocol with pluggable fusion handlers',
    privacyMechanism: 'Supports local-only data retention and extensible privacy plugins',
    dataset: 'Framework architecture (domain-agnostic)',
    evaluationProtocol: 'White paper describing enterprise FL architecture & fusion protocols',
    explainability: 'Not built-in as AML triage UI',
    calibrationDrift: 'Addresses data heterogeneity at framework level',
    limitations: 'General-purpose FL middleware white paper, not an AML-specific model or benchmark',
    methodSummary:
      'Defines an enterprise architecture for coordinating distributed parties, handling communication, heterogeneity, and pluggable model fusion across diverse ML libraries.',
    caveats:
      'White paper from 2020 describing framework design rather than AML-specific empirical metrics.',
    contribution:
      'Foundational enterprise blueprint for cross-organization FL orchestration.',
    differsFromFedSecureAI:
      'General-purpose enterprise FL framework; FedSecureAI is a domain-specific AML sequence evaluation & reporting application.',
    sourceUrl: 'https://arxiv.org/abs/2007.10987',
    sourceLabel: 'arXiv:2007.10987',
  },
  {
    id: 'consilient-aml',
    name: 'Consilient Core AML/CFT Model',
    shortLabel: 'Consilient (Commercial)',
    category: 'Commercial Platform',
    publicationStatus: 'Commercial Product Documentation',
    authorsOrOrg: 'Consilient',
    year: 2025,
    architecture: 'Proprietary behavioral AML/CFT risk models',
    federatedStrategy: 'Cross-institutional Federated Learning (Dozer / federated deployment)',
    privacyMechanism: 'Models travel to data behind institutional firewalls; no raw PII pooling',
    dataset: 'Proprietary live banking transaction networks',
    evaluationProtocol: 'Commercial bank validation & regulatory deployments (metrics not public)',
    explainability: 'Commercial compliance & investigator workflow',
    calibrationDrift: 'Continuous multi-bank model updates',
    limitations: 'Closed-source proprietary platform; architecture details and public benchmark metrics are not openly inspectable',
    methodSummary:
      'Commercial AML/CFT platform that trains behavioral risk models across multiple financial institutions using federated learning so sensitive customer data never leaves each bank.',
    caveats:
      'Product marketing/documentation page; public benchmark scores and source code are not available for independent verification.',
    contribution:
      'Demonstrates real-world regulatory and commercial viability of cross-institution federated AML.',
    differsFromFedSecureAI:
      'Production multi-bank commercial platform; FedSecureAI is an open academic/prototype system evaluated on public IBM AML datasets.',
    sourceUrl: 'https://consilient.com/our-products/core-aml-cft-model',
    sourceLabel: 'Consilient Product Page',
  },
];

const CATEGORY_BADGES: Record<Exclude<SourceCategory, 'All'>, string> = {
  'This Project': 'bg-emerald-950/80 text-emerald-300 border-emerald-700/60',
  'Research Paper': 'bg-indigo-950/80 text-indigo-300 border-indigo-700/60',
  Survey: 'bg-sky-950/80 text-sky-300 border-sky-700/60',
  'Open-Source Example': 'bg-amber-950/80 text-amber-300 border-amber-700/60',
  'Commercial Platform': 'bg-purple-950/80 text-purple-300 border-purple-700/60',
};

export const ComparativeAnalysisPage: React.FC = () => {
  const [categoryFilter, setCategoryFilter] = useState<SourceCategory>('All');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [sortBy, setSortBy] = useState<'year' | 'name' | 'category'>('year');
  const [sortAsc, setSortAsc] = useState<boolean>(false);
  const [expandedId, setExpandedId] = useState<string | null>('fedsecureai');

  const filteredEntries = useMemo(() => {
    const q = searchQuery.trim().toLowerCase();
    const list = COMPARATIVE_ENTRIES.filter((entry) => {
      if (categoryFilter !== 'All' && entry.category !== categoryFilter) return false;
      if (!q) return true;
      return (
        entry.name.toLowerCase().includes(q) ||
        entry.architecture.toLowerCase().includes(q) ||
        entry.federatedStrategy.toLowerCase().includes(q) ||
        entry.privacyMechanism.toLowerCase().includes(q) ||
        entry.dataset.toLowerCase().includes(q)
      );
    });

    return [...list].sort((a, b) => {
      if (a.isFedSecureAI && !b.isFedSecureAI) return -1;
      if (!a.isFedSecureAI && b.isFedSecureAI) return 1;
      let cmp = 0;
      if (sortBy === 'year') cmp = a.year - b.year;
      else if (sortBy === 'name') cmp = a.name.localeCompare(b.name);
      else cmp = a.category.localeCompare(b.category);
      return sortAsc ? cmp : -cmp;
    });
  }, [categoryFilter, searchQuery, sortBy, sortAsc]);

  const handleSort = (field: 'year' | 'name' | 'category') => {
    if (sortBy === field) {
      setSortAsc((prev) => !prev);
    } else {
      setSortBy(field);
      setSortAsc(field !== 'year');
    }
  };

  return (
    <div className="space-y-8 animate-fade-in-up">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-slate-800 pb-5">
        <div className="flex items-center space-x-2.5">
          <div className="p-2 rounded-xl bg-indigo-600/20 border border-indigo-500/30 text-indigo-400">
            <BookOpen className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-2xl sm:text-3xl font-black text-slate-900 dark:text-white tracking-tight">
              Comparative Analysis: Federated AML Systems & Literature
            </h1>
            <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 mt-1">
              Source-verified comparison of FedSecureAI against arXiv preprints, SSRN/conference studies, open-source FL examples, and commercial platforms.
            </p>
          </div>
        </div>
      </div>

      {/* 1-Line Methodological Note */}
      <div
        role="note"
        title="Each entry uses different datasets, class balances, split protocols, and thresholds; avoid direct headline-metric ranking."
        className="px-4 py-3 rounded-xl bg-amber-950/40 border border-amber-500/40 flex items-center justify-between gap-3 text-xs sm:text-sm text-amber-200"
      >
        <div className="flex items-center space-x-2.5 min-w-0">
          <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0" />
          <span className="truncate sm:whitespace-normal">
            Methodological Note: Metrics across systems reflect different datasets, splits, and thresholds.
          </span>
        </div>
        <span
          title="Each entry uses different datasets (IBM AML, PaySim, VASP crypto graphs, or private bank networks), class balances, split protocols, decision thresholds, and privacy budgets."
          className="text-amber-300 shrink-0 cursor-help"
        >
          <Info className="w-4 h-4" />
        </span>
      </div>

      {/* Pinned This Project Highlight Card */}
      {COMPARATIVE_ENTRIES.filter((e) => e.category === 'This Project').map((proj) => (
        <div
          key={`pinned-${proj.id}`}
          className="rounded-2xl border-2 border-indigo-500/60 bg-indigo-950/20 dark:bg-indigo-950/30 p-5 shadow-md space-y-2"
        >
          <div className="flex flex-wrap items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-0.5 rounded-full text-[11px] font-mono font-semibold bg-indigo-600 text-white">
                This Project
              </span>
              <h2 className="text-base font-bold text-slate-900 dark:text-white">{proj.name}</h2>
            </div>
            <span className="text-xs font-mono text-indigo-400">{proj.dataset}</span>
          </div>
          <p className="text-xs sm:text-sm text-slate-700 dark:text-slate-300">
            {proj.architecture} · {proj.federatedStrategy} · {proj.explainability}
          </p>
        </div>
      ))}

      {/* Controls: Category Filter + Search */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-4">
        <div className="flex flex-wrap items-center gap-1.5">
          <span className="text-xs font-mono text-slate-500 dark:text-slate-400 flex items-center space-x-1 mr-1">
            <Filter className="w-3.5 h-3.5" />
            <span>Type:</span>
          </span>
          {(
            ['All', 'This Project', 'Research Paper', 'Survey', 'Open-Source Example', 'Commercial Platform'] as SourceCategory[]
          ).map((cat) => (
            <button
              key={cat}
              type="button"
              onClick={() => setCategoryFilter(cat)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition cursor-pointer border ${
                categoryFilter === cat
                  ? 'bg-indigo-600 text-white border-indigo-500 shadow-sm'
                  : 'bg-slate-100 dark:bg-slate-950 text-slate-700 dark:text-slate-300 border-slate-200 dark:border-slate-800 hover:border-indigo-500/50'
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="relative w-full lg:w-72">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter by architecture, dataset, privacy..."
            aria-label="Filter comparative systems"
            className="w-full pl-9 pr-3 py-2 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-900 dark:text-white focus:outline-none focus:border-indigo-500"
          />
        </div>
      </div>

      {/* Mobile Card List (< 768px) */}
      <div className="md:hidden space-y-3">
        {filteredEntries.length === 0 ? (
          <div className="py-8 text-center text-slate-500 text-xs rounded-2xl border border-slate-800">
            No comparative entries match your current filter.
          </div>
        ) : (
          filteredEntries.map((entry) => {
            const isExpanded = expandedId === entry.id;
            return (
              <div
                key={`comp-mob-${entry.id}`}
                className={`rounded-2xl border p-4 space-y-2 ${
                  entry.category === 'This Project'
                    ? 'border-indigo-500/60 bg-indigo-950/20'
                    : 'border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900'
                }`}
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <h3 className="text-sm font-bold text-slate-900 dark:text-white">{entry.name}</h3>
                    <span className="text-[11px] font-mono text-slate-500 dark:text-slate-400">
                      {entry.category} · {entry.year} · {entry.dataset}
                    </span>
                  </div>
                  <button
                    type="button"
                    onClick={() => setExpandedId(isExpanded ? null : entry.id)}
                    className="p-2 min-h-[36px] rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-600 dark:text-slate-300 shrink-0 cursor-pointer"
                    aria-label="Toggle entry details"
                  >
                    {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                  </button>
                </div>
                <p className="text-xs text-slate-700 dark:text-slate-300">
                  {entry.architecture} · {entry.federatedStrategy}
                </p>
                {isExpanded && (
                  <div className="pt-2 border-t border-slate-200 dark:border-slate-800 space-y-1.5 text-xs text-slate-600 dark:text-slate-300">
                    <p><strong className="text-slate-900 dark:text-white">Privacy:</strong> {entry.privacyMechanism}</p>
                    <p><strong className="text-slate-900 dark:text-white">Evaluation:</strong> {entry.evaluationProtocol}</p>
                    <p><strong className="text-slate-900 dark:text-white">Explainability:</strong> {entry.explainability}</p>
                    <p><strong className="text-slate-900 dark:text-white">Summary:</strong> {entry.methodSummary}</p>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      {/* Comparison Matrix Table (Desktop/Tablet) */}
      <div className="hidden md:block bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl shadow-xl overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse text-xs">
            <thead>
              <tr className="bg-slate-100 dark:bg-slate-950 border-b border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-400 uppercase text-[10px] font-mono">
                <th className="py-3.5 px-4 min-w-[230px] sticky left-0 z-10 bg-slate-100 dark:bg-slate-950">
                  <button
                    type="button"
                    onClick={() => handleSort('name')}
                    className="flex items-center space-x-1 font-bold hover:text-indigo-400 cursor-pointer"
                  >
                    <span>System / Paper</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </button>
                </th>
                <th className="py-3.5 px-3 min-w-[130px]">
                  <button
                    type="button"
                    onClick={() => handleSort('year')}
                    className="flex items-center space-x-1 font-bold hover:text-indigo-400 cursor-pointer"
                  >
                    <span>Year & Status</span>
                    <ArrowUpDown className="w-3 h-3" />
                  </button>
                </th>
                <th className="py-3.5 px-3 min-w-[170px]">Architecture</th>
                <th className="py-3.5 px-3 min-w-[170px]">Federated Strategy</th>
                <th className="py-3.5 px-3 min-w-[170px]">Privacy Mechanism</th>
                <th className="py-3.5 px-3 min-w-[140px]">Dataset</th>
                <th className="py-3.5 px-3 min-w-[190px]">Evaluation Protocol</th>
                <th className="py-3.5 px-3 min-w-[150px]">Explainability</th>
                <th className="py-3.5 px-3 min-w-[150px]">Calibration / Drift</th>
                <th className="py-3.5 px-3 min-w-[190px]">Limitations</th>
                <th className="py-3.5 px-3 min-w-[90px] text-center">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-200 dark:divide-slate-800/70">
              {filteredEntries.length === 0 ? (
                <tr>
                  <td colSpan={11} className="py-8 text-center text-slate-500 text-xs">
                    No comparative entries match your current filter.
                  </td>
                </tr>
              ) : (
                filteredEntries.map((entry) => {
                  const isExpanded = expandedId === entry.id;
                  return (
                    <React.Fragment key={entry.id}>
                      <tr
                        className={`transition-colors ${
                          entry.isFedSecureAI
                            ? 'bg-emerald-950/20 hover:bg-emerald-950/30'
                            : 'hover:bg-slate-50 dark:hover:bg-slate-800/40'
                        }`}
                      >
                        <td className="py-3.5 px-4 align-top">
                          <div className="space-y-1.5">
                            <span
                              className={`inline-block px-2 py-0.5 rounded-full text-[10px] font-mono border ${
                                CATEGORY_BADGES[entry.category]
                              }`}
                            >
                              {entry.category}
                            </span>
                            <div className="font-bold text-slate-900 dark:text-white leading-snug">
                              {entry.name}
                            </div>
                            {entry.sourceUrl ? (
                              <a
                                href={entry.sourceUrl}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="inline-flex items-center space-x-1 text-[11px] font-mono text-indigo-600 dark:text-indigo-400 hover:underline"
                              >
                                <span>{entry.sourceLabel}</span>
                                <ExternalLink className="w-3 h-3" />
                              </a>
                            ) : (
                              <span className="inline-flex items-center space-x-1 text-[11px] font-mono text-emerald-600 dark:text-emerald-400">
                                <ShieldCheck className="w-3 h-3" />
                                <span>{entry.sourceLabel}</span>
                              </span>
                            )}
                          </div>
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          <div className="font-mono font-bold">{entry.year}</div>
                          <div className="text-[11px] text-slate-500 dark:text-slate-400 mt-0.5">
                            {entry.publicationStatus}
                          </div>
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.architecture}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.federatedStrategy}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.privacyMechanism}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.dataset}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.evaluationProtocol}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.explainability}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-700 dark:text-slate-300">
                          {entry.calibrationDrift}
                        </td>
                        <td className="py-3.5 px-3 align-top text-slate-600 dark:text-slate-400">
                          {entry.limitations}
                        </td>
                        <td className="py-3.5 px-3 align-top text-center">
                          <button
                            type="button"
                            onClick={() => setExpandedId(isExpanded ? null : entry.id)}
                            aria-expanded={isExpanded}
                            className="inline-flex items-center space-x-1 px-2.5 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 hover:bg-indigo-600 hover:text-white text-slate-700 dark:text-slate-200 text-[11px] font-semibold transition cursor-pointer"
                          >
                            <span>{isExpanded ? 'Hide' : 'Expand'}</span>
                            {isExpanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                          </button>
                        </td>
                      </tr>

                      {isExpanded && (
                        <tr className="bg-slate-50/80 dark:bg-slate-950/70">
                          <td colSpan={11} className="p-5">
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 text-xs">
                              <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1.5">
                                <div className="font-bold text-indigo-600 dark:text-indigo-400 uppercase text-[10px] font-mono">
                                  Method Summary
                                </div>
                                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                                  {entry.methodSummary}
                                </p>
                              </div>
                              <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1.5">
                                <div className="font-bold text-amber-600 dark:text-amber-400 uppercase text-[10px] font-mono">
                                  Dataset & Evaluation Caveats
                                </div>
                                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                                  {entry.caveats}
                                </p>
                              </div>
                              <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1.5">
                                <div className="font-bold text-emerald-600 dark:text-emerald-400 uppercase text-[10px] font-mono">
                                  Key Contribution
                                </div>
                                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                                  {entry.contribution}
                                </p>
                              </div>
                              <div className="p-3.5 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 space-y-1.5">
                                <div className="font-bold text-violet-600 dark:text-violet-400 uppercase text-[10px] font-mono">
                                  How It Differs From FedSecureAI
                                </div>
                                <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
                                  {entry.differsFromFedSecureAI}
                                </p>
                                <div className="pt-1 text-[11px] font-mono text-slate-500 dark:text-slate-400">
                                  Authors / Org: {entry.authorsOrOrg}
                                </div>
                              </div>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Footer Note */}
      <div className="flex items-start space-x-2 text-xs text-slate-500 dark:text-slate-400">
        <Info className="w-4 h-4 shrink-0 mt-0.5 text-indigo-400" />
        <span>
          FedSecureAI metrics reflect evaluated held-out test results on the IBM AML benchmark splits.
        </span>
      </div>
    </div>
  );
};
