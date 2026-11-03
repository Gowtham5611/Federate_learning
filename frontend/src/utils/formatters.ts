/**
 * Utility formatting helpers for AML risk metrics and feature names.
 */

export function formatMetric(
  value: number | null | undefined,
  digits: number = 4,
  fallback: string = 'N/A'
): string {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return value.toFixed(digits);
}

export function formatPercent(
  value: number | null | undefined,
  digits: number = 1,
  fallback: string = 'N/A'
): string {
  if (value == null || !Number.isFinite(value)) {
    return fallback;
  }
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatRatioPercent(
  num: number | null | undefined,
  den: number | null | undefined,
  digits: number = 1,
  fallback: string = '0.0%'
): string {
  if (num == null || den == null || !Number.isFinite(num) || !Number.isFinite(den) || den <= 0) {
    return fallback;
  }
  return `${((num / den) * 100).toFixed(digits)}%`;
}

export function formatProbability(prob: number | null | undefined): string {
  return formatPercent(prob, 2, '0.00%');
}

export function formatScore(prob: number | null | undefined): string {
  return formatMetric(prob, 6, '0.000000');
}

export function formatThreshold(thresh: number | null | undefined): string {
  return formatMetric(thresh, 4, '0.0000');
}

export function formatTimestamp(isoString: string): string {
  try {
    const d = new Date(isoString);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
      ' ' + d.toLocaleDateString([], { month: 'short', day: 'numeric' });
  } catch {
    return isoString;
  }
}

export function formatFeatureName(name: string): string {
  // Convert snake_case or __cat_X into human readable AML terminology
  if (name.includes('__cat_')) {
    const [base, cat] = name.split('__cat_');
    const cleanBase = base.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
    return `${cleanBase} (Type ${cat})`;
  }
  return name.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

export function getRiskLevel(prob: number, threshold: number): {
  level: 'CRITICAL_RISK' | 'ELEVATED' | 'LOW_RISK';
  label: string;
  badgeClass: string;
} {
  if (prob >= threshold) {
    return {
      level: 'CRITICAL_RISK',
      label: 'FLAGGED LAUNDERING ALERT',
      badgeClass: 'bg-rose-950/80 text-rose-300 border-rose-500/50',
    };
  } else if (prob >= threshold * 0.75) {
    return {
      level: 'ELEVATED',
      label: 'ELEVATED SURVEILLANCE',
      badgeClass: 'bg-amber-950/80 text-amber-300 border-amber-500/50',
    };
  }
  return {
    level: 'LOW_RISK',
    label: 'CLEARED LEGITIMATE',
    badgeClass: 'bg-emerald-950/80 text-emerald-300 border-emerald-500/50',
  };
}

