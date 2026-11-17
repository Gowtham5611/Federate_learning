import React from 'react';

interface ChartSkeletonProps {
  title?: string;
  height?: string;
  className?: string;
}

export const ChartSkeleton: React.FC<ChartSkeletonProps> = ({
  title = 'Loading Visualization...',
  height = 'h-80',
  className = '',
}) => {
  return (
    <div className={`bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-3xl p-6 shadow-xl space-y-4 animate-pulse ${className}`}>
      <div className="flex items-center justify-between border-b border-slate-200 dark:border-slate-800 pb-3">
        <div className="space-y-1.5">
          <div className="text-xs font-mono text-slate-500 dark:text-slate-400 font-bold">{title}</div>
          <div className="h-3 w-72 bg-slate-100 dark:bg-slate-800/60 rounded-md" />
        </div>
        <div className="flex space-x-1.5">
          <div className="h-7 w-16 bg-slate-200 dark:bg-slate-800 rounded-lg" />
          <div className="h-7 w-16 bg-slate-200 dark:bg-slate-800 rounded-lg" />
        </div>
      </div>

      <div className={`w-full ${height} flex items-end justify-between px-8 pb-4 pt-10 space-x-3`}>
        {[40, 65, 80, 50, 90, 70, 85, 60, 95, 75, 85, 60].map((h, i) => (
          <div key={i} className="flex-1 flex flex-col items-center space-y-2">
            <div
              className="w-full bg-slate-200 dark:bg-slate-800 rounded-t-md"
              style={{ height: `${h}%` }}
            />
            <div className="h-2.5 w-6 bg-slate-200 dark:bg-slate-800 rounded" />
          </div>
        ))}
      </div>

      <div className="flex items-center justify-center space-x-6 pt-3 border-t border-slate-200 dark:border-slate-800">
        <div className="h-3 w-24 bg-slate-200 dark:bg-slate-800 rounded" />
        <div className="h-3 w-24 bg-slate-200 dark:bg-slate-800 rounded" />
        <div className="h-3 w-24 bg-slate-200 dark:bg-slate-800 rounded" />
      </div>
    </div>
  );
};
