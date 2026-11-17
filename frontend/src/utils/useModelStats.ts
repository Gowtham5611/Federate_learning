import { useMemo } from 'react';
import { BenchmarkModelRecord, SupportedDataset } from '../types/api';

export interface ModelStats {
  totalResearchConfigs: number;
  datasetConfigsCount: number;
  distinctArchitecturesCount: number;
  distinctArchitecturesList: string[];
  algorithmsList: string[];
  formatCountLabel: (filteredCount?: number) => string;
}

export function useModelStats(
  models: BenchmarkModelRecord[] = [],
  selectedDataset: SupportedDataset = 'HI-Small'
): ModelStats {
  return useMemo(() => {
    const datasetModels = models.filter((m) => m.dataset === selectedDataset);
    
    const archSet = new Set<string>();
    const algoSet = new Set<string>();

    datasetModels.forEach((m) => {
      if (m.architecture) archSet.add(m.architecture.toUpperCase());
      if (m.algorithm) algoSet.add(m.algorithm);
    });

    const distinctArchitecturesList = Array.from(archSet).sort();
    const algorithmsList = Array.from(algoSet).sort();

    const formatCountLabel = (filteredCount?: number) => {
      if (filteredCount !== undefined && filteredCount !== datasetModels.length) {
        return `Showing ${filteredCount} of ${datasetModels.length} ${selectedDataset} configs (${models.length} registered in registry)`;
      }
      return `${models.length} registered models · ${datasetModels.length} ${selectedDataset} configs shown`;
    };

    return {
      totalResearchConfigs: models.length,
      datasetConfigsCount: datasetModels.length,
      distinctArchitecturesCount: distinctArchitecturesList.length,
      distinctArchitecturesList,
      algorithmsList,
      formatCountLabel,
    };
  }, [models, selectedDataset]);
}
