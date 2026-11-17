/**
 * Chart and Data Export Utilities
 */

export function exportSvgToPng(svgElement: SVGSVGElement | null, filename: string = 'chart.png'): void {
  if (!svgElement) return;

  try {
    const serializer = new XMLSerializer();
    const svgString = serializer.serializeToString(svgElement);
    const svgBlob = new Blob([svgString], { type: 'image/svg+xml;charset=utf-8' });
    const URL = window.URL || window.webkitURL || window;
    const blobURL = URL.createObjectURL(svgBlob);

    const image = new Image();
    image.onload = () => {
      const canvas = document.createElement('canvas');
      const scale = 2; // high-DPI
      const bbox = svgElement.viewBox.baseVal;
      canvas.width = (bbox.width || svgElement.clientWidth || 800) * scale;
      canvas.height = (bbox.height || svgElement.clientHeight || 400) * scale;

      const context = canvas.getContext('2d');
      if (!context) return;

      context.scale(scale, scale);
      // Background fill based on dark/light mode
      const isDark = document.documentElement.classList.contains('dark');
      context.fillStyle = isDark ? '#0f172a' : '#ffffff';
      context.fillRect(0, 0, canvas.width, canvas.height);

      context.drawImage(image, 0, 0);

      const pngURL = canvas.toDataURL('image/png');
      const downloadLink = document.createElement('a');
      downloadLink.href = pngURL;
      downloadLink.download = filename.endsWith('.png') ? filename : `${filename}.png`;
      document.body.appendChild(downloadLink);
      downloadLink.click();
      document.body.removeChild(downloadLink);
      URL.revokeObjectURL(blobURL);
    };
    image.src = blobURL;
  } catch (err) {
    console.error('Failed to export chart as PNG:', err);
  }
}

export function exportDataToCsv(headers: string[], rows: (string | number)[][], filename: string = 'export.csv'): void {
  try {
    const csvContent = [
      headers.join(','),
      ...rows.map((row) =>
        row
          .map((cell) => {
            const str = String(cell ?? '');
            if (str.includes(',') || str.includes('"') || str.includes('\n')) {
              return `"${str.replace(/"/g, '""')}"`;
            }
            return str;
          })
          .join(',')
      ),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename.endsWith('.csv') ? filename : `${filename}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error('Failed to export data as CSV:', err);
  }
}
