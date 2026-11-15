const fs = require('fs');
const path = require('path');
const { chromium } = require('../.venv/Lib/site-packages/playwright/driver/package/index.js');

const REPO_ROOT = path.resolve(__dirname, '..');
const DIST_DIR = path.join(REPO_ROOT, 'frontend', 'dist');
const SCREENSHOTS_DIR = path.join(REPO_ROOT, 'results', 'ui_screenshots');
if (!fs.existsSync(SCREENSHOTS_DIR)) {
  fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
}

// Load demo meta data from authoritative run pred_1794327770_2951fbed
const metaPath = path.join(REPO_ROOT, 'results', 'uploads', 'pred_1794327770_2951fbed_meta.json');
const demoMeta = JSON.parse(fs.readFileSync(metaPath, 'utf8'));

// Parse actual predictions from results/uploads/pred_1794327770_2951fbed.csv
const csvPath = path.join(REPO_ROOT, 'results', 'uploads', 'pred_1794327770_2951fbed.csv');
const csvLines = fs.readFileSync(csvPath, 'utf8').trim().split('\n');
const headers = csvLines[0].split(',').map(h => h.trim());

const allCsvRows = [];
for (let i = 1; i < csvLines.length; i++) {
  const parts = csvLines[i].split(',');
  const rowObj = {};
  headers.forEach((h, idx) => {
    rowObj[h] = parts[idx] ? parts[idx].trim() : '';
  });
  const prob = parseFloat(rowObj['predicted_probability'] || '0');
  const origRowId = i; // 1-based index from CSV
  allCsvRows.push({
    row_id: origRowId,
    transaction_id: `TX_${String(origRowId).padStart(5, '0')}`,
    sender_name: null,
    receiver_name: null,
    transaction_name: null,
    timestamp: rowObj['Timestamp'] || '',
    from_account: rowObj['Account'] || '',
    to_account: rowObj['Account.1'] || '',
    amount: parseFloat(rowObj['Amount Paid'] || '0'),
    currency: rowObj['Payment Currency'] || '',
    predicted_probability: prob,
    risk_score: prob * 100,
    decision: prob >= 0.337784 ? 'FLAGGED_LAUNDERING' : 'CLEARED_LEGITIMATE',
    predicted_label: prob >= 0.337784 ? 'SUSPICIOUS' : 'LEGITIMATE',
    ground_truth_label: parseInt(rowObj['Is Laundering'] || '0'),
    actual_label: parseInt(rowObj['Is Laundering'] || '0') === 1 ? 'SUSPICIOUS' : 'LEGITIMATE',
    sequence_length: 1,
    valid_history_count: 0,
    is_history_sufficient: false,
    usable_history_note: 'Cold-start: single-tx account without history; velocity features defaulted to 0.0.',
  });
}

// Sort by predicted_probability descending: top 15 are all flagged alerts, followed by top cleared suspects
allCsvRows.sort((a, b) => b.predicted_probability - a.predicted_probability);
const samplePredictions = allCsvRows.slice(0, 50);

const demoResult = {
  prediction_id: demoMeta.prediction_id || 'pred_1794327770_2951fbed',
  filename: demoMeta.filename || 'ibm_aml_schema_synthetic_20000.csv',
  file_hash: demoMeta.file_hash || '3c1e1b72263ed5040b53939882a2df49b505383b430967b93e5e4ecab0556011',
  total_rows: 20000,
  evaluated_sequences: 20000,
  flagged_laundering_count: 15,
  cleared_count: 19985,
  execution_time_ms: 1240,
  model_name: demoMeta.model_name || 'Fallback: no sequence history (LightGBM)',
  model_id: 'HI-Small',
  dataset: 'HI-Small',
  decision_threshold: 0.337784,
  has_labels: true,
  has_ground_truth: true,
  evaluation_metrics: demoMeta.eval_metrics || demoMeta.evaluation_metrics || {},
  operating_points: demoMeta.operating_points || (demoMeta.eval_metrics && demoMeta.eval_metrics.operating_points) || {},
  risk_distribution: demoMeta.risk_distribution || [
    { bin_range: '0.0 - 0.1', lower: 0.0, upper: 0.1, count: 19850, percentage: 99.25 },
    { bin_range: '0.1 - 0.2', lower: 0.1, upper: 0.2, count: 100, percentage: 0.5 },
    { bin_range: '0.2 - 0.3', lower: 0.2, upper: 0.3, count: 35, percentage: 0.175 },
    { bin_range: '0.3 - 0.4', lower: 0.3, upper: 0.4, count: 4, percentage: 0.02 },
    { bin_range: '0.4 - 0.5', lower: 0.4, upper: 0.5, count: 2, percentage: 0.01 },
    { bin_range: '0.5 - 0.6', lower: 0.5, upper: 0.6, count: 3, percentage: 0.015 },
    { bin_range: '0.6 - 0.7', lower: 0.6, upper: 0.7, count: 3, percentage: 0.015 },
    { bin_range: '0.7 - 0.8', lower: 0.7, upper: 0.8, count: 1, percentage: 0.005 },
    { bin_range: '0.8 - 0.9', lower: 0.8, upper: 0.9, count: 2, percentage: 0.01 },
    { bin_range: '0.9 - 1.0', lower: 0.9, upper: 1.0, count: 0, percentage: 0.0 },
  ],
  predictions: samplePredictions,
  validity_banner: demoMeta.validity_banner || {
    status: 'VALID',
    title: 'Universal Transaction Schema Validated',
    message: 'Successfully preprocessed 20,000 transactions across 14 banks and 4 currency formats.',
    dataset_identified: 'HI-Small',
    row_count: 20000,
    is_compatible: true,
  },
};

// Benchmark models list for Model Registry
const sampleBenchmarkModels = [
  {
    dataset: 'HI-Small',
    config_id: 'config_3',
    model_name: 'hi_small_lstm_fedavg_iid_seed42',
    architecture: 'LSTM',
    distribution: 'IID',
    algorithm: 'FedAvg',
    threshold: 0.337784,
    test_pr_auc: 0.475977,
    test_roc_auc: 0.955481,
    test_f1: 0.5359,
    test_precision: 0.8183,
    test_recall: 0.3984,
    test_specificity: 0.9988,
    test_accuracy: 0.9978,
    test_balanced_accuracy: 0.6986,
    test_mcc: 0.5694,
    test_cohen_kappa: 0.5348,
    alerts_per_10k: 8.62,
    brier_score: 0.0012,
    test_ece: 0.0009,
    log_loss: 0.0072,
    is_champion: true,
  },
  {
    dataset: 'HI-Small',
    config_id: 'config_1',
    model_name: 'hi_small_gru_fedavg_iid_seed42',
    architecture: 'GRU',
    distribution: 'IID',
    algorithm: 'FedAvg',
    threshold: 0.5,
    test_pr_auc: 0.3950,
    test_roc_auc: 0.9548,
    test_f1: 0.3401,
    test_precision: 0.2782,
    test_recall: 0.4374,
    test_specificity: 0.9972,
    test_accuracy: 0.9962,
    test_balanced_accuracy: 0.7173,
    test_mcc: 0.3468,
    test_cohen_kappa: 0.3385,
    alerts_per_10k: 27.82,
    brier_score: 0.0028,
    test_ece: 0.0018,
    log_loss: 0.0129,
  },
  {
    dataset: 'HI-Small',
    config_id: 'config_2',
    model_name: 'hi_small_gru_fedprox_mu0.0001_iid_seed42',
    architecture: 'GRU',
    distribution: 'IID',
    algorithm: 'FedProx',
    threshold: 0.5,
    test_pr_auc: 0.2496,
    test_roc_auc: 0.9357,
    test_f1: 0.2581,
    test_precision: 0.2211,
    test_recall: 0.3100,
    test_specificity: 0.9975,
    test_accuracy: 0.9963,
    test_balanced_accuracy: 0.6538,
    test_mcc: 0.2598,
    test_cohen_kappa: 0.2562,
    alerts_per_10k: 24.80,
    brier_score: 0.0030,
    test_ece: 0.0022,
    log_loss: 0.0153,
  },
  {
    dataset: 'HI-Small',
    config_id: 'config_4',
    model_name: 'hi_small_lstm_fedprox_mu0.0001_iid_seed42',
    architecture: 'LSTM',
    distribution: 'IID',
    algorithm: 'FedProx',
    threshold: 0.5,
    test_pr_auc: 0.2824,
    test_roc_auc: 0.9106,
    test_f1: 0.3079,
    test_precision: 0.8597,
    test_recall: 0.1875,
    test_specificity: 0.9996,
    test_accuracy: 0.9982,
    test_balanced_accuracy: 0.5936,
    test_mcc: 0.3998,
    test_cohen_kappa: 0.3065,
    alerts_per_10k: 3.86,
    brier_score: 0.0029,
    test_ece: 0.0015,
    log_loss: 0.0199,
  },
  {
    dataset: 'HI-Small',
    config_id: 'config_5',
    model_name: 'hi_small_bilstm_fedavg_iid_seed42',
    architecture: 'BILSTM',
    distribution: 'IID',
    algorithm: 'FedAvg',
    threshold: 0.5,
    test_pr_auc: 0.4485,
    test_roc_auc: 0.9498,
    test_f1: 0.4612,
    test_precision: 0.7245,
    test_recall: 0.3378,
    test_specificity: 0.9991,
    test_accuracy: 0.9980,
    test_balanced_accuracy: 0.6685,
    test_mcc: 0.4932,
    test_cohen_kappa: 0.4598,
    alerts_per_10k: 8.24,
    brier_score: 0.0014,
    test_ece: 0.0011,
    log_loss: 0.0088,
  },
  {
    dataset: 'HI-Small',
    config_id: 'centralized_lstm',
    model_name: 'centralized_lstm_baseline',
    architecture: 'LSTM',
    distribution: 'Centralized',
    algorithm: 'Centralized',
    threshold: 0.5,
    test_pr_auc: 0.4771,
    test_roc_auc: 0.9572,
    test_f1: 0.5412,
    test_precision: 0.8250,
    test_recall: 0.4025,
    test_specificity: 0.9989,
    test_accuracy: 0.9979,
    test_balanced_accuracy: 0.7007,
    test_mcc: 0.5755,
    test_cohen_kappa: 0.5401,
    alerts_per_10k: 8.66,
    brier_score: 0.0012,
    test_ece: 0.0008,
    log_loss: 0.0070,
  },
];

const VIEWPORTS = [
  { name: '1920_wide_desktop', width: 1920, height: 1080 },
  { name: '1440_standard_desktop', width: 1440, height: 900 },
  { name: '1280_small_desktop', width: 1280, height: 800 },
  { name: '1024_tablet_landscape', width: 1024, height: 768 },
  { name: '768_tablet_portrait', width: 768, height: 1024 },
  { name: '390_mobile_viewport', width: 390, height: 844 },
];

async function runCapture() {
  const browser = await chromium.launch({ headless: true });

  for (const vp of VIEWPORTS) {
    console.log(`\n======================================================`);
    console.log(`TESTING VIEWPORT: ${vp.name} (${vp.width}x${vp.height})`);
    console.log(`======================================================`);

    const context = await browser.newContext({
      viewport: { width: vp.width, height: vp.height },
    });

    // In-memory intercept: route fulfillment without external network sockets
    await context.route('**', (route) => {
      const reqUrl = route.request().url();
      const u = new URL(reqUrl);

      // API mocks
      if (u.pathname.includes('/models') || u.pathname.includes('/research/models')) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(sampleBenchmarkModels),
        });
      }
      if (u.pathname.includes('/health')) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({ status: 'healthy', readiness: 'ready', loaded_models: ['HI-Small', 'LI-Medium'] }),
        });
      }
      if (u.pathname.includes('/curves')) {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify([
            { round: 1, val_pr_auc: 0.31, val_roc_auc: 0.88, val_loss: 0.02 },
            { round: 2, val_pr_auc: 0.38, val_roc_auc: 0.92, val_loss: 0.015 },
            { round: 3, val_pr_auc: 0.42, val_roc_auc: 0.94, val_loss: 0.012 },
            { round: 4, val_pr_auc: 0.46, val_roc_auc: 0.95, val_loss: 0.009 },
            { round: 5, val_pr_auc: 0.476, val_roc_auc: 0.955, val_loss: 0.007 },
          ]),
        });
      }
      if (u.pathname.includes('/thresholds')) {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) });
      }
      if (u.pathname.includes('/communication')) {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify([]) });
      }
      if (u.pathname.includes('/client-partitions')) {
        return route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(null) });
      }

      // Static assets from frontend/dist/
      let localPath = path.join(DIST_DIR, u.pathname === '/' ? 'index.html' : u.pathname);
      if (!fs.existsSync(localPath)) {
        localPath = path.join(DIST_DIR, 'index.html');
      }

      const ext = path.extname(localPath).toLowerCase();
      const contentTypes = {
        '.html': 'text/html',
        '.js': 'text/javascript',
        '.css': 'text/css',
        '.json': 'application/json',
        '.png': 'image/png',
        '.svg': 'image/svg+xml',
      };
      const contentType = contentTypes[ext] || 'application/octet-stream';

      try {
        const fileBytes = fs.readFileSync(localPath);
        return route.fulfill({
          status: 200,
          contentType,
          body: fileBytes,
        });
      } catch (err) {
        return route.fulfill({
          status: 404,
          contentType: 'text/plain',
          body: 'Not found',
        });
      }
    });

    const page = await context.newPage();

    // Navigate to in-memory URL
    await page.goto('http://fedsecureai.local/', { waitUntil: 'domcontentloaded' });

    // Set localStorage demo result
    await page.evaluate((data) => {
      localStorage.setItem('fedsecureai_current_result', JSON.stringify(data));
    }, demoResult);

    await page.reload({ waitUntil: 'domcontentloaded' });
    await page.waitForSelector('header', { timeout: 8000 });
    await new Promise((r) => setTimeout(r, 600));

    // 1. Check Horizontal Overflow on Evaluation Page
    const scrollW1 = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientW1 = await page.evaluate(() => document.documentElement.clientWidth);
    const overflow1 = scrollW1 > clientW1;
    console.log(`[Evaluation Page] scrollWidth: ${scrollW1}, clientWidth: ${clientW1}, Overflow: ${overflow1}`);
    if (overflow1) {
      const off = await page.evaluate(() => {
        let maxR = 0, elMax = null;
        document.querySelectorAll('*').forEach(el => {
          const r = el.getBoundingClientRect().right;
          if (r > maxR) { maxR = r; elMax = { tag: el.tagName, class: el.className, text: (el.innerText||'').slice(0, 40), r }; }
        });
        return { maxR, elMax };
      });
      console.log('    --> Offender on Eval Page:', JSON.stringify(off));
    }

    const evalScreenshot = path.join(SCREENSHOTS_DIR, `${vp.name}_01_aml_evaluation.png`);
    await page.screenshot({ path: evalScreenshot, fullPage: true });
    console.log(`✓ Saved: ${path.basename(evalScreenshot)}`);

    // 2. Navigate to Results & Analytics Page
    const navBtn = page.locator("button:has-text('Results & Analytics'), button:has-text('Analytics')").first();
    await navBtn.click();
    await new Promise((r) => setTimeout(r, 600));

    // Check Horizontal Overflow on Analytics Page
    const scrollW2 = await page.evaluate(() => document.documentElement.scrollWidth);
    const clientW2 = await page.evaluate(() => document.documentElement.clientWidth);
    const overflow2 = scrollW2 > clientW2;
    console.log(`[Analytics Page] scrollWidth: ${scrollW2}, clientWidth: ${clientW2}, Overflow: ${overflow2}`);

    // Mode 1: Fixed Canonical Threshold
    const fixedModeBtn = page.locator("button:has-text('Fixed Canonical Threshold')").first();
    if (await fixedModeBtn.isVisible()) {
      await fixedModeBtn.click();
      await new Promise((r) => setTimeout(r, 400));
    }

    // Test Heatmap View Controls: Switch to Table mode and back to Grid
    const hmTableBtn = page.locator("button[title='Tabular Matrix View']").first();
    if (await hmTableBtn.isVisible()) {
      await hmTableBtn.click();
      await new Promise((r) => setTimeout(r, 300));
    }
    const hmGridBtn = page.locator("button[title='Interactive Heatmap Grid']").first();
    if (await hmGridBtn.isVisible()) {
      await hmGridBtn.click();
      await new Promise((r) => setTimeout(r, 300));
    }

    // Test Measure Switching on Heatmap
    const meanProbBtn = page.locator("button:has-text('Mean Risk Probability')").first();
    if (await meanProbBtn.isVisible()) {
      await meanProbBtn.click();
      await new Promise((r) => setTimeout(r, 200));
    }
    const countBtn = page.locator("button:has-text('Transaction Count')").first();
    if (await countBtn.isVisible()) {
      await countBtn.click();
      await new Promise((r) => setTimeout(r, 200));
    }

    // Test Pie Chart View Toggles
    const actualPieBtn = page.locator("button:has-text('Actual Labels')").first();
    if (await actualPieBtn.isVisible()) {
      await actualPieBtn.click();
      await new Promise((r) => setTimeout(r, 200));
    }
    const confusionPieBtn = page.locator("button:has-text('Confusion Outcomes')").first();
    if (await confusionPieBtn.isVisible()) {
      await confusionPieBtn.click();
      await new Promise((r) => setTimeout(r, 200));
    }

    const mode1Screenshot = path.join(SCREENSHOTS_DIR, `${vp.name}_02_results_fixed_threshold.png`);
    await page.screenshot({ path: mode1Screenshot, fullPage: true });
    console.log(`✓ Saved: ${path.basename(mode1Screenshot)}`);

    // Mode 2: Fixed Capacity: 50 / 1k
    const capModeBtn = page.locator("button:has-text('Fixed Capacity: 50 alerts / 1,000')").first();
    if (await capModeBtn.isVisible()) {
      await capModeBtn.click();
      await new Promise((r) => setTimeout(r, 400));
    }
    const mode2Screenshot = path.join(SCREENSHOTS_DIR, `${vp.name}_03_results_fixed_capacity.png`);
    await page.screenshot({ path: mode2Screenshot, fullPage: true });
    console.log(`✓ Saved: ${path.basename(mode2Screenshot)}`);

    // Verify Flagged Transactions Table is visible
    const flaggedHeader = page.locator("text=Transaction Evaluation Records & Triage").first();
    const hasFlaggedTable = await flaggedHeader.isVisible();
    console.log(`✓ Transaction Records Table visible: ${hasFlaggedTable}`);

    // Test Inspect modal
    const inspectBtn = page.locator("button:has-text('Inspect')").first();
    if (await inspectBtn.isVisible()) {
      await inspectBtn.click();
      await new Promise((r) => setTimeout(r, 300));
      const inspectClose = page.locator("button:has-text('Close Inspector')").first();
      if (await inspectClose.isVisible()) {
        await inspectClose.click();
        await new Promise((r) => setTimeout(r, 200));
      }
    }

    // 3. Navigate to Model Registry / Benchmark Page
    const registryNav = page.locator("button:has-text('Model Registry'), button:has-text('Models')").first();
    if (await registryNav.isVisible()) {
      await registryNav.click();
      await new Promise((r) => setTimeout(r, 600));

      // View 1: Comparative Bar Chart
      const barScreenshot = path.join(SCREENSHOTS_DIR, `${vp.name}_04_model_registry_barchart.png`);
      await page.screenshot({ path: barScreenshot, fullPage: true });
      console.log(`✓ Saved: ${path.basename(barScreenshot)}`);

      // View 2: Performance Heatmap
      const heatmapTab = page.locator("button:has-text('Performance Heatmap')").first();
      if (await heatmapTab.isVisible()) {
        await heatmapTab.click();
        await new Promise((r) => setTimeout(r, 500));
        const heatmapScreenshot = path.join(SCREENSHOTS_DIR, `${vp.name}_05_model_registry_heatmap.png`);
        await page.screenshot({ path: heatmapScreenshot, fullPage: true });
        console.log(`✓ Saved: ${path.basename(heatmapScreenshot)}`);

        // Test Accessible Table View Toggle
        const tableToggle = page.locator("button:has-text('Accessible Table')").first();
        if (await tableToggle.isVisible()) {
          await tableToggle.click();
          await new Promise((r) => setTimeout(r, 400));
          const tableScreenshot = path.join(SCREENSHOTS_DIR, `${vp.name}_06_model_registry_table_view.png`);
          await page.screenshot({ path: tableScreenshot, fullPage: true });
          console.log(`✓ Saved: ${path.basename(tableScreenshot)}`);
        }
      }
    }

    await context.close();
  }

  await browser.close();
  console.log(`\n🎉 Verification & Screenshot capture completed successfully across all 6 viewports!`);
}

runCapture().catch((err) => {
  console.error('Fatal error during capture:', err);
  process.exit(1);
});
