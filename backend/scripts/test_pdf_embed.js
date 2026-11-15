const path = require('path');
const fs = require('fs');
const { chromium } = require('../.venv/Lib/site-packages/playwright/driver/package/index.js');

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage();
  const pdfBytes = fs.readFileSync('results/uploads/final_demo_report_pred_1794301880_85a48325.pdf');
  const base64Pdf = pdfBytes.toString('base64');
  
  await page.setContent(`
    <!DOCTYPE html>
    <html>
      <body style="margin: 0; padding: 0; background: #eee;">
        <embed src="data:application/pdf;base64,${base64Pdf}" type="application/pdf" width="1000" height="1400" />
      </body>
    </html>
  `);
  await page.waitForTimeout(2000);
  await page.screenshot({ path: 'results/embed_test.png' });
  console.log('Saved results/embed_test.png');
  await browser.close();
})();
