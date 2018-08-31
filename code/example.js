const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
        executablePath: '/home/app/chrome-linux/chrome',
        headless: false,
        args: ['--no-sandbox']
      }
  );
  const page = await browser.newPage();
  await page.goto('https://example.com');
  await page.screenshot({path: 'example.png'});

  await browser.close();
})();
