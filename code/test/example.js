const puppeteer = require('puppeteer');

(async () => {
// 异步函数
  const browser = await puppeteer.launch({
        executablePath: '/home/app/chrome-linux/chrome',
        headless: true,
        args: ['--no-sandbox']
      }
  );
  const page = await browser.newPage();
  // 异步的执行
  await page.goto('https://example.com');
  await page.screenshot({path: 'example.png'});

  await browser.close();
})();
