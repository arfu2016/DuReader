const puppeteer = require('puppeteer');

(async () => {
// 异步函数
  const browser = await puppeteer.launch({
        executablePath: '/usr/bin/google-chrome-stable',
        // google-chrome
        headless: false,
        // args: ['--no-sandbox']
      }
  );
  const user_agent = await browser.userAgent()
  console.log('user_agent:', user_agent)

  const page = await browser.newPage();
  // 异步的执行
  await page.goto('https://www.weibo.com', {waitUntil: 'networkidle0'});
  // networkidle2
  // https://example.com
  await page.waitForNavigation({waitUntil: 'load'})
  await page.screenshot({path: 'weibo.png'});

  await browser.close();
})();
