const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({
        executablePath: '/opt/google/chrome/google-chrome',
        headless: true,  // true, false
        // args: ['--user-agent=PuppeteerAgent']
      }
    );

    const page = await browser.newPage();
    await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/67.0.3372.0 Safari/537.36');
    await page.goto('https://www.baidu.com');

    console.log(await page.evaluate('navigator.userAgent'));

    await browser.close();
})();
