const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
        executablePath: '/opt/google/chrome/google-chrome',
        headless: true,  // false
        // Generating a pdf is currently only supported in Chrome headless.
        // args: ['--no-sandbox']
      }
  );
  const page = await browser.newPage();
  // await page.goto('chrome://version'); this does not work
  await page.goto('https://www.computerhope.com/issues/ch001329.htm', {waitUntil: 'networkidle2'});
  await page.screenshot({path: 'chrome_version.png'});
  await page.pdf({path: 'chrome_version.pdf', format: 'A4'});

  await browser.close();
})();
