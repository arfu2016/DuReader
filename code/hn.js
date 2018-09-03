const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
        executablePath: '/usr/bin/google-chrome-stable',
        // google-chrome
        headless: true,  // false
        // Generating a pdf is currently only supported in Chrome headless.
        // args: ['--no-sandbox']
      }
  );
  const page = await browser.newPage();
  await page.goto('https://news.ycombinator.com', {waitUntil: 'networkidle2'});
  await page.pdf({path: 'hn.pdf', format: 'A4'});

  await browser.close();
})();
