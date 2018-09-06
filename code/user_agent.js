const puppeteer = require('puppeteer');

(async () => {
  const browser = await puppeteer.launch({
        executablePath: '/opt/google/chrome/google-chrome',
        headless: true,  // true, false
        // Generating a pdf is currently only supported in Chrome headless.
        // args: ['--no-sandbox']
        args: ['--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36']
        // args: ['--user-agent=PuppeteerAgent']
      }
  );
  const user_agent = await browser.userAgent()

  console.log('user_agent:', user_agent)

  await browser.close();
})();
