const puppeteer = require('puppeteer');

(async () => {
    const browser = await puppeteer.launch({
        executablePath: '/opt/google/chrome/google-chrome',
        headless: true,  // true, false
        args: ['--user-agent=Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/68.0.3440.106 Safari/537.36']
        // args: ['--user-agent=PuppeteerAgent']
      }
    );
    const user_agent = await browser.userAgent()
    console.log('user_agent:', user_agent)

    const context = await browser.createIncognitoBrowserContext();
    // Create a new page in a pristine context.
    const page = await context.newPage();
    // Do stuff
    page.on('load', () => console.log("Loaded: " + page.url()));
    // page.on('request', res => console.log(res));
    await page.goto('https://www.weibo.com', {waitUntil: 'networkidle2'});
    await page.goto(page.url(), {waitUntil: 'load'});
    // networkidle2
    // await page.waitForNavigation({waitUntil: 'load'})

    console.log(await page.content())

    await browser.close();
})();
