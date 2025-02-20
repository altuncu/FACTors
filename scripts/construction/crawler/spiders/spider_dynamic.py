import scrapy
from ..util import *
from ..items import FCscraperItem
from scrapy.selector import Selector

class DynamicSpider(scrapy.Spider):
    """A template spider which can use to implement a crawler for a dynamic fact-checking website.
    - Fill in the required fields and CSS selectors to scrape the required data.
    - If the scraped data needs further postprocessing, add the required code in the parse_content method.

    Args:
        scrapy (scrapy.Spider): default scrapy spider class

    Yields:
        FCscraperItem: An item object containing the scraped data described in the items.py file
    """
    name = '<Enter Spider Name Here>'
    allowed_domains = ['<Enter Allowed Domain Here>']
    start_urls = ['<Enter Start URL Here>']
    url_set = set()

    custom_settings = { 
        'FEEDS': {
            '<Path to the output file>': {
                'format': 'csv', 
                'overwrite': True,
            }
        },
        # Settings for Playwright
        'DOWNLOAD_HANDLERS': {
            "http": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
            "https": "scrapy_playwright.handler.ScrapyPlaywrightDownloadHandler",
        },
        'PLAYWRIGHT_LAUNCH_OPTIONS': {
            "headless": False,
        },
    }

    # Enter the CSS selectors for the required fields
    css_selectors = {
        'claim_review': 'script[type="application/ld+json"]::text',
        'articles': '',
        'next_page': '',
        'content': [''],
        'title': '',
        'date_published': '',
        'author_name': '',
        'organisation': '',
        # For the cases when ClaimReview is not present
        'claim': '',
        'rating': ''
    }

    def start_requests(self):
        """Function to start the requests to the website

        Yields:
            Request: Calls the parse method to extract the links to the articles
        """
        yield scrapy.Request(
            url=f"{self.start_urls[0]}",
            meta=dict(
                playwright=True,
                playwright_include_page=True,
                
            ),
            callback=self.parse,
        )

    async def parse(self, response):
        """Function to parse the response and extract the links to the articles

        Args:
            response (Response): Response object containing the HTML content of the page

        Yields:
            Request: Calls the parse_content method to extract the required data from the articles
        """
        page = response.meta["playwright_page"]
        page.set_default_timeout(10000)
        
        await page.wait_for_timeout(5000)
        try:
            while True:
                button = page.locator(self.css_selectors['next_page'])
                await button.wait_for()

                if not button:
                    print("No 'Load more' button found.")
                    break
                
                is_disabled = await button.is_disabled()
                if is_disabled:
                    print("Button is disabled.")
                    break
                
                await button.scroll_into_view_if_needed()
                await button.click()
                await page.wait_for_timeout(750)
        except Exception as error:
            print(f"Error: {error}")
            pass
        
        print("Getting content")
        content = await page.content()
        print("Parsing content")
        selector = Selector(text=content)
        links = selector.css(self.css_selectors['articles']).getall()

        for link in links:
            if self.allowed_domains[0] in link:
                link = response.urljoin(link)
                if link not in self.url_set:
                    self.url_set.add(link)
                    yield scrapy.Request(
                        url=link, headers=self.headers, callback=self.parse_content
                    )
    
    def parse_content(self, response):
        """Function to extract the required data from the articles

        Args:
            response (Response): Response object containing the HTML content of the page

        Yields:
            FCscraperItem: One or more items containing the scraped data described in the items.py file
        """
        items = init_item(FCscraperItem(url=response.request.url), response, self.css_selectors)
        for item in items:
            yield item

    