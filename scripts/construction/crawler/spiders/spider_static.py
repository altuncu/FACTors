import scrapy
from ..util import *
from ..items import FCscraperItem

class StaticSpider(scrapy.Spider):
    """A template spider which can use to implement a crawler for a static fact-checking website.
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

    def parse(self, response):
        """Function to parse the response and extract the links to the articles

        Args:
            response (Response): Response object containing the HTML content of the page

        Yields:
            Request: Calls the parse_content method to extract the required data from the articles
        """
        links = response.css(self.css_selectors['articles']).getall()
        next_page = response.css(self.css_selectors['next_page']).get()
        for link in links:
            link = response.urljoin(link)
            if link not in self.url_set:
                self.url_set.add(link)
                yield scrapy.Request(
                    url=link, headers=self.headers, callback=self.parse_content
                )
        if next_page:
            next_page = response.urljoin(next_page)
            if next_page not in self.url_set:
                self.url_set.add(next_page)
                yield scrapy.Request(
                    url=next_page, headers=self.headers, callback=self.parse
                )
    
    def parse_content(self, response):
        """Function to extract the required data from the articles

        Args:
            response (Response): Response object containing the HTML content of the page

        Yields:
            FCscraperItem: One or more items containing the scraped data described in the items.py file
        """
        items = init_item(FCscraperItem(url=response.request.url), response, self.css_selectors)
        for item in items: # Using a for loop for the cases when multiple claims are present
            yield item

    