# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class FCscraperItem(scrapy.Item):
    """The class for the items that will be scraped from the website

    Args:
        scrapy (Item): Default scrapy item
    """
    title = scrapy.Field()
    url = scrapy.Field()
    content = scrapy.Field()
    date_published = scrapy.Field()
    rating = scrapy.Field()
    author_name = scrapy.Field()
    claim = scrapy.Field()
    organisation = scrapy.Field()