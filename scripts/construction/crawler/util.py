import json
import re
import html2text

# Initialise item(s) from the response
def init_item(instance, response, css_selectors):
    """Function to initialise the item(s) from the response based on the given CSS selectors

    Args:
        instance (FCscraperItem): An empty instance of the FCscraperItem class to be filled with the scraped data
        response (Response): Response object containing the HTML content of the page
        css_selectors (dict): CSS selectors for the required fields

    Returns:
        FCscraperItem: Initialised instance of the FCscraperItem class containing the scraped data
    """
    items = set()

    content = extract_content(response, css_selectors)
    title, author_name, organisation = extract_metadata(response, css_selectors)

    # Get the schema.org JSON
    cr_found = False
    if css_selectors['claim_review'] != '':
        scripts = response.css(css_selectors['claim_review']).getall()
        for script in scripts:
            if 'ClaimReview' in script:
                item = process_script(script, instance)
                if item:
                    items.add(item)
                    cr_found = True
    if not cr_found:
        item = extract_claim_rating(instance, response, css_selectors)
        items.add(item)
                            
    for i in items:
        i["title"] = title
        i["author_name"] = clean_content(", ".join(author_name))
        i["organisation"] = organisation
        if isinstance(content, list) and content != []:
            i["content"] = clean_content(" ".join(content))
        else:
            i["content"] = None
        date_str = response.css(css_selectors['date_published']).get()
        if date_str:
            i["date_published"] = date_str
    
    return list(items)

def extract_content(response, css_selectors):
    """Helper function to extract the content from the response based on the given CSS selectors

    Args:
        response (Response): Response object containing the HTML content of the page
        css_selectors (dict): CSS selectors for the required fields

    Returns:
        string: Extracted content from the response
    """
    for path in css_selectors['content']:
        content = response.css(path).getall()
        check = ''.join(content).replace('&nbsp;', '').replace('<br>', '').strip()
        if content and content != [] and check != '' and len(content) > 1:
            return content

def extract_metadata(response, css_selectors):
    """Helper function to extract the metadata from the response based on the given CSS selectors

    Args:
        response (Response): Response object containing the HTML content of the page
        css_selectors (dict): CSS selectors for the required fields

    Returns:
        string: Extracted title, author, and organisation from the response
    """
    title = response.css(css_selectors['title']).get()
    if css_selectors['organisation'] != '':
        organisation = response.css(css_selectors['organisation']).get()
    else:
        organisation = ''
    if css_selectors['author_name'] != '':
        author_name = response.css(css_selectors['author_name']).getall()
        author_name = list(set(author_name))
    else:
        author_name = organisation
    return title, author_name, organisation

def process_script(script, instance):
    """Helper function to look for the ClaimReview schema

    Args:
        script (string): One or more JSON scripts that might contain the ClaimReview schema
        instance (_type_): _description_

    Returns:
        FCscraperItem: The item containing the scraped data
    """
    try:
        json_script = json.loads(script)
        if not isinstance(json_script, list):
            json_script = [json_script]
        for script in json_script:
            item = seek_claimreview(instance, script)
            if item:
                if 'claim' not in item:
                    item['claim'] = 'No claim found'
                if 'rating' not in item:
                    item['rating'] = 'No rating found'
                return item
    except Exception as e:
        print(e)
        print("Error parsing JSON script:", script)
        return None

def seek_claimreview(item, script):
    """Helper function to handle different formats of the ClaimReview schema

    Args:
        item (FCscraperItem): FCscraperItem instance to be filled with the scraped data
        script (string): JSON script that might contain the ClaimReview schema

    Returns:
        FCscraperItem: FCscraperItem instance containing the scraped data
    """
    type_param = '@type'
    graph_param = '@graph'
    if type_param in script and script[type_param] == 'ClaimReview':
        item = process_claimreview(script, item)
        return item
    elif graph_param in script and script[graph_param][0][type_param] == 'ClaimReview':
        item = process_claimreview(script[graph_param][0], item)
        return item
    else:
        return None

def process_claimreview(script, item):
    """Helper function to extract the required fields from the ClaimReview schema

    Args:
        script (string): ClaimReview schema JSON script
        item (FCscraperItem): FCscraperItem instance to be filled with the scraped data

    Returns:
        FCscraperItem: FCscraperItem instance containing the scraped data
    """
    if "claimReviewed" in script:
        item["claim"] = clean_content(script["claimReviewed"])
    if "datePublished" in script:
        item["date_published"] = script["datePublished"]
    if "reviewRating" in script and "alternateName" in script["reviewRating"]:
        item["rating"] = script["reviewRating"]["alternateName"]
    elif "reviewRating" in script and "ratingValue" in script["reviewRating"]:
        item["rating"] = "Rating Value: " + str(script["reviewRating"]["ratingValue"])
    return item

def extract_claim_rating(item, response, css_selectors):
    """Alternative function to extract the claim and rating from the response 
    when the ClaimReview schema is not present

    Args:
        item (FCscraperItem): FCscraperItem instance to be filled with the scraped data
        response (Response): Response object containing the HTML content of the page
        css_selectors (dict): CSS selectors for the required fields

    Returns:
        FCscraperItem: FCscraperItem instance containing the scraped data
    """
    item["date_published"] = response.css(css_selectors['date_published']).get()
    if "rating" in css_selectors and css_selectors['rating'] != '' and response.css(css_selectors['rating']).get():
        item["rating"] = clean_content(response.css(css_selectors['rating']).get())
    else:
        item["rating"] = 'No rating found'
        
    if 'claim' in css_selectors and css_selectors['claim'] != '' and response.css(css_selectors['claim']).get():
        item["claim"] = clean_content(response.css(css_selectors['claim']).get())
    elif 'claim_alternative' in css_selectors and css_selectors['claim_alternative'] != '' and response.css(css_selectors['claim_alternative']).get():
        item["claim"] = clean_content(response.css(css_selectors['claim_alternative']).get())
    else:
        item["claim"] = 'No claim found'
    return item

def clean_content(text):
    """Helper function to clean the extracted text

    Args:
        text (string): Input text to be cleaned

    Returns:
        string: Cleaned text
    """
    text = text.replace(r'<blockquote>', '"')
    text = text.replace(r'</blockquote>', '"')
    text = text.replace(r'<strong>', '')
    text = text.replace(r'</strong>', '')

    h = html2text.HTML2Text()
    html2text.hn = lambda _:0
    h.ignore_links = True
    h.ignore_images = True
    h.ignore_emphasis = True
    h.ignore_links = True
    h.ignore_tables = True
    h.body_width = 0
    h.unicode_snob = True
    h.single_line_breaks = True
    output = h.handle(text)
    output = re.sub(r'\n+', ' ', output)

    return re.sub(' +', ' ', output).strip()
