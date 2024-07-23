import asyncio
from playwright.async_api import async_playwright
from bs4 import BeautifulSoup
import csv, json
from tranco import Tranco
import aiosqlite
from testing import predict,main1

async def extract_html(url):
    """
        Loads a chromium browser and navigates to the url input as parameter
            -> It waits until the page reaches a netwrok idle status or until a timout is reached 
            -> It then extracts the html contents and returns it
    """
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  
        context = await browser.new_context()
        page = await context.new_page()
        try:
            
            print(f"Navigating to {url}")
            await page.goto(url, wait_until='load', timeout=45000)  # Increased timeout for loading
            print(f"Page navigation completed for {url}")
            await page.wait_for_load_state('networkidle', timeout=15000)  # Additional wait for network to be idle
            print(f"Network is idle for {url}")

        except Exception as e:
            print(f"Page didn't fully load: {e}")

        html = await page.content()
        await browser.close()
        return html

async def parse_html(html):
    """
        Parses the html content input as paramenter using BeautifulSoup
            -> Extracts all the text from tags bellow 
            -> Appends the text to a list text_sections which the function returns

            -> The tags extract alot of unnecessary text and so a better selection will ensure better inference times for the model
    """
    soup = BeautifulSoup(html, 'html.parser')
    text_sections = []
    
    # Remove all script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Extract text from all divs and other elements
    for div in soup.find_all(['div', 'p', 'span', 'li', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):  # Need to play around with these, some unnecessary
        text = div.get_text(separator=" ", strip=True)
        if text:  
            text_sections.append(text)
    
    return text_sections

# ---> Can use to see what extracted contents look like <---
# async def save_to_json(text_sections, filename):
    
#     processed_sections = []
#     for section in text_sections:
#         words = section.split()
#         if len(words) > 20:  # if section is longer than 20 words, split it
#             for i in range(0, len(words), 20):
#                 processed_sections.append(' '.join(words[i:i+20]))
#         else:
#             processed_sections.append(section)

#     processed_sections = list(set(processed_sections)) # Removing duplicates

#     # Save to a JSON file
#     with open(filename, 'w', encoding='utf-8') as jsonfile:
#         json.dump(processed_sections, jsonfile, ensure_ascii=False, indent=4)
#############################################################

async def save_to_db(url, rank, age_verification, category=None, host_country=None, visits=None, error=None, verification_type=None):
    """
        Creates and saves to the sqlite database: 
            -> It creates it if non-existent in the current repertory
            -> It then inserts the parameters passed when calling the function into the database
    """
    
    async with aiosqlite.connect('websites.db') as db:
        await db.execute('''
            CREATE TABLE IF NOT EXISTS website (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL,
                rank INTEGER,
                age_verification BOOLEAN,
                category TEXT,
                host_country TEXT,
                visits INTEGER,
                error TEXT,
                verification_type TEXT   
            )
        ''')
        await db.execute('''
            INSERT INTO website (url, rank, age_verification, category, host_country, visits, error, verification_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (url, rank,  age_verification, category, host_country, visits, error, verification_type))
        await db.commit()

async def extract_text_analyse(text_sections):
    """
        Prepares and returns a list of text to analyse without duplicates: 
            -> It splits sentences of more that 20 words, as this would cause a problem for the model later
            -> It then creates a new list without duplicates and returns it
    """
    # Split sections longer than 20 words into multiple sections
    processed_sections = []
    for section in text_sections:
        words = section.split()
        if len(words) > 20:  # If section is longer than 20 words, split it
            for i in range(0, len(words), 20):
                processed_sections.append(' '.join(words[i:i+20]))
        else:
            processed_sections.append(section)

    processed_sections = list(set(processed_sections)) # Removing duplicates
    return processed_sections

async def main():
    """
        Calls in all the functions to extract, analyse and save to the database: 
            -> it gets the links from tranco -user can specify how many to go through 
            -> It then runs the model on the extracted text by calling main1 from testing.py
            -> finally it saves the results to the database
    """
    t = Tranco(cache=True, cache_dir='.tranco')
    latest_list = t.list()
    url_list = latest_list.top(1) 
    for rank,url in enumerate(url_list):
        try:
            print(f"Curent URL :{url}")
            print(f"Processing URL: {url}")

            domain_name = url.split('.')[0]  # Adjusted to extract the correct domain part
            filename = domain_name + '.json'
            url = 'https://www.'+ url +'/' # Necessary for url's to work
            #url = 'https://www.leafly.com/'     --> use as a positive check <--
            html = await extract_html(url)            
            chunks = await parse_html(html)
            
            age_verification = 0 
            text_to_analyse = await extract_text_analyse(chunks)
            #print(text_to_analyse)
            age_verifications = main1(text_to_analyse)
            for verification in age_verifications:
                if verification >= 0.999: # Set this as acceptance threshold, this should avoid most false positives given model tends to be very sure about true positives
                    age_verification = 1
                    break
            
            await save_to_db(url, rank, age_verification)
            print(f"Successfully processed and saved {url} to dataBase")
        except Exception as e:
            print(f"An error occurred while processing {url}: {e}")
            await save_to_db(url, rank, False, error=str(e))

asyncio.run(main())