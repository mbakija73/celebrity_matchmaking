#gather additional celebrities by scroll through all celebrities in popular page by letter
import requests
import json
import os
from bs4 import BeautifulSoup
import re
import time
import string
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options


def get_name(url):
    pattern = r'(?<=\/dating\/).*'
    match = re.search(pattern, url)
    if match:
        return match.group(0)
    return None


def get_html(url):
    try:
        #time.sleep()
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.text
        else:
            print(f"Error: Unable to fetch URL. Status code: {response.status_code}")
            return None
    except requests.exceptions.Timeout:
        print(f"Error: Request to {url} timed out.")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: {e}")
        return None
    except requests.exceptions.Timeout as e:
        print(f"Timeout Error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: {e}")
        return None

def extract_data_from_html(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    table = soup.find('div', id='ff-dating-history-table')
    relationships = []
    
    if table:
        #skip the header row
        rows = table.find_all('tr')[1:]  
        for row in rows:
            columns = row.find_all('td')
            
            if len(columns) < 8:
                continue
            
            #get data from each column
            name_tag = columns[1].find('a')
            name = name_tag.text.strip() if name_tag else 'Unknown'
            href = name_tag['href'] if name_tag else 'Unknown'
            relationship_type = columns[2].text.strip()
            start_date = columns[4].text.strip()
            end_date = columns[5].text.strip()
            length = columns[6].text.strip()
            
            relationships.append({
                'name': name,
                'href': href,
                'relationship_type': relationship_type,
                'start_date': start_date,
                'end_date': end_date,
                'length': length
            })

    details_section = soup.find('h4', class_='ff-auto-details')
    details = {}
    
    if details_section:
        details_table = details_section.find_next_sibling('table')
        
        if details_table:
            rows = details_table.find_all('tr')
            
            for row in rows:
                cols = row.find_all('td')
                if len(cols) == 2:
                    key = cols[0].text.strip()
                    value = cols[1].text.strip()
                    
                    if key == 'First Name':
                        details['First Name'] = value
                    elif key == 'Middle Name':
                        details['Middle Name'] = value
                    elif key == 'Last Name':
                        details['Last Name'] = value
                    elif key == 'Maiden Name':
                        details['Maiden Name'] = value
                    elif key == 'Full Name at Birth':
                        details['Full Name at Birth'] = value
                    elif key == 'Alternative Name':
                        details['Alternative Name'] = value
                    elif key == 'Birthday':
                        details['Birthday'] = value
                    elif key == 'Birthplace':
                        details['Birthplace'] = value
                    elif key == 'Height':
                        details['Height'] = value
                    elif key == 'Weight':
                        details['Weight'] = value
                    elif key == 'Zodiac Sign':
                        details['Zodiac Sign'] = value
                    elif key == 'Sexuality':
                        details['Sexuality'] = value
                    elif key == 'Religion':
                        details['Religion'] = value
                    elif key == 'Ethnicity':
                        details['Ethnicity'] = value
                    elif key == 'Nationality':
                        details['Nationality'] = value
                    elif key == 'Occupation':
                        details['Occupation'] = value
                    
                    expected_keys = [
                        'First Name', 'Middle Name', 'Last Name', 'Maiden Name', 
                        'Full Name at Birth', 'Alternative Name', 'Birthday', 
                        'Birthplace', 'Height', 'Weight', 'Zodiac Sign', 
                        'Sexuality', 'Religion', 'Ethnicity', 'Nationality', 
                        'Occupation'
                    ]
                    for key in expected_keys:
                        if key not in details:
                            details[key] = 'NA'
    
    return {
        'relationships': relationships,
        'details': details
    }

def save_to_json(data, file_name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'info/')
    file_path = os.path.join(file_path, file_name)
    #print(file_path)
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)

def save_to_json_2(data, filename): 
    if os.path.exists(filename):
        with open(filename, 'r') as file:
            try:
                existing_data = json.load(file)
            except json.JSONDecodeError:
                existing_data = []  
    else:
        existing_data = []

    #combine the new data with the existing data
    if isinstance(existing_data, list):
        existing_data.extend(data) 
    else:
        print("Error: Existing data is not a list, can't append.")

    with open(filename, 'w') as file:
        json.dump(existing_data, file, indent=4)

   

def update_names_file(name):
    current_directory = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_directory, 'names.json')
    names = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            try:
                names = json.load(file)
            except json.JSONDecodeError:
                names = []
    else:
        
        #if file non existant create and initialize it
        with open(file_path, 'w') as file:
            json.dump(names, file)

    #check if the name is already in the list
    if name not in names:
        names.append(name)
        with open(file_path, 'w') as file:
            json.dump(names, file, indent=4)
        
        n = get_name(name)
        print(f'"{n}" added to names.json.')
        html_code = get_html(name)
        data = extract_data_from_html(html_code)
        n = n + '.json'
        save_to_json(data, n)
        for relationship in data['relationships']:
            href = relationship['href']
            update_names_file(href)
    else:
        print(f'Name "{name}" already exists in names.json.')

def gather_scroll():
    gecko_driver_path =  '/usr/local/bin/geckodriver' #REPLACE with local file
    firefox_binary_path = '/usr/bin/firefox'
    options = webdriver.FirefoxOptions()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-gpu')
    options2 = Options()
    options2.set_preference('network.stricttransportsecurity.preloadlist', False)
    options2.set_preference('security.enterprise_roots.enabled', True)
    service = Service(gecko_driver_path)
    driver = webdriver.Firefox(service=service, options=options2)

    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_directory, 'info/')
    files = os.listdir(folder_path)

    #iterate through alphabet
    for letter in string.ascii_lowercase:
        url = f'https://www.whosdatedwho.com/popular?letter={letter}'
        driver.get(url)
        for i in range(6):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
        html_code = driver.execute_script("return document.documentElement.outerHTML;")
        soup = BeautifulSoup(html_code, 'html.parser')
        table = soup.find('div', class_='ff-box-grid ff-medium-square')

        #get all the href celebrity names
        names = [a['href'] for a in table.find_all('a', href=True)]

        for name in names:
            print(name)

        #save the URLs to a JSON file
        save_to_json2(names, "atoznames.json")
        

def process_unique_names(atoz_file, unique_file, folder):
    #load data from atoz.json
    with open(atoz_file, 'r') as file:
        try:
            atoz_data = json.load(file)
        except json.JSONDecodeError:
            print(f"Error reading {atoz_file}, it might be empty or corrupted.")
            return

    #find all the celebrities from popular page that have already been gathered
    #save into list of unique names
    unique_names = []

    #check if uniqueatoz.json already exists
    if os.path.exists(unique_file):
        with open(unique_file, 'r') as file:
            try:
                unique_names = json.load(file)
            except json.JSONDecodeError:
                unique_names = []

    #iterate through names in atoz_data
    for url in atoz_data:
        name = get_name(url)
        file_path = os.path.join(folder, f"{name}.json")

        #if the file doesn't exist add the name to unique_names
        if not os.path.exists(file_path):
            unique_names.append(name)

    #update unique names in uniqueatoz.json
    with open(unique_file, 'w') as file:
        json.dump(unique_names, file, indent=4)

    #print the lengths of atoz.json and uniqueatoz.json
    print(f"Length of {atoz_file}: {len(atoz_data)}")
    print(f"Length of {unique_file}: {len(unique_names)}")



def main():
    
    gather_scroll()
    process_unique_names('atoznames.json', 'uniqueatoz.json', 'info')
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_directory, 'info/')
    with open('uniqueatoz.json', 'r') as file:
        files = json.load(file)
    print(len(files))
    
    for file_name in files:

        file_path = os.path.join(folder_path, file_name + '.json')
        if(not os.path.isfile(file_path)):
            update_names_file("https://www.whosdatedwho.com/dating/" + file_name)
        else:
            print(file_name,"already done")
        

if __name__ == "__main__":
    main()