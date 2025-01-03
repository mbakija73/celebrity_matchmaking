#starting with one celebrity(katy perry) and branching out collecting most of nodes (27,794) from one large connected graph
import requests
import json
import os
from bs4 import BeautifulSoup
import re
import time

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
    
    #get relationships
    table = soup.find('div', id='ff-dating-history-table')
    relationships = []
    
    if table:
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

    
    #get all features
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
                    
                    #add details if missing them
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
                    
                    #all expected keys (not using all as features of course)
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
    #save relationship data and feature personal data to the json
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
        
        #if the file non-existant, create and initialize with an empty list
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


def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(current_directory, 'info/')
    files = os.listdir(folder_path)
    
    #iterate over all files in info
    for file_name in files:
        #check if the file is a JSON file
        if file_name.endswith('.json'):
            file_path = os.path.join(folder_path, file_name)
            
            #open and read 
            with open(file_path, 'r') as file:
                try:
                    data = json.load(file)
                    
                    if 'relationships' in data:
                        for relationship in data['relationships']:
                            href = relationship.get('href')
                            update_names_file(href)
                            
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {file_name}")
    #url = 'https://www.whosdatedwho.com/dating/katy-perry'
    #update_names_file(url)
   
if __name__ == "__main__":
    main()





