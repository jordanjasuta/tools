import concurrent.futures
import sys
import time
import requests
import json
from geopy.geocoders import Nominatim



# API INFO
geolocator = Nominatim(user_agent="trythis")
url = 'http://www.7timer.info/bin/api.pl'




# define function to use API on locations
def get_temps(country_js):   # doc[country]
    for city in country_js:
        print(country_js[city]['name'])
        location = geolocator.geocode(country_js[city]['name'])
        country_js[city]['lon'] = location.longitude
        country_js[city]['lat'] = location.latitude

        params = {}
        params['lon'] = location.longitude
        params['lat'] = location.latitude
        params['product'] = 'civil'
        params['output'] = 'json'

        x = requests.get(url, params=params)

        temps = []
        for timepoint in json.loads(x.text)['dataseries']:
            temps.append(timepoint['temp2m'])

        country_js[city]['temps'] = temps

        return country_js


def process_pages_in_parallel(pages, max_workers):
    # We can use a with statement to ensure threads are cleaned up promptly
    page_results = dict()
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_result = {executor.submit(get_temps, doc[country]): country for country in doc}
        for result in concurrent.futures.as_completed(future_to_result):
            completed_page = future_to_result[result]

            try:
                updated_info = result.result()
            except Exception as exc:
                print('exception generated:', (completed_page, exc))
                page_results = None
            else:
                print('Page is done: ', (completed_page, updated_info))
                page_results = updated_info
    end = time.time()
    print('Elapsed = %.1f sec with %d threads' % (end - start, max_workers))

    return page_results


if __name__ == '__main__':
    max_workers = int(sys.argv[1])

    file = 'doc.json'
    with open(file) as f:
        doc = json.load(f)

    page_results = process_pages_in_parallel(doc, max_workers)

    # Assemble results - comes back unordered
    for page_num in sorted(page_results.keys()):
        print('%d: %s' % (page_num, page_results))
