import concurrent.futures
import sys
import time
import requests
import json
from geopy.geocoders import Nominatim



# API INFO
geolocator = Nominatim(user_agent="morenames")
url = 'http://www.7timer.info/bin/api.pl'




# define function to use API on locations
def get_temps(country_js):   # doc[country]
    for city in country_js:
        for each in ['capital', 'second_city']:
            if each in country_js.keys():
                # Geocoding can be done through Nominatim but
                # it will cut access after 3 threads max. :/
                # With an unlimited API, the benefits of
                # multithreading can have even greater returns.
                location = geolocator.geocode(country_js[each]['name'])
                country_js[each]['lon'] = location.longitude
                country_js[each]['lat'] = location.latitude

                params = {}
                params['lon'] = country_js[each]['lon']
                params['lat'] = country_js[each]['lat']
                params['product'] = 'civil'
                params['output'] = 'json'

                x = requests.get(url, params=params)

                temps = []
                for timepoint in json.loads(x.text)['dataseries']:
                    temps.append(timepoint['temp2m'])

                    country_js[each]['temps'] = temps

        return country_js


def process_pages_in_parallel(doc, max_workers):
    # We can use a with statement to ensure threads are cleaned up promptly
    page_results = dict()
    start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Start the load operations and mark each future with its URL
        future_to_result = {executor.submit(get_temps, doc[country]): country for country in doc}
        # future_to_result = {executor.submit(get_temps, doc[country]): (country: {(inner_v) for (city, inner_v) in outer_v.items()} for (country, outer_v) in doc.items())}

        for result in concurrent.futures.as_completed(future_to_result):
            completed_page = future_to_result[result]

            try:
                updated_info = result.result()
            except Exception as exc:
                print('exception generated:', (completed_page, exc))
                # updated_json = None
            else:
                print('Country: ', completed_page)
                print('Updated info: ', updated_info)
                doc[completed_page] = updated_info
    end = time.time()
    print('Elapsed = %.1f sec with %d threads' % (end - start, max_workers))

    return doc


if __name__ == '__main__':
    max_workers = int(sys.argv[1])

    file = 'doc_w_geo.json'
    with open(file) as f:
        doc = json.load(f)

    updated_data = process_pages_in_parallel(doc, max_workers)

    # save to json
    with open('updated_data.json', 'w') as f:
        json.dump(updated_data, f)
