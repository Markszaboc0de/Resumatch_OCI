import json
import time
import requests
import os
from app import app, db, Job_Descriptions

def geocode_location(city, country):
    """
    Fetches latitude and longitude from Open-Meteo Geocoding API.
    """
    query = f"{city}"
    if country:
        query += f", {country}"
    
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={query}&count=1&format=json"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        if 'results' in data and len(data['results']) > 0:
            result = data['results'][0]
            lat = result.get('latitude')
            lon = result.get('longitude')
            return lat, lon
    except Exception as e:
        print(f"Error geocoding {query}: {e}")
    
    return None, None

def build_location_cache():
    CACHE_FILE = 'location_cache.json'
    
    # Load existing cache to avoid re-querying
    cache = {}
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, 'r') as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}
                
    with app.app_context():
        # Get all unique active locations
        locations = db.session.query(Job_Descriptions.city, Job_Descriptions.country)\
            .filter(Job_Descriptions.active_status == True, 
                    Job_Descriptions.city.isnot(None),
                    Job_Descriptions.city != '')\
            .distinct().all()
            
        print(f"Found {len(locations)} unique locations.")
        
        updates = 0
        for city, country in locations:
            key = f"{city},{country}" if country else f"{city}"
            
            if key not in cache:
                print(f"Geocoding: {key}...")
                lat, lon = geocode_location(city, country)
                
                if lat is not None and lon is not None:
                    cache[key] = {'lat': lat, 'lon': lon}
                    updates += 1
                else:
                    cache[key] = None # Mark as failed to avoid re-querying every time unless cleared
                
                # Sleep briefly to respect API rate limits (Open Meteo allows ~10,000 requests/day, but good practice)
                time.sleep(0.5)
                
        if updates > 0:
            with open(CACHE_FILE, 'w') as f:
                json.dump(cache, f, indent=4)
            print(f"Saved {updates} new locations to cache.")
        else:
            print("No new locations to cache.")

if __name__ == "__main__":
    build_location_cache()
