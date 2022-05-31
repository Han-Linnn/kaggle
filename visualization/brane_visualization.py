#!/usr/bin/env python3
import os
import sys
import yaml 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
#from wordcloud import WordCloud
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium
#from folium import plugins


def visualization():
    #csv_path = './train.csv'
    #cwd = os.getcwd()
    #filepath = cwd+'/train.csv'
    df = pd.read_csv(f'train.csv')
    try:
        #1.Visualization of the missing data in the form of a chart
        plt.figure(figsize = (15, 10))
        plt.title("Visualizing the Missing Data", fontsize = 20)
        msno.bar(df, color = (1, 1, 0.5), sort = "ascending", figsize = (15, 10))
        plt.savefig(f"/data/issingdata.png")
        plt.show()
        plt.close()

        #2.Visualization Location count: Bar chart representation of the locations from where the highest number of tweets originate
        custom_colors = ['#000000', '#E31E33', '#4A53E1', '#F5AD02', '#94D5EA', '#F6F8F7']
        custom_palette = sns.set_palette(sns.color_palette(custom_colors))
        sns.palplot(sns.color_palette(custom_colors), size = 1)
        plt.tick_params(axis = 'both', labelsize = 0, length = 0)
        plt.figure(figsize = (15, 13))
        ax = plt.axes()
        ax.set_facecolor('black')
        ax = ((df.location.value_counts())[:10]).plot(kind = 'bar', color = custom_colors[2], linewidth = 2, edgecolor = 'white')
        plt.title('Location Count', fontsize = 30)
        plt.xlabel('Location', fontsize = 25)
        plt.ylabel('Count', fontsize = 25)
        ax.xaxis.set_tick_params(labelsize = 15, rotation = 30)
        ax.yaxis.set_tick_params(labelsize = 15)
        bbox_args = dict(boxstyle = 'round', fc = '0.9')
        for p in ax.patches:
                ax.annotate('{:.0f}'.format(p.get_height()), (p.get_x() + 0.15, p.get_height() + 2),
                        bbox = bbox_args,
                        color = custom_colors[2],
                        fontsize = 15)
        plt.savefig(f"/data/locationcount.png")
        plt.close()

        #3.Visualization Location Map
        new_df = pd.DataFrame()
        new_df['location'] = ((df['location'].value_counts())[:10]).index
        new_df['count'] = ((df['location'].value_counts())[:10]).values
        geolocator = Nominatim(user_agent = 'Rahil')
        geocode = RateLimiter(geolocator.geocode, min_delay_seconds = 0.5)
        lat = {}
        long = {}
        for i in new_df['location']:
            location = geocode(i)
            lat[i] = location.latitude
            long[i] = location.longitude
        new_df['latitude'] = new_df['location'].map(lat)
        new_df['longitude'] = new_df['location'].map(long)
        map = folium.Map(location = [10.0, 10.0], tiles = 'CartoDB dark_matter', zoom_start = 1.5)
        #markers = []
        title = '''<h1 align = "center" style = "font-size: 35px"><b>Top 10 Tweet Locations</b></h1>'''
        for i, r in new_df.iterrows():
            loss = r['count']
            if r['count'] > 0:
                counts = r['count'] * 0.4
                folium.CircleMarker([float(r['latitude']), float(r['longitude'])], radius = float(counts), color = custom_colors[1], fill = True).add_to(map)
        map.get_root().html.add_child(folium.Element(title))
        map
        map.save(f"/data/location_map.html")
        return "Visualization is done, all files are saved to the path -> ./data/"

    except IOError as e:
        return f"ERROR: {e} ({e.errno})" 


if __name__ == "__main__":
    command = sys.argv[1]

    functions = {
        "visualization": visualization,
    }

    output = functions[command]()
    print(yaml.dump({"output": output}))