import torch
from pygsig.graph import StaticGraphTemporalSignal
from torch_geometric.data import Data
import torch_geometric.transforms as T
import pandas as pd
import json
import requests
from tqdm import tqdm
import numpy as np

class SubductionZone(object):
    def __init__(self,
                 poly_path='../datasets/subduction/polygon.geojson', # region of interest
                 rast_path='../datasets/subduction/interface.grd', # subduction interface depth
                 label_path='../datasets/subduction/nonlinear.json', # nonlinear/linear labels
                 site_path='../datasets/subduction/site_location.geojson',
                 data_path='../datasets/subduction/site_data.geojson',
                 dtw_path='../datasets/subduction/dtw.pt',
                 start_date=pd.Timestamp('2008-01-01 11:59:00+0000', tz='UTC'),
                 end_date=pd.Timestamp('2023-12-31 11:59:00+0000', tz='UTC'),
                 task='classification',
                 download=False
                ):
        self.start_date = start_date
        self.end_date = end_date
        self.poly_path = poly_path
        self.label_path = label_path
        self.rast_path = rast_path
        self.site_path = site_path
        self.data_path = data_path
        self.dtw_path = dtw_path
        self.task = task
        if download:
            self.download()
        self.load_data()
        self.X = np.stack([group[['e','n','u']].values for t,group in self.df.groupby('t')]).transpose(1,0,-1)
        if self.task == 'classification':
            self.y = np.stack([ group.label.unique() for _,group in self.df.groupby('siteID')])
        if self.task == 'regression':
            self.y = np.stack([ group.depth.unique() for _,group in self.df.groupby('siteID')])
        self.stations = self.gdf_location.siteID.values

    def download(self):

        import geopandas as gpd
        import rasterio
        from shapely.geometry import Point, Polygon
        # Load raster of subduction interface depths
        with open(self.poly_path, 'rb') as f:
            json_polygon = json.load(f)
        coordinates = json_polygon["features"][0]["geometry"]["coordinates"][0]
        polygon = Polygon(coordinates)
        polygon_str = "POLYGON((" + ",".join(f"{x}+{y}" for x, y in polygon.exterior.coords) + "))"
        polygon_str = polygon_str.replace(" ", "+")
        with rasterio.open(self.rast_path) as src:
            raster = src.read(1)
            transform = src.meta['transform']
        
        # Load nonlinear/linear labeled data
        label_df = pd.read_json(self.label_path)

        # Load siteID coordinates from GeoNET API
        base_url = 'https://fits.geonet.org.nz/'
        endpoint = 'site'
        typeID = 'e'
        url = base_url + endpoint + '?typeID=' + typeID + '&within=' + polygon_str
        sites = requests.get(url)
        # Handle AI failures
        if sites.status_code != 200:
            raise ConnectionError(f"Failed to fetch data from GeoNET API. Status code: {sites.status_code}")
        json_data = sites.json()['features']
        
        def geo_to_raster_idx(lat, lon, transform):
            col = int((lon - transform.c) / transform.a)
            row = int((lat - transform.f) / transform.e)
            return row, col

        heights = []
        depths = []
        siteIDs = []
        geometries = []
        x_coords = []
        y_coords = []
        labels = []
        marker_colors  = []
        marker_size = []
        marker_symbol = []

        # fetch the labels and features for the sites
        for _, val in enumerate(json_data):
            geometry = val['geometry']
            properties = val['properties']
            siteID = properties['siteID']
            nonlinear = label_df[label_df['siteID'] == siteID]['nonlinear']
            if nonlinear.empty:
                continue
            else:
                label = nonlinear.values[0]
                lat_coord = geometry['coordinates'][1]
                lon_coord = geometry['coordinates'][0]
                row, col = geo_to_raster_idx(lat_coord, lon_coord, transform)
                siteID = properties['siteID']
                height = properties['height']
                depth = raster[row, col]*1e3 # Convert to meters
                siteIDs.append(siteID)
                depths.append(depth)
                geometries.append(Point(lon_coord, lat_coord,depth))
                x_coords.append(lon_coord)
                y_coords.append(lat_coord)
                heights.append(height)
                labels.append(label)

                # Markers for the map
                if label == 1:
                    marker_colors.append("red")
                    marker_size.append("small")
                    marker_symbol.append('square')
                if label == 0:
                    marker_colors.append("black")
                    marker_size.append("small")
                    marker_symbol.append('circle')
                if label == -1:
                    marker_colors.append("grey")
                    marker_size.append("small")
                    marker_symbol.append('circle')
        # save location data in a .geojson file
        self.gdf_location = gpd.GeoDataFrame({   'siteID': siteIDs, 
                                            'd': depths, 
                                            'h': heights, 
                                            'x': x_coords,
                                            'y': y_coords,
                                            'label': labels,
                                            'marker-color': marker_colors,
                                            'marker-size': marker_size,
                                            'marker-symbol': marker_symbol,
                                            'geometry': geometries}, crs="EPSG:4326")
        self.gdf_location.to_file(self.site_path, driver="GeoJSON")
        self.gdf_location = self.gdf_location.to_crs("EPSG:2193") # Convert to NZTM
        
        # Load data from GeoNet API
        endpoint = 'observation'
        df = pd.DataFrame()
        common_timestamps = pd.date_range(self.start_date, self.end_date, freq='D')
        for siteID in tqdm(siteIDs):
            df_east = pd.read_csv(base_url + endpoint + '?typeID=e&siteID=' + siteID, parse_dates=True, index_col='date-time')
            df_east = df_east.reindex(common_timestamps).rename(columns={' e (mm)': 'e', ' error (mm)': 'e_err'})
            df_north = pd.read_csv(base_url + endpoint + '?typeID=n&siteID=' + siteID, parse_dates=True, index_col='date-time')
            df_north = df_north.reindex(common_timestamps).rename(columns={' n (mm)': 'n', ' error (mm)': 'n_err'})
            df_up = pd.read_csv(base_url + endpoint + '?typeID=u&siteID=' + siteID, parse_dates=True, index_col='date-time')
            df_up = df_up.reindex(common_timestamps).rename(columns={' u (mm)': 'u', ' error (mm)': 'u_err'})
            df_merged = pd.merge(pd.merge(df_east, df_north, left_index=True, right_index=True), df_up, left_index=True, right_index=True)
            df_merged['siteID'] = siteID
            df = pd.concat([df, df_merged])

        df.reset_index(inplace=True)
        df = df.rename(columns={'index': 't'})
        self.df = df[['siteID', 't', 'e', 'n', 'u']]
        self.df = pd.merge(self.df, self.gdf_location[['siteID','d','h','x','y','label','geometry']], on='siteID', how='left') # Merge with location data
        self.df[['e', 'n', 'u']] = self.df[['e', 'n', 'u']].ffill().bfill() # Fill missing values with backward fill and forward fill
        # normalize the data
        self.df['e_n'] = (self.df.e - self.df.e.mean())/self.df.e.std() 
        self.df['n_n'] = (self.df.n - self.df.n.mean())/self.df.n.std()
        self.df['u_n'] = (self.df.u - self.df.u.mean())/self.df.u.std()
        self.df['d_n'] = (self.df.d - self.df.d.mean())/self.df.d.std()
        self.df['h_n'] = (self.df.h - self.df.h.mean())/self.df.h.std()
        self.df['x_n'] = (self.df.x-self.df.x.min())/(self.df.x.max()-self.df.x.min())
        self.df['y_n'] = (self.df.y-self.df.y.min())/(self.df.y.max()-self.df.y.min())
        # save the data in a .geojson file
        self.df = gpd.GeoDataFrame(self.df, geometry='geometry', crs="EPSG:4326") # WGS 84
        self.df.to_file(self.data_path, driver="GeoJSON") # Save to file

    def load_data(self):
        import geopandas as gpd
        self.df = gpd.read_file(self.data_path,driver='GeoJSON')
        self.gdf_location = gpd.read_file(self.site_path, driver="GeoJSON").to_crs("EPSG:2193")

    def get_graph(self, k=None, r= None, normalize=False, lag=None):
        
        self.locations = torch.stack([torch.tensor(self.gdf_location.geometry.x.values),torch.tensor(self.gdf_location.geometry.y.values)]).T
        if normalize:
            self.locations = (self.locations - self.locations.mean(dim=0))/self.locations.std(dim=0)
        self.siteIDs = list(self.gdf_location.siteID.values)       

        # Part 1: create the graph
        points = Data(pos=self.locations)
        if k is not None:
            transform = T.KNNGraph(k=k,force_undirected=True,loop=False)
        elif r is not None:
            transform = T.RadiusGraph(r=r,loop=False)

        graph = transform(points)

        # Part 2: attach the features and targets
        features = []
        targets = []
        positions = []
        
        if self.task == 'forecasting':
            pre_features = []
            pre_positions = []
            for _, group in self.df.groupby('t'):
                if normalize:
                    pre_features.append(group.e_n.values)
                if not normalize:
                    pre_features.append(group.e.values)
            for t in range(lag, len(pre_features)):
                x = np.stack(pre_features[t-lag:t]).T
                y = pre_features[t]
                features.append(x)
                targets.append(y)
                positions.append(self.locations)

        for t, group in self.df.groupby('t'):  # loop through the data by timestamp
            if self.task == 'regression':
                group.reset_index(drop=True, inplace=True)
                if normalize:
                    features.append(group[['e_n', 'n_n', 'u_n']].values)
                    positions.append(self.locations)
                    targets.append(group.d_n.values)
                if not normalize:
                    features.append(group[['e', 'n', 'u']].values)
                    positions.append(self.locations)
                    targets.append(group.d.values)
            if self.task == 'classification':
                group.reset_index(drop=True, inplace=True)
                if not normalize:
                    features.append(group[['e', 'n', 'u']].values)
                    positions.append(self.locations)
                    targets.append(group.label.values)   
                if normalize:
                    features.append(group[['e_n', 'n_n', 'u_n']].values)
                    positions.append(self.locations)
                    targets.append(group.label.values)

        return StaticGraphTemporalSignal(
            edge_index=graph.edge_index,
            edge_weight=graph.edge_attr,
            features=features,
            targets=targets,
            positions=positions
        )