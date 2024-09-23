
import torch
from pygsig.graph import KernelKNNGraph,KernelRadiusGraph,StaticGraphTemporalSignal
from torch_geometric.data import Data
import torch_geometric.transforms as T
import pandas as pd
import json
import requests
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

class SubductionZone(object):
    def __init__(self,
                 poly_path='../datasets/subduction/polygon.geojson', # region of interest
                 rast_path='../datasets/subduction/interface.grd', # subduction interface depth
                 label_path='../datasets/subduction/nonlinear.json', # nonlinear/linear labels
                 site_path='../datasets/subduction/site_location.geojson',
                 data_path='../datasets/subduction/site_data.geojson',
                 start_date=pd.Timestamp('2010-01-01 11:59:00+0000', tz='UTC'),
                 end_date=pd.Timestamp('2023-12-15 11:59:00+0000', tz='UTC'),
                 redownload=True
        ):
        self.start_date = start_date
        self.end_date = end_date
        self.poly_path = poly_path
        self.label_path = label_path
        self.rast_path = rast_path
        self.site_path = site_path
        self.data_path = data_path
        if redownload:
            self._download()

    def _download(self):
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
        json_data = sites.json()['features']
        
        def geo_to_raster_idx(lat, lon, transform):
            col = int((lon - transform.c) / transform.a)
            row = int((lat - transform.f) / transform.e)
            return row, col

        subduct = []
        heights = []
        depths = []
        siteIDs = []
        geometries = []
        labels = []
        marker_colors  = []
        marker_size = []
        marker_symbol = []

        for _, val in enumerate(json_data):
            geometry = val['geometry']
            properties = val['properties']
            lat_coord = geometry['coordinates'][1]
            lon_coord = geometry['coordinates'][0]
            row, col = geo_to_raster_idx(lat_coord, lon_coord, transform)
            siteID = properties['siteID']
            height = properties['height']
            depth = raster[row, col]*1e3 # Convert to meters
            nonlinear = label_df[label_df['siteID'] == siteID]['nonlinear']
            siteIDs.append(siteID)
            depths.append(depth)
            geometries.append(Point(lon_coord, lat_coord,depth))
            heights.append(height)

            if nonlinear.empty:
                label = -1
            else:
                label = nonlinear.values[0]
            labels.append(label)
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

        # Load data from GeoNet API
        gdf_location = gpd.GeoDataFrame({   'siteID': siteIDs, 
                                            'depth': depths, 
                                            'height': heights, 
                                            'nonlinear': labels,
                                            'marker-color': marker_colors,
                                            'marker-size': marker_size,
                                            'marker-symbol': marker_symbol,
                                            'geometry': geometries}, crs="EPSG:4326")
        gdf_location.to_file(self.site_path, driver="GeoJSON")
        gdf_location = gdf_location.to_crs("EPSG:2193") # Convert to NZTM
        
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
        df_data = df[['siteID', 't', 'e', 'n', 'u']]
        gdf_data = pd.merge(df_data, gdf_location, on='siteID', how='left') # Merge with location data
        gdf_data[['e', 'n', 'u']] = gdf_data[['e', 'n', 'u']].ffill().bfill() # Fill missing values with backward fill and forward fill
        gdf_data = gpd.GeoDataFrame(gdf_data, geometry='geometry', crs="EPSG:4326") # WGS 84
        gdf_data.to_file(self.data_path, driver="GeoJSON") # Save to file

    def _from_file(self):
        import geopandas as gpd
        gdf_data = gpd.read_file(self.data_path,driver='GeoJSON')
        gdf_location = gpd.read_file(self.site_path, driver="GeoJSON")
        gdf_location = gdf_location.to_crs("EPSG:2193") # Convert to NZTM
        return gdf_data,gdf_location

    def load_data(self, task_name='nonlinear',include_time=False, k=None, r= None, bandwidth = None):
        # Checks
        if task_name not in ['depth','nonlinear']:
            raise ValueError('task_name must be either "depth" or "nonlinear"')
        
        # Part 1: load the data
        gdf_data,gdf_location = self._from_file()

        # Part 2: make the timestamps numeric
        gdf_data['t'] = (gdf_data['t'] - self.start_date)/(self.end_date - self.start_date)

        # Part 2: create the graph
        positions = torch.stack([torch.tensor(gdf_location.geometry.x.values),
                                 torch.tensor(gdf_location.geometry.y.values),
                                 ]).T
        points = Data(pos=positions)
        if (k is not None) and (bandwidth is not None):
            transform = KernelKNNGraph(k=k,bandwidth=bandwidth)
        elif (r is not None) and (bandwidth is not None):
            transform = T.Compose([KernelRadiusGraph(r=r,bandwidth=bandwidth),T.RemoveIsolatedNodes()])
        elif (k is not None) and (bandwidth is None):
            transform = T.KNNGraph(k=k,force_undirected=True,loop=False)
        elif (r is not None) and (bandwidth is None):
            transform = T.Compose([T.RadiusGraph(r=r,loop=False),T.RemoveIsolatedNodes()])

        graph = transform(points)

        # Part 3: attach the features and targets

        features = []
        targets = []
        for _,group in gdf_data.groupby('t'): # loop through the data by timestamp
            group.reset_index(drop=True,inplace=True) 
            if task_name == 'depth':
                targets.append(group.depth.values)
            if task_name == 'nonlinear':
                targets.append(group.nonlinear.values)
            if include_time:
                features.append(group[['t','e','n','u']].values)
            else:
                features.append(group[['e','n','u']].values)

        return StaticGraphTemporalSignal(
            edge_index=graph.edge_index,
            edge_weight=graph.edge_attr,
            features=features,
            targets=targets,
            positions=positions
        )