import os
import pandas as pd
import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import osmnx as ox
import rasterio
from rasterio.merge import merge
from rasterio.mask import mask
from shapely.geometry import box, LineString

import t4c22
from t4c22.t4c22_config import load_basedir, load_road_graph

root = load_basedir(fn="t4c22_config.json", pkg=t4c22)
city = 'madrid'


class DataEnricher():

    buffer_m = 100

    crs_reprj = {'london': 27700,
                 'madrid': 2062,
                 'melbourne': 3110
                 }

    land_cover_reclassification = {
        'greeneries': [2, 3, 4, 8, 9, 10, 11],
        'water_bodies': [15, 18],
        'built_up': [17],
        'other': [0, 1, 7, 14, 16]
    }

    population_file_schemas = {'london': {'pop_column': 'Population',
                                          'lon_column': 'Lon',
                                          'lat_column': 'Lat'
                                          },
                               'madrid': {'pop_column': 'esp_general_2020',
                                          'lon_column': 'longitude',
                                          'lat_column': 'latitude'
                                          },
                               'melbourne': {'pop_column': 'population_2020',
                                             'lon_column': 'longitude',
                                             'lat_column': 'latitude'
                                             }
                               }
    pois = {
        'leisure': {
            'amenity': ['theatre', 'nightclub', 'cinema', 'swimming_pool'],
            'leisure': ['park', 'playground', 'sports_centre', 'stadium']
        },
        "traffic": {
            "highway": ['traffic_signals', 'crossing', 'stop']
        },
        'tourism': {
            'tourism': ['attraction', 'museum', 'artwork', 'picnic_site', 'viewpoint', 'zoo', 'theme_park'],
            'historic': ['monument', 'memorial', 'castle']
        },
        "catering": {
            'amenity': ['restaurant', 'fast_food', 'cafe', 'pub', 'bar', 'food_court', 'biergarten'],
        },
        'transport': {
            'railway': ['station', 'halt', 'tram_stop'],
            'highway': ['bus_stop'],
            'amenity': ['taxi', 'airport', 'ferry_terminal']
        },
        "shopping": {
            'shop': True
        }
    }

    def __init__(self, root, city, skip_supersegments: bool = True) -> None:
        self.root = root
        self.city = city
        self.skip_supersegments = skip_supersegments

        self.city_epsg = self.crs_reprj[self.city]
        self.population_file_schema = self.population_file_schemas[self.city]

    def prepare_inputs_for_processing_stages(self):

        df_edges, df_nodes, df_supersegments = load_road_graph(
            self.root, self.city, skip_supersegments=self.skip_supersegments)

        df_edges['num_lanes'] = 1

        lanes_num_series = df_edges['lanes'].str.extractall(
            '(\d+)').unstack().fillna('0').astype(int).apply(lambda x: np.sum(x), axis=1)

        df_edges.loc[lanes_num_series.index, ['num_lanes']] = lanes_num_series

        df_edges.drop(columns=['lanes', 'tunnel'], inplace=True)

        highway_list_index = df_edges['highway'].str.extractall(
            r"(\[)").index.get_level_values(0)
        df_edges.loc[highway_list_index, 'highway'] = df_edges.loc[highway_list_index, 'highway'].apply(
            lambda x: eval(x)[0])

        self.df_edges = df_edges

        df_edges = df_edges.merge(df_nodes[['node_id', 'x', 'y']],
                                  left_on='u',
                                  right_on='node_id'
                                  )
        df_edges = df_edges.merge(df_nodes[['node_id', 'x', 'y']],
                                  left_on='v',
                                  right_on='node_id',
                                  suffixes=('_start', '_end')
                                  )
        df_edges['geometry'] = df_edges.apply(lambda row:
                                              LineString([[row['x_start'], row['y_start']], [row['x_end'], row['y_end']]]), axis=1
                                              )
        gdf_edges = gpd.GeoDataFrame(
            df_edges.drop(columns=['node_id_start', 'x_start', 'y_start',
                          'node_id_end', 'x_end', 'y_end']),
            crs=4326,
            geometry='geometry'
        ).to_crs(self.city_epsg)

        gdf_edges['geometry'] = gdf_edges['geometry'].buffer(
            self.buffer_m)
        gdf_edges['area_km2'] = np.round(
            gdf_edges.geometry.area/1000000, 2)
        self.df_edges['area_km2'] = gdf_edges['area_km2']
        self.gdf_edges = gdf_edges.to_crs(4326)

        self.city_bounds = self.gdf_edges.total_bounds
        self.city_bounding_box = box(*self.city_bounds)

    def population_enrichment(self):

        population_file_path = f'{self.root}/preprocessing_data/population/{self.city}/population.csv'
        population = pd.read_csv(population_file_path)
        population = population[population[self.population_file_schema['lat_column']].between(self.city_bounds[1], self.city_bounds[3]) &
                                population[self.population_file_schema['lon_column']].between(
                                    self.city_bounds[0], self.city_bounds[2])
                                ].reset_index(drop=True)

        gdf_population = gpd.GeoDataFrame(population,
                                          geometry=gpd.points_from_xy(
                                              population[self.population_file_schema['lon_column']], population[self.population_file_schema['lat_column']]),
                                          crs=4326
                                          )
        sjoin = gpd.sjoin(self.gdf_edges,
                          gdf_population, predicate='contains')
        mean_population = sjoin.groupby(
            ['u', 'v'])[self.population_file_schema['pop_column']].mean().reset_index(name='mean_pop')
        self.df_edges = self.df_edges.merge(
            mean_population, on=['u', 'v'], how='left')
        self.df_edges['mean_pop'].fillna(0, inplace=True)

    def pois_enrichment(self):

        cache_dir = f'{self.root}/preprocessing_data/pois/{self.city}/.cache'
        ox.settings.cache_folder = cache_dir

        pois_gdf = None

        for pois_class, tags in self.pois.items():
            gdf = ox.features_from_polygon(self.city_bounding_box, tags=tags)
            gdf['geometry'] = gdf.to_crs(self.city_epsg).centroid.to_crs(4326)
            gdf['pois_class'] = pois_class
            gdf = gdf[['pois_class', 'geometry']]

            if pois_gdf is None:
                pois_gdf = gdf
            else:
                pois_gdf = pd.concat(
                    [pois_gdf, gdf], axis=0, ignore_index=True)

        print(pois_gdf['pois_class'].value_counts())
        sjoin = gpd.sjoin(self.gdf_edges, pois_gdf, predicate='contains')
        sum_pois = sjoin.groupby(
            ['u', 'v', 'pois_class']).size().reset_index(name='counts')
        sum_pois = sum_pois.pivot(index=['u', 'v'], columns=[
                                  'pois_class'], values="counts").fillna(0)
        sum_pois['pois_total'] = sum_pois.sum(axis=1)
        sum_pois.reset_index(inplace=True)
        self.df_edges = self.df_edges.merge(
            sum_pois, on=['u', 'v'], how='left')
        self.df_edges[list(self.pois.keys())+['pois_total']] = self.df_edges[list(
            self.pois.keys())+['pois_total']].div(self.df_edges['area_km2'], axis=0)

    def land_cover_enrichment(self):
        land_cover_city_dir = f'{self.root}/preprocessing_data/land_cover/{self.city}'

        tif_files = os.listdir(f'{land_cover_city_dir}/raw')
        datasets = []
        if len(tif_files) != 1:
            for tif_file in tif_files:
                data = rasterio.open(os.path.join(
                    f'{land_cover_city_dir}/raw', tif_file), 'r')
                datasets.append(data)
            land_cover_np_arr, out_transform = merge(datasets)
            out_meta = data.meta.copy()
            out_meta.update({
                'driver': 'GTiff',
                'height': land_cover_np_arr.shape[1],
                'width': land_cover_np_arr.shape[2],
                'transform': out_transform
            })
            merged_output_path = f'{land_cover_city_dir}/processed/merged.tif'
            with rasterio.open(merged_output_path, 'w', **out_meta) as dest:
                dest.write(land_cover_np_arr)

            land_cover_tif = rasterio.open(merged_output_path, 'r')

        else:
            land_cover_tif = rasterio.open(os.path.join(
                f'{land_cover_city_dir}/raw', tif_files[0]), 'r')

        land_cover_masked, out_transform = mask(dataset=land_cover_tif,
                                                shapes=[
                                                    self.city_bounding_box],
                                                crop=True
                                                )

        out_meta = land_cover_tif.meta.copy()
        out_meta.update({
            'driver': 'GTiff',
            'height': land_cover_masked.shape[1],
            'width': land_cover_masked.shape[2],
            'transform': out_transform
        })

        masked_output_path = f'{land_cover_city_dir}/processed/masked.tif'
        with rasterio.open(masked_output_path, 'w', **out_meta) as dest:
            dest.write(land_cover_masked)

        land_cover_masked_tif = rasterio.open(masked_output_path, 'r')

        land_cover_edges = np.zeros((self.gdf_edges.shape[0],
                                     19
                                     ))
        for i in range(self.gdf_edges.shape[0]):
            edge = self.gdf_edges.iloc[i]['geometry']
            out_arr, out_transform = mask(dataset=land_cover_masked_tif,
                                          shapes=[edge],
                                          crop=True
                                          )
            values, counts = np.unique(out_arr-1, return_counts=True)
            land_cover_edges[i, values[:-1]] = counts[:-1]

        land_cover_edges_reclassified = np.zeros((
            self.gdf_edges.shape[0],
            len(self.land_cover_reclassification.keys())
        ))

        for i, (class_name, class_idxs) in enumerate(self.land_cover_reclassification.items()):
            land_cover_edges_reclassified[:, i] = np.sum(
                land_cover_edges[:, class_idxs], axis=1)

        percentages = np.round(land_cover_edges_reclassified /
                               land_cover_edges_reclassified.sum(axis=1, keepdims=True) * 100, 0)
        percentages = percentages.astype(np.uint8)

        self.df_edges[list(
            self.land_cover_reclassification.keys())] = percentages

    def run(self):

        self.prepare_inputs_for_processing_stages()
        self.population_enrichment()
        self.pois_enrichment()
        self.land_cover_enrichment()
        object_columns = ['importance', 'highway',
                          'oneway']
        non_object_columns = list(
            set(list(self.df_edges.columns)) - set(object_columns))
        self.df_edges = self.df_edges[non_object_columns+object_columns]
        table = pa.Table.from_pandas(self.df_edges)
        pq.write_table(
            table, f'{self.root}/road_graph/{self.city}/road_edges_enriched.parquet')


DE = DataEnricher(root, city)
DE.run()
