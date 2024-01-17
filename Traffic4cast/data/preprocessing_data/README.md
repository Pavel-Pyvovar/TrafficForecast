The preprocessing data structure 
```
/preprocessing_data
├── land_cover
│   │
│   ├── london
│   │   ├── processed (to store intermdeiate tifs - masked/merged)
│   │   └── raw (to store raw tif files derived from GLAD Lab website)
│   │
│   ├── madrid
│   │   ├── processed (to store intermdeiate tifs - masked)
│   │   └── raw (to store raw tif files derived from GLAD Lab website)
│   │
│   └── melbourne
│       ├── processed (to store intermdeiate tifs - masked)
│       └── raw (to store raw tif files derived from GLAD Lab website)
│
├── pois (for this preprocessing stage no input needed, but for the sake of efficiency the  folders created to cache derived data from OSM, so in the next iteration you can reuse it)
│   │
│   ├── london 
│   │
│   ├── madrid
│   │
│   └── melbourne
│
└── population (to store csv files derived from Meta population dataset stored in HXD, files should be named population.csv)
    ├── london
    │   └── population.csv
    │
    ├── madrid
    │   └── population.csv
    │
    └── melbourne
        └── population.csv
```
URL for Land Cover datasets: https://glad.umd.edu/dataset/global-land-cover-land-use-v1
URL for Population datasets: https://data.humdata.org/organization/meta?q=population+density&sort=if%28gt%28last_modified%2Creview_date%29%2Clast_modified%2Creview_date