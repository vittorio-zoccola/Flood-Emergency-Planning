# Flood-Emergency-Planning
# Isle of Wight Evacuation Plan - University College London (UCL) project implementing a Python application, main.py, crafted for swift evacuation guidance on the Isle of Wight. Identify highest point within 5km, optimize route, and visualize on a 20km x 20km map. Contact: Vittorio Zoccola - vittoriozoccola@gmail.com

With an impending threat of extreme flooding on the Isle of Wight, residents are urged to evacuate on foot to the nearest high ground. This project aims to assist the emergency response authority by developing a Python application, main.py, to quickly advise individuals on the fastest route to the highest point of land within a 5km radius.

To accomplish this task, a Python program must be created, adhering to specific tasks and guidelines. The application involves user input, identification of the highest point, determination of the nearest Integrated Transport Network (ITN) nodes, calculation of the shortest path using Naismith's rule, and the plotting of a comprehensive map.

    User Input: Obtain the user's British National Grid coordinates and ensure they are within a specified bounding box.
    Highest Point Identification: Identify the highest point within a 5km radius from the user's location, utilizing elevation data.
    Nearest ITN Nodes: Identify the nearest ITN nodes to the user and the highest point.
    Shortest Path: Determine the shortest route using Naismith's rule, accounting for terrain elevation.
    Map Plotting: Generate a 20km x 20km background map, overlaying elevation data, and plotting the user's starting point, highest point, and shortest route.

Data for the project includes shape files defining the island and roads, a JSON file defining the ITN graph, and raster files for elevation and background mapping. Packages permitted NumPy, Pandas, Pyproj, Shapely, Geopandas, Rasterio, RTREE, NetworkX.

**  Link to Download the "Material" Folder **
[Link to My File on Dropbox](https://www.dropbox.com/scl/fo/5hlwdz75w2yiebfl56e5t/h?rlkey=nwej8joz0qhh76zdg7xjvrxyf&dl=0)

