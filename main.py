import geopandas as gpd
import json
import matplotlib.pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib.cm import ScalarMappable
import networkx as nx
import numpy as np
import pyproj
from rasterio.features import geometry_mask
import rasterio
import rasterio.mask as rtm
import rasterio.plot as rtp
from rasterio.windows import Window
from rtree import index
import shapely.geometry as sgeo
from shapely.geometry import Point

def get_elevation(coords, raster, transform):
    """
    Retrieves the elevation from a raster at the specified coordinates.

    Args:
        coords (tuple): Coordinates (x, y) for which elevation is needed.
        raster (numpy.ndarray): Elevation raster data.
        transform (rasterio.transform.Affine): Affine transformation for raster coordinates.

    Returns:
        float: Elevation value at the specified coordinates.
    """
    x, y = coords
    x0 = transform.c
    y0 = transform.f
    resX = transform.a
    resY = transform.e
    col = int((x - x0) / resX)
    row = int((y - y0) / resY)

    return raster[0, row, col]


class Node:
    """
    Represents a node in the integrated transport network.

    Attributes:
        id (str): Node identifier.
        coords (tuple): Coordinates (x, y) of the node.
    """

    def __init__(self, id, coords) -> None:
        self.id = id
        self.coords = coords


class RoadLink:
    """
    Represents a road link in the integrated transport network.

    Attributes:
        id (str): Road link identifier.
        length (float): Length of the road link.
        coords (list): List of coordinates defining the road link.
        start (Node): Starting node of the road link.
        end (Node): Ending node of the road link.
        natureOfRoad (str): Nature of the road.
        descriptiveTerm (str): Descriptive term for the road.

    Methods:
        road_coords2series: Converts road coordinates to x and y series for plotting.
    """

    def __init__(self, id, length, coords, start, end, natureOfRoad, descriptiveTerm) -> None:
        self.id = id
        self.length = length
        self.coords = coords
        self.start = start
        self.end = end
        self.natureOfRoad = natureOfRoad
        self.descriptiveTerm = descriptiveTerm
        self.xSeries = []
        self.ySeries = []

    def road_coords2series(self):
        """
        Converts road coordinates to x and y series for plotting.

        Returns:
            int: A placeholder value (0).
        """
        xSeries = []
        ySeries = []
        for coord in self.coords:
            xSeries.append(coord[0])
            ySeries.append(coord[1])
        self.xSeries, self.ySeries = xSeries, ySeries
        return 0


class FloodEmergencyManager:
    """
    Manages emergency response operations on an island.

    Attributes:
        shapefile_path (str): Path to the shapefile for the island.
        elevation_raster_path (str): Path to the elevation raster data.
        itn_file (str): Path to the integrated transport network data file.
        background_raster_path (str): Path to the background raster data.

        island_shape (geopandas.geodataframe.GeoDataFrame): GeoDataFrame representing the island shape.
        bng_proj (pyproj.Transformer): Coordinate transformation from EPSG:4326 to EPSG:27700.
        user_easting (float): Easting coordinate of the user.
        user_northing (float): Northing coordinate of the user.
        highest_point_location (tuple): Coordinates of the highest point on the island.
        nearest_node_user (Node): Nearest integrated transport network node to the user.
        nearest_node_highest_point (Node): Nearest integrated transport network node to the highest point.

        itn_data (dict): Integrated Transport Network data.
        itn_gdf (geopandas.geodataframe.GeoDataFrame): GeoDataFrame for road links.
        itn_rtree_index (rtree.index.Index): R-tree index for spatial queries.

        nodes (list): List of Node objects.
        roads (list): List of RoadLink objects.
        dict_nodes (dict): Dictionary mapping node IDs to Node objects.
        dict_roads (dict): Dictionary mapping road link IDs to RoadLink objects.
        shortest_path (list): List of RoadLink objects representing the shortest path.

    Methods:
        load_itn_data: Load integrated transport network data from a file.
        create_gdf_from_itn_data: Create GeoDataFrame from ITN data.
        create_rtree_index: Create R-tree index for spatial queries.
        find_nearest_itn_node: Find the nearest integrated transport network node to a given location.
        load_itn_data_t4: Load ITN data for additional tasks (Task 4).
    """

    def __init__(self, shapefile_path, elevation_raster_path, itn_file, background_raster_path):
        # Initialization of the FloodEmergencyManager class
        # Loading geographic data and initializing variables

        # Load shapefile for the island
        self.time_taken = None
        self.shapefile_path = shapefile_path
        self.elevation_raster_path = elevation_raster_path
        self.itn_file = itn_file
        self.background_raster_path = background_raster_path
        self.island_shape = gpd.read_file(shapefile_path)  # Reading the shapefile using GeoPandas
        self.bng_proj = pyproj.Transformer.from_proj("EPSG:4326", "EPSG:27700")  # Coordinate transformation
        self.user_easting = None
        self.user_northing = None
        self.highest_point_location = None
        self.nearest_node_user = None
        self.nearest_node_highest_point = None  # Added attribute for highest point location

        # Loading ITN data and creating GeoDataFrame and R-tree index
        self.itn_data = self.load_itn_data()
        self.itn_gdf = self.create_gdf_from_itn_data()
        self.itn_rtree_index = self.create_rtree_index()

        # Additional attributes for Task 4 and Task 5
        self.nodes = []
        self.roads = []
        self.dict_nodes = {}
        self.dict_roads = {}
        self.shortest_path = []
        self.ascending_section = []
        self.load_itn_data_t4()

    # Task 1 / 6: User Input
    def get_user_input(self):
        """
        Get user input for easting and northing coordinates within the British National Grid.
        Ask the user if they want to extend the region.

        Raises:
            ValueError: If the entered coordinates are outside the allowed application area or island shape.

        Returns:
            None
        """
        while True:
            try:
                print("Welcome to the Floor Emergency Plan")
                extend_region = input(
                    "Would you like to enter the coordinates within an extended region? (yes/no): ").lower()

                if extend_region == "yes":
                    self.min_easting, self.min_northing = 425000, 75000
                    self.max_easting, self.max_northing = 470000, 100000
                elif extend_region == "no":
                    self.min_easting, self.min_northing = 430000, 80000
                    self.max_easting, self.max_northing = 465000, 95000
                else:
                    raise ValueError("Invalid input. Please enter 'yes' or 'no'.")

                self.user_easting = float(input("Enter the easting coordinate (British National Grid): "))
                self.user_northing = float(input("Enter the northing coordinate (British National Grid): "))

                if not self.is_within_application_area(self.user_easting, self.user_northing):
                    raise ValueError("Coordinates are outside the application area. Exiting the application.")

                user_point = Point(self.user_easting, self.user_northing)

                # Check if the user point is within the island shape
                if not user_point.within(self.island_shape.unary_union):
                    raise ValueError("The entered point is not within the island shape. Please try again.")

                print(f"User coordinates: Easting {self.user_easting}, Northing {self.user_northing}")
                break  # Break out of the loop if valid input
            except ValueError as e:
                print(f"Error: {e}")
                print("Enter valid coordinates.")

    def is_within_application_area(self, easting, northing):
        """
        Check if the given coordinates are within the allowed application area.

        Args:
            easting (float): Easting coordinate.
            northing (float): Northing coordinate.

        Returns:
            bool: True if coordinates are within the application area, False otherwise.
        """
        min_easting, min_northing = self.min_easting, self.min_northing
        max_easting, max_northing = self.max_easting, self.max_northing

        if (
                min_easting <= easting <= max_easting
                and min_northing <= northing <= max_northing
                and self.is_within_allowed_box(easting, northing)
        ):
            return True
        else:
            print("Coordinates are outside the application area. Exiting the application.")
            return False

    def is_within_allowed_box(self, easting, northing):
        """
        Check if the given coordinates are within the allowed box.

        Args:
            easting (float): Easting coordinate.
            northing (float): Northing coordinate.

        Returns:
            bool: True if coordinates are within the allowed box, False otherwise.
        """
        min_easting, min_northing = self.min_easting, self.min_northing
        max_easting, max_northing = self.max_easting, self.max_northing
        return min_easting <= easting <= max_easting and min_northing <= northing <= max_northing

    # Task 2: Highest Point Identification
    def find_highest_point(self, user_easting, user_northing):
        """
        Find the highest point within a circular area of 5 km radius around the user's coordinates.

        Args:
            user_easting (float): Easting coordinate of the user.
            user_northing (float): Northing coordinate of the user.

        Returns:
            None
        """
        with rasterio.open('Material/elevation/SZ.asc') as src:
            user_point = Point(user_easting, user_northing)
            radius = 5000  # Radius in meters

            # Create a circular buffer around the user's point
            user_buffer = user_point.buffer(radius)

            # Mask the raster with the buffer geometry
            mask = geometry_mask([user_buffer], out_shape=(src.height, src.width), transform=src.transform,
                                 invert=False)  # invert=False to get the mask inside the buffer
            masked_data = np.ma.masked_array(src.read(1), mask)

            # Find the position of the highest point within the masked array
            highest_point_index = np.unravel_index(np.argmax(masked_data, axis=None), masked_data.shape)

            # Calculate the geographic coordinates of the highest point
            highest_point_coords = src.xy(highest_point_index[0], highest_point_index[1])

            print(f"Coordinates of the highest point within the 5 km circle: {highest_point_coords}")

            # Perform additional operations if necessary
            self.print_highest_point_info(user_point, highest_point_coords)

            # Find the nearest ITN node to the user's coordinates
            nearest_node_user = self.find_nearest_itn_node((user_easting, user_northing))
            print("Nearest ITN node to user:", nearest_node_user["geometry"].coords[0])

            # Find the nearest ITN node to the highest point
            nearest_node_highest_point = self.find_nearest_itn_node(highest_point_coords)
            print("Nearest ITN node to highest point:", nearest_node_highest_point["geometry"].coords[0])

            # Set the highest point location attribute
            self.highest_point_location = highest_point_coords
            self.nearest_node_user = nearest_node_user
            self.nearest_node_highest_point = nearest_node_highest_point

    def calculate_circular_window(self, src, user_easting, user_northing, radius):
        """
        Calculate the circular window indices based on the user's coordinates and radius.

        Args:
            src (rasterio.DatasetReader): Raster dataset.
            user_easting (float): Easting coordinate of the user.
            user_northing (float): Northing coordinate of the user.
            radius (float): Radius of the circular window in meters.

        Returns:
            Tuple: Window, window_col, and window_row.
        """
        pixel_size = abs(src.transform[0])  # Pixel size

        # Calculate circular window indices
        col, row = src.index(user_easting, user_northing)
        window_col = int(col - radius // pixel_size)
        window_row = int(row - radius // pixel_size)
        window_width = int(2 * radius // pixel_size)
        window_height = int(2 * radius // pixel_size)

        # Ensure the window dimensions are non-negative
        window_col = max(0, window_col)
        window_row = max(0, window_row)
        window_width = max(1, window_width)  # Ensure a minimum width of 1
        window_height = max(1, window_height)  # Ensure a minimum height of 1

        # Limit the window to the image dimensions
        window_col = min(window_col, src.width - window_width)
        window_row = min(window_row, src.height - window_height)

        # Create circular window
        window = Window(window_col, window_row, window_width, window_height)

        return window, window_col, window_row

    def print_highest_point_info(self, user_point, highest_point_coords):
        """
        Print information about the highest point and its relationship with the user's circular area.

        Args:
            user_point (shapely.geometry.Point): User's point.
            highest_point_coords (tuple): Coordinates of the highest point.

        Returns:
            None
        """
        # Check if the coordinates of the highest point are within the user's 5 km circle
        distance_to_user = user_point.distance(Point(highest_point_coords[0], highest_point_coords[1]))

        print(f"Distance of the Highest Point from the center of the user's circle: {distance_to_user} meters")

        if distance_to_user <= 5000:
            print(
                "The point with the highest elevation is within the 5 km circle centered on the "
                "user's coordinates.")
        else:
            print(
                "The point with the highest coordinates and elevation is outside the 5 km circle centered on the "
                "user's coordinates.")

 #NB: I wrote most of the code from scratch, but we have used ChatGPT (Open AI, https://openai.com/) as a generative AI tool to effectively structure certain code segment related to Task 2

    # Task 3: Nearest Integrated Transport Network
    def load_itn_data(self):
        """
        Load Integrated Transport Network (ITN) data from a JSON file.

        Returns:
            dict: Loaded ITN data.
        """
        try:
            with open(self.itn_file, 'r') as file:
                itn_data = json.load(file)
            return itn_data
        except Exception as e:
            print(f"Error loading ITN data: {e}")
            return None

    def create_gdf_from_itn_data(self):
        """
        Create a GeoDataFrame from ITN road link data.

        Returns:
            GeoDataFrame: GeoDataFrame containing road link features.
        """
        features = []
        for link_id, link_info in self.itn_data.get("roadnodes", {}).items():
            coords = link_info.get("coords", [])
            if len(coords) >= 2:
                features.append({
                    'id': link_id,
                    'geometry': Point(coords)
                })

        itn_gdf = gpd.GeoDataFrame(features, crs='EPSG:27700')
        return itn_gdf

    def create_rtree_index(self):
        """
        Create an R-tree index for efficient spatial queries on ITN road link geometries.

        Returns:
            index.Index: R-tree index.
        """
        idx = index.Index()
        for i, geom in enumerate(self.itn_gdf.geometry):
            idx.insert(i, geom.bounds)
        return idx

    def find_nearest_itn_node(self, location):
        """
        Find the nearest ITN node to a given location.

        Args:
            location (tuple): Coordinates (e.g., (easting, northing)).

        Returns:
            GeoSeries: GeoSeries representing the nearest ITN node.
        """
        point = Point(location)
        nearest_idx = list(self.itn_rtree_index.nearest(point.bounds, 1))
        nearest_node = self.itn_gdf.iloc[nearest_idx[0]]
        return nearest_node

# NB: I wrote most of the code from scratch, but we have used ChatGPT (Open AI, https://openai.com/) as a generative AI tool to effectively structure certain code segment related to Task 3

    # Task 4: Shortest Path
    def load_itn_data_t4(self):
        """
        Load ITN data for Task 4, including road nodes and links, and create corresponding Node and RoadLink objects.
        """
        with open(self.itn_file, 'r') as file:
            data = json.load(file)

        for id, values in data['roadnodes'].items():
            node_ = Node(id, values['coords'])
            self.nodes.append(node_)
            self.dict_nodes[id] = node_

        for id, values in data['roadlinks'].items():
            length = values['length']
            coords = values['coords']
            start = self.dict_nodes[values['start']]
            end = self.dict_nodes[values['end']]
            natureOfRoad = values['natureOfRoad']
            descriptiveTerm = values['descriptiveTerm']
            road_ = RoadLink(id, length, coords, start, end, natureOfRoad, descriptiveTerm)
            road_.road_coords2series()
            self.roads.append(road_)
            self.dict_roads[id] = road_

    def calculate_shortest_path(self):
        """
        Calculate the shortest path using networkx library, considering elevation and road length.
        """

        # Open elevation raster file and read image data
        global i
        elevation_raster = rasterio.open(self.elevation_raster_path)
        elevation_image = np.array(elevation_raster.read())

        # Create an empty graph using networkx
        G = nx.Graph()

        # Creating a dictionary to store node elevations
        node_elevations = {}

        # Updating the graph with node elevations
        for node in self.nodes:
            x, y = node.coords
            elevation = get_elevation((x, y), elevation_image, elevation_raster.transform)
            node_elevations[node.id] = elevation
            G.add_node(node.id, coords=node.coords)

        # Adding edges (roads) to the graph with weights based on distance and elevation difference
        for road in self.roads:
            walking_speed = 5000.0 / 60
            additional_time_per_meter = 1.0 / 10
            total_elevation_additional_time = 0

            for i in range(len(road.coords) - 1):
                this_elev = get_elevation(road.coords[i], elevation_image, elevation_raster.transform)
                next_elev = get_elevation(road.coords[i + 1], elevation_image, elevation_raster.transform)
                elevation_additional_time = (next_elev - this_elev) * additional_time_per_meter

                # Ensure that the additional time is non-negative
                if elevation_additional_time <= 0:
                    elevation_additional_time = 0

                total_elevation_additional_time += elevation_additional_time

            time_weight = road.length / walking_speed + total_elevation_additional_time
            G.add_edge(road.start, road.end, weight=time_weight)

        # Calculating the shortest path in the graph
        shortest_path_nodes = nx.shortest_path(
            G, source=self.dict_nodes[self.nearest_node_user.id],
            target=self.dict_nodes[self.nearest_node_highest_point.id], weight="weight")

        # Identify the corresponding RoadLink objects for the shortest path and store them
        for i in range(len(shortest_path_nodes) - 1):
            for road in self.roads:
                if (shortest_path_nodes[i].id == road.start.id and shortest_path_nodes[i + 1].id == road.end.id) \
                        or (
                        shortest_path_nodes[i].id == road.end.id and shortest_path_nodes[i + 1].id == road.start.id):
                    self.shortest_path.append(road)

        # Add ascending section
        for road in self.shortest_path:
            node = Node('', 0)
            asc_coords = []
            for i in range(len(road.coords) - 1):
                this_elev = get_elevation(road.coords[i], elevation_image, elevation_raster.transform)
                next_elev = get_elevation(road.coords[i + 1], elevation_image, elevation_raster.transform)
                elevation = next_elev - this_elev

                if elevation > 0:
                    if i == 0 or road.coords[i - 1][0] != road.coords[i][0]:
                        asc_coords.append(road.coords[i])

                    asc_coords.append(road.coords[i + 1])

            asc_road = RoadLink(f'asc road {i}', 0, asc_coords, node, node, '', '')
            asc_road.road_coords2series()
            self.ascending_section.append(asc_road)

        # Calculating the total time based on the shortest path
        total_time_minutes = sum(G[shortest_path_nodes[i]][shortest_path_nodes[i + 1]]['weight']
                                 for i in range(len(shortest_path_nodes) - 1))

        # Convert the total time from minutes to hours and minutes
        total_hours = int(total_time_minutes // 60)
        remaining_minutes = int(total_time_minutes % 60)

        # Format the output string
        formatted_time = f"{total_hours}h {remaining_minutes}m"
        print(f"Estimated time: {formatted_time}")

        # Assigns the value of the variable formatted_time to the class attribute time_taken
        self.time_taken = formatted_time

# NB: I wrote most of the code from scratch, but we have used ChatGPT (Open AI, https://openai.com/) as a generative AI tool to effectively structure certain code segment related to Task 4

    # Task 5: Map Plotting
    def plot_map_with_elements(self):
        # Set center_point to user's input coordinates
        center_point = (self.user_easting, self.user_northing)

        surrounding_distance = 10000
        buffer_radius = 5000
        user_loc = Point(center_point[0], center_point[1])

        # Create a buffered area around the user's location
        buffered_area = user_loc.buffer(buffer_radius)

        # Define bounding box coordinates based on the center point and surrounding distance
        bounds = sgeo.box(
            center_point[0] - surrounding_distance,
            center_point[1] - surrounding_distance,
            center_point[0] + surrounding_distance,
            center_point[1] + surrounding_distance
        )
        background_raster = rasterio.open(self.background_raster_path)

        # Create a window based on the specified bounds and read the raster
        gdf = gpd.GeoDataFrame(geometry=[bounds], crs='epsg:27700')

        # Spatial intersection of the clipping area with raster data
        clipped_raster, clipped_raster_transform = rtm.mask(background_raster, gdf.geometry, crop=True)

        # Get the affine transformation parameters from the raster
        transform_map = clipped_raster_transform
        data_map = np.squeeze(clipped_raster, axis=0)
        color_map = background_raster.colormap(1)
        mapped_data_map = np.zeros((data_map.shape[0], data_map.shape[1], 3), dtype=np.uint8)
        # Iterate through each value in the color map provided by the raster
        for value, rgb_color in color_map.items():
            # Create a boolean mask for pixels with the current value in the raster
            mask_map = (data_map == value)

            # Assign the RGB color values to the corresponding pixels in the mapped data array
            mapped_data_map[mask_map] = rgb_color[:3]

        mapped_data_map = mapped_data_map.transpose((2, 0, 1))

        # Read the elevation raster and clip it to the buffered area
        elevation_raster = rasterio.open(self.elevation_raster_path)
        geometry = gpd.GeoDataFrame(geometry=[buffered_area], crs='epsg:27700')
        clipped_elevation, transform_el = rtm.mask(elevation_raster, geometry.geometry, crop=True, nodata=np.nan)

        # Plotting starts here
        fig, ax = plt.subplots(figsize=(10, 10))
        # isle_of_wight.plot(ax=ax, facecolor='none', edgecolor='black')
        rtp.show(mapped_data_map, ax=ax, transform=transform_map)
        rtp.show(clipped_elevation, ax=ax, transform=transform_el, cmap='terrain', alpha=0.7)

        # Plot road
        for road in self.shortest_path:
            plt.plot(road.xSeries, road.ySeries, c='r', lw=3)

        for road in self.ascending_section:
            for i in range(int(len(road.xSeries) / 2)):
                plt.plot(road.xSeries[i * 2:(i + 1) * 2], road.ySeries[i * 2:(i + 1) * 2], c='y', lw=1)

        # Plot user point
        plt.scatter(self.user_easting, self.user_northing, c='orange')

        zero_map = np.zeros(mapped_data_map.shape)
        rtp.show(zero_map, ax=ax, transform=transform_map, alpha=0)

        # Plot the highest point marker
        plt.scatter(self.highest_point_location[0], self.highest_point_location[1], c='blue', marker='o',
                    label='Highest point')

        zero_map = np.zeros(mapped_data_map.shape)
        rtp.show(zero_map, ax=ax, transform=transform_map, alpha=0)

        # Scale Bar
        scalebar = ScaleBar(1, location='lower right')
        ax.add_artist(scalebar)

        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='User Location'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Highest Point'),
            plt.Line2D([0], [0], color='r', linewidth=2, label='Shortest Path'),
            plt.Line2D([0], [0], color='y', linewidth=2, label='Ascending Section')
        ]

        ax.legend(handles=legend_elements, loc='upper left')

        # North Arrow
        arrow_text = u'\u2191'
        ax.text(0.95, 0.94, arrow_text, fontsize=22, ha='center', va='center', transform=ax.transAxes)
        ax.text(0.95, 0.975, 'N', transform=ax.transAxes, fontsize=10, ha='center', va='center', color='black')

        # Elevation Color Bar
        vmin = np.nanmin(clipped_elevation)
        vmax = np.nanmax(clipped_elevation)
        norm = plt.Normalize(vmin, vmax)
        sm = ScalarMappable(cmap='terrain', norm=norm)
        sm.set_array([])  # An empty array is set as a workaround
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05, ticks=np.linspace(vmin, vmax, 11))
        cbar.set_label('Elevation (m)')

        # Estimated Time
        time_text = f"Estimated time: {self.time_taken}"
        plt.annotate(time_text, xy=(0.05, 0.05), xycoords='axes fraction', fontsize=10, color='black',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plt.xlabel('Easting (m)')
        plt.ylabel('Northing (m)')
        plt.title('Flood Emergency Path Planning')
        # Show the plot
        plt.show()


def main():
    emergency_response = FloodEmergencyManager(
        'Material/shape/isle_of_wight.shp',
        'Material/elevation/SZ.asc',
        'Material/itn/solent_itn.json',
        'Material/background/raster-50k_2724246.tif'
    )

    # Task 1: User Input
    emergency_response.get_user_input()

    # Task 2: Highest Point Identification
    emergency_response.find_highest_point(emergency_response.user_easting, emergency_response.user_northing)

    # Task 4: Shortest Path
    emergency_response.calculate_shortest_path()

    # Task 5: Map Plotting
    emergency_response.plot_map_with_elements()


if __name__ == "__main__":
    main()
