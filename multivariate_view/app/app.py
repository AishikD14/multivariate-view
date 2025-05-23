from pathlib import Path

from matplotlib import pyplot as plt
import numba
import numpy as np

import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from trame.app import get_server
from trame.assets.remote import download_file_from_google_drive
from trame.decorators import TrameApp, change, life_cycle
from trame.ui.vuetify3 import VAppLayout
from trame.widgets import client, html, plotly, vtk, vuetify3 as v
from multivariate_view.widgets import radvolviz
from .assets import ASSETS
from scipy import ndimage

from .compute import (
    compute_gbc,
    data_topology_reduction,
    gbc_to_rgb,
    rotate_coordinates,
)
from .io import load_dataset
from .volume_view import VolumeView
import time
from scipy.spatial import cKDTree
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# We will cache downloaded data examples in this directory.
EXAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / 'data'
EXAMPLE_GOOGLE_DRIVE_ID = '1nI_hzrqbGBypUU7jMbWnF7-PkqNMiwqB'
EXAMPLE_DATA_REF = 'https://doi.org/10.1038/s43246-022-00259-x'

data_dict = {
    "CeCoFeGd": {
        "data_path": (
            EXAMPLE_DATA_DIR / 'CeCoFeGd.h5'
        ),
        "segment_path": EXAMPLE_DATA_DIR / 'volume_labels_CeCoFeGd.npy',
    },
    "thigh_sarcoma": {
        "data_path": (
            EXAMPLE_DATA_DIR / 'thigh_sarcoma.h5'
        ),
        "segment_path": EXAMPLE_DATA_DIR / 'volume_labels_thigh_sarcoma.npy',
    },
}

@TrameApp()
class App:
    def __init__(self, server=None):
        self.server = get_server(server, client_type='vue3')

        # Add this line to hook into browser connect
        self.ctrl.on_client_connected.add(self._on_ready)

        # CLI
        self.server.cli.add_argument(
            "--data", help="Path to the file to load", default=None
        )
        self.server.cli.add_argument(
            "--nan", help="Replace NaN to specific value", default=0
        )
        self.server.cli.add_argument(
            "--enable-preprocessing",
            help="Enable additional control on data pre-processing",
            dest="preprocess",
            action='store_true',
            default=True,
        )
        self.server.cli.add_argument(
            "--normalize-channels",
            help="Normalize each channel to be between 0 and 1",
            action="store_true",
            default=False,
        )
        self.server.cli.add_argument(
            "--opacity-channel",
            help="Set the specified channel to be opacity only",
            default=None,
        )
        self.server.cli.add_argument(
            "--label-map",
            help="Set a path to a label map file",
            default=None,
        )
        self.server.cli.add_argument(
            "--use-supervoxel",
            help="Use supervoxel segmentation",
            dest="use_supervoxel",
            action="store_true",
            default=False,
        )

        args, _ = self.server.cli.parse_known_args()
        self.enable_preprocessing = args.preprocess
        self.nan_replacement = args.nan
        self.normalize_channels = args.normalize_channels
        self.opacity_channel = args.opacity_channel
        self.label_map_file = args.label_map
        self.label_map = None
        self.use_supervoxels = args.use_supervoxel

        if self.label_map_file is not None:
            # Load the label map
            # We assume this is the same shape as the data with
            # the padding removed.
            self.label_map = np.load(self.label_map_file)

        # Set this if you want label map names other than "0, 1, 2, ..."
        self.label_map_names = None

        self.volume_view = VolumeView()

        self.unrotated_gbc = None
        self.unrotated_components = None

        self.gbc_data = None
        self.rgb_data = None
        self.opacity_data = None

        # Set the default values
        self.num_clusters = 5
        self.cluster_method = "kmeans"
        self.use_autoencoder = False
        self.dataset = "CeCoFeGd"

        self.ui = self._build_ui()
        self.load_data()
        self.create_table()

        if self.server.hot_reload:
            self.ctrl.on_server_reload.add(self._build_ui)

    def _on_ready(self, **kwargs):
        pass
    
    def load_data(self):
        try:
            header, data = load_dataset(Path(data_dict[self.dataset]["data_path"]))

        except Exception as e:
            print(f"Error loading dataset: {e}")

            self.dataset == "CeCoFeGd"

            EXAMPLE_DATA_DIR.mkdir(parents=True, exist_ok=True)

            print(
                f'\nDefaulting to example: {data_dict[self.dataset]["data_path"].name}'
            )

            citation_str = f'* Example data citation: {EXAMPLE_DATA_REF} *'
            boundary_str = '*' * len(citation_str)
            print(f'\n{boundary_str}\n{citation_str}\n{boundary_str}\n')

            if not data_dict[self.dataset]["data_path"].exists():
                # Automatically download the example dataset, and put it in the
                # data directory.
                print(f'Downloading example dataset to: {data_dict[self.dataset]["data_path"]}')
                download_file_from_google_drive(
                    EXAMPLE_GOOGLE_DRIVE_ID, data_dict[self.dataset]["data_path"]
                )

            header, data = load_dataset(Path(data_dict[self.dataset]["data_path"]))

        if self.dataset == "thigh_sarcoma":
            header[0], header[1] = header[1], header[0]
            data[:, :, :, [0, 1]] = data[:, :, :, [1, 0]]

        print(header)
        print("Data shape on load ", data.shape)

        # Handle NaN if provided
        if self.nan_replacement is not None:
            data[np.isnan(data)] = float(self.nan_replacement)

        # Remove padding so it will render faster.
        # This removes faces that are all zeros recursively until
        # the first non-zero voxel is hit.
        # Our sample data has a *lot* of padding.
        data = _remove_padding_uniform(data)
        print("Data shape after padding removal ", data.shape)

        if self.opacity_channel is not None:
            # Extract the opacity data
            opacity_idx = header.index(self.opacity_channel)
            self.opacity_data = _normalize_data(data[:, :, :, opacity_idx])

            # Set all data less than 80% to 0, and then re-normalize
            # self.opacity_data[self.opacity_data < 0.8] = 0
            # self.opacity_data = _normalize_data(self.opacity_data**5)

            header.pop(opacity_idx)
            data = np.delete(data, opacity_idx, axis=3)

        self.state.component_labels = header

        # Remember the data shape (without the multichannel part)
        self.data_shape = data.shape[:-1]
        self.num_channels = data.shape[-1]

        self.raw_unpadded_flattened_data = data.reshape(
            np.prod(self.data_shape), self.num_channels
        )

        print(f"Number of channels: {self.num_channels}")
        print(f"Raw unpadded flattened data shape: {self.raw_unpadded_flattened_data.shape}")

        if self.normalize_channels:
            # Normalize each channel to be between 0 and 1
            for i in range(data.shape[-1]):
                data[:, :, :, i] = _normalize_data(data[:, :, :, i])
        else:
            data = _normalize_data(data)

        print("Data shape after normalization ", data.shape)

        # ------------------------------------------------------------------------------------------------

        if self.use_supervoxels:

            print("---------------------------------------------------")

            if not data_dict[self.dataset]["segment_path"].exists():
                # Generate supervoxels
                segments, _ = self.generate_supervoxel(data, True)
                
            else:
                # Load the supervoxels
                segments = np.load(data_dict[self.dataset]["segment_path"])
                print("Labels loaded.")
                print("Number of supervoxels:", len(np.unique(segments)))
                print("Range of labels:", np.min(segments), np.max(segments))
                print("Shape of labels:", segments.shape)
                self.segments_info = segments

            print("---------------------------------------------------")

            self.handle_supervoxels(data, segments)

        # ------------------------------------------------------------------------------------------------

        fields = None
        if self.enable_preprocessing:
            self.arrays_raw = {}
            fields = {}

            all_zero_voxels = np.all(np.isclose(data, 0), axis=3)
            for idx, name in enumerate(header):
                array = data[:, :, :, idx]
                min_val = np.nanmin(array)
                max_val = np.nanmax(array)
                fields[name] = {
                    "label": name,
                    "data_range": [min_val, max_val],
                    "focus_range": [min_val, max_val],
                    "histogram": None,
                    "enabled": True,
                    "color": "black",
                }

                # Save array for later processing
                self.arrays_raw[name] = array

        # Provide control on data arrays
        self.state.data_channels = fields
        print("Data channels ", fields)

        # Store the data in a flattened form. It is easier to work with.
        flattened_data = data.reshape(
            np.prod(self.data_shape), self.num_channels
        )
        print("Flattened data shape ", flattened_data.shape)
        self.nonzero_indices = ~np.all(np.isclose(flattened_data, 0), axis=1)

        # Only store nonzero data. We will reconstruct the zeros later.
        self.nonzero_data = flattened_data[self.nonzero_indices]
        print("Nonzero data shape ", self.nonzero_data.shape)

        # ------------------------------------------------------------------------------------------------

        if self.use_supervoxels:
        
            print("---------------------------------------------------")

            nonzero_final_colored_volume = self.flattened_final_colored_volume[self.nonzero_indices]
            print("Nonzero final colored volume shape ", nonzero_final_colored_volume.shape)

            self.kmeans_rgb_data = nonzero_final_colored_volume

            print("---------------------------------------------------")

        # ------------------------------------------------------------------------------------------------

        # Trigger an update of the data
        self.update_gbc()

    def handle_supervoxels(self, data, segments):
        unique_segments = np.unique(segments)
        num_segments = unique_segments.size
        num_channels = data.shape[-1]

        # Preallocate result array
        segment_vectors = np.zeros((num_segments, num_channels), dtype=data.dtype)

        # Compute mean per channel per segment
        for c in range(num_channels):
            segment_vectors[:, c] = ndimage.mean(data[..., c], labels=segments, index=unique_segments)

        print("Segment vector shape ", segment_vectors.shape)

        print("---------------------------------------------------")

        # Perform K-means clustering on the segment vectors to get an initial set of clusters

        start_time = time.time()

        self.clusterArray = {}
        self.indexArray = None

        for i in range(self.num_clusters):
            self.clusterArray[str(i)] = False

        self.state.cluster_array = self.clusterArray

        if self.cluster_method == "kmeans":
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(segment_vectors)

            print(f"K-means clustering completed in {time.time() - start_time:.2f} seconds")

        print("---------------------------------------------------")

        cluster_color_dict = {}

        colormap = plt.cm.get_cmap("tab10")

        for i in range(self.num_clusters):  # 5 clusters
            cluster_color_dict[i] = colormap(i)

        for cluster_id in np.unique(cluster_labels):
            cluster_color_dict[cluster_id] = list(map(lambda x: x, cluster_color_dict[cluster_id][:3]))

        print("Cluster Color Dictionary for K-means clusters: ", cluster_color_dict)

        print("---------------------------------------------------")

        final_colored_volume = np.zeros(data.shape)
        self.final_cluster_labels = np.zeros(data.shape)

        final_colored_volume = final_colored_volume[:, :, :, :3]
        self.final_cluster_labels = self.final_cluster_labels[:, :, :, 0]

        segment_ids = segments  
        cluster_ids = cluster_labels[segment_ids]

        # Assign final cluster labels
        self.final_cluster_labels = cluster_ids

        max_cluster_id = max(cluster_color_dict.keys())
        color_array = np.zeros((max_cluster_id + 1, 3), dtype=np.float32) 

        for cid, color in cluster_color_dict.items():
            color_array[cid] = color 

        final_colored_volume[:] = color_array[cluster_ids]

        print("Final colored volume shape ", final_colored_volume.shape)

        print("Final cluster labels shape ", self.final_cluster_labels.shape)

        self.flattened_final_colored_volume = final_colored_volume.reshape(np.prod(self.data_shape), 3)

        print("Flattened final colored volume shape ", self.flattened_final_colored_volume.shape)

        print("---------------------------------------------------")

    def create_table(self):
        if self.label_map is None:
            # Nothing to do
            return

        # Fake table data for now
        table_headers = [{"title": "Name", "value": "name"}]
        for i, name in enumerate(self.state.component_labels):
            table_headers.append({
                "title": f'{name} (%)',
                "value": str(i),
                "align": "end",
            })

        table_content = []

        label_values = np.unique(self.label_map)

        data = self.raw_unpadded_flattened_data.reshape(
              (*self.data_shape, self.num_channels)
        )
        if self.normalize_channels:
            # Normalize all channels separately
            for i in range(data.shape[-1]):
                data[:, :, :, i] = _normalize_data(data[:, :, :, i])
        else:
            # Normalize them all together
            data = _normalize_data(data)

        if self.label_map_names:
            labels = self.label_map_names
        else:
            labels = list(map(str, label_values))

        for name, value in zip(labels, label_values):
            # Calculate the percent of each element
            matching_voxels = data[self.label_map == value]

            # Remove voxels that are all close to zero
            matching_voxels = matching_voxels[
                ~np.all(np.isclose(matching_voxels, 0), axis=1)]

            mean_values = matching_voxels.mean(axis=0)
            # Get each mean to add up to 1
            mean_values /= mean_values.sum()

            row = {"id": value.item(), "name": name}
            for i in range(len(self.state.component_labels)):
                row[str(i)] = f'{mean_values[i] * 100:6.2f}'

            table_content.append(row)

        self.state.table_headers = table_headers
        self.state.table_content = table_content

    def update_histograms(self, use_log_histogram):
        # histogram always use the full spectrum of the data
        # TODO: should we then store the two instances (log/non-log) instead of re-calculating ?
        nonzero_indices = ~np.all(
            np.isclose(self.raw_unpadded_flattened_data, 0), axis=1
        )
        nonzero_data = self.raw_unpadded_flattened_data[nonzero_indices]

        for idx, name in enumerate(self.state.component_labels):
            histogram_array = nonzero_data[:, idx]
            hist_count = np.histogram(histogram_array, bins=200)[0].astype(
                float
            )

            zero_counts = np.isclose(hist_count, 0)
            # Perform log scaling, as that is easier to see. Ignore zeros.
            if use_log_histogram:
                hist_count[~zero_counts] = np.log10(hist_count[~zero_counts])
            max_count = hist_count.max()
            hist = [int(v / max_count * 100) for v in hist_count.tolist()]
            self.state.data_channels[name]['histogram'] = hist

    def update_gbc(self):
        gbc, components = compute_gbc(self.nonzero_data)

        print("Nonzero data shape ", self.nonzero_data.shape)
        print("GBC shape ", gbc.shape)
        print("Components shape ", components.shape)
        print("-----------------------------------------------------------------")

        self.unrotated_gbc = gbc
        self.state.unrotated_component_coords = components.tolist()

        self.update_bin_data()
        self.update_voxel_colors()

    @change('use_log_histogram')
    def on_use_log_histogram(self, use_log_histogram, **kwargs):
        self.update_histograms(use_log_histogram)
        # make data_channels dirty so that the UI element of the histogram is updated
        self.state.dirty("data_channels")

    @change('w_bins', 'w_sample_size')
    def update_bin_data(self, **kwargs):
        num_samples = self.state.w_sample_size
        num_bins = self.state.w_bins

        print("Number of samples ", num_samples)
        print("Number of bins ", num_bins)

        # Perform random sampling
        sample_idx = np.random.choice(
            len(self.unrotated_gbc), size=num_samples
        )
        data = self.unrotated_gbc[sample_idx]
        unrotated_bin_data = data_topology_reduction(data, num_bins)
        self.state.unrotated_bin_data = unrotated_bin_data.tolist()

        print("Data shape before sampling", self.unrotated_gbc.shape)
        print("Data shape after sampling", unrotated_bin_data.shape)
        print("-------------------------------------------------------------")

    @change('w_rotation')
    def update_voxel_colors(self, **kwargs):
        angle = np.radians(self.state.w_rotation)
        gbc = rotate_coordinates(self.unrotated_gbc, angle)

        self.gbc_data = gbc
        self.rgb_data = gbc_to_rgb(gbc)

        self.update_volume_data()

    @change(
        "table_selection",
        "unselected_opacity_multiplier",
    )
    def update_volume_data(self, **kwargs):
        if any(x is None for x in (self.rgb_data, self.gbc_data)):
            return

        rgb = self.rgb_data

        print("RGB Data shape", self.rgb_data.T.shape)
        if self.use_supervoxels:
            print("Original Kmeans RGB Data shape", self.kmeans_rgb_data.shape)
            if self.rgb_data.T.shape != self.kmeans_rgb_data.shape:
                print("Kmeans RGB Data shape", self.filtered_kmeans_rgb_data.shape)
            else:
                print("Kmeans RGB Data shape", self.kmeans_rgb_data.shape)

        print("-------------------------------------------------------------")

        # Reconstruct full data with rgba values
        full_data = np.zeros((np.prod(self.data_shape), 4))
        if self.use_supervoxels:
            if self.rgb_data.T.shape != self.kmeans_rgb_data.shape:
                full_data[self.nonzero_indices, :3] = self.filtered_kmeans_rgb_data
            else:
                full_data[self.nonzero_indices, :3] = self.kmeans_rgb_data
        else:
            full_data[self.nonzero_indices, :3] = rgb.T

        if self.opacity_data is None:
            # Make nonzero voxels have an alpha of the mean of the channels.
            full_data[self.nonzero_indices, 3] = self.nonzero_data.mean(axis=1)
            if self.state.table_selection:
                idx = self.state.table_selection[0]
                # Significantly decrease opacity of non-selected voxels
                selected_voxels = self.label_map == idx
                multiplier = self.state.unselected_opacity_multiplier
                full_data[~selected_voxels.flatten(), 3] *= multiplier
        else:
            full_data[self.nonzero_indices, 3] = self.opacity_data.flatten()[
                self.nonzero_indices
            ]

        full_data = full_data.reshape((*self.data_shape, 4))

        # Set the data on the volume
        self.volume_view.set_data(full_data)

        # Update the mask data too. This will trigger an update.
        self.update_mask_data()

    @change(
        'lens_center',
        'show_groups',
        'w_lradius',
        'w_linvert',
        'w_clip_x',
        'w_clip_y',
        'w_clip_z',
        'cluster_array',
        'selected_supervoxels'
    )
    def update_mask_data(self, **kwargs):
        if any(x is None for x in (self.rgb_data, self.gbc_data)):
            return

        if 'filter-cluster' in self.state.show_groups:
            # Get value of cluster_array from arguments
            new_cluster_array = kwargs.get('cluster_array', None)
            clusterChanged = False
            if new_cluster_array is not None and (self.clusterArray != new_cluster_array):
                clusterChanged = True
                self.clusterArray = new_cluster_array

        else:
            clusterChanged = False

            for i in range(self.num_clusters):
                if self.state.cluster_array[str(i)]:
                    clusterChanged = True
                    break
            
            if clusterChanged:
                # Set all values of clusterArray to False
                for i in range(self.num_clusters):
                    self.clusterArray[str(i)] = False

                self.state.cluster_array = self.clusterArray # UI not updating

        alpha = self.compute_alpha(clusterChanged)
        mask_ref = self.volume_view.mask_reference
        mask_ref[self.nonzero_indices] = alpha
        self.volume_view.mask_data.Modified()

        self.update_histograms(True)

        # Update the view
        self.ctrl.view_update()

        # Also update the statistics
        self.update_displayed_voxel_means()

    @change("show_groups")
    def update_displayed_voxel_means(self, **kwargs):
        first_call = not hasattr(self, '_initial_display_voxel_means_call')
        if not first_call and not self.voxel_means_enabled:
            # Only perform this on the first call if voxel means is not enabled
            return

        if first_call:
            self._initial_display_voxel_means_call = False

        alpha = self.volume_view.mask_reference[self.nonzero_indices]
        raw_nonzero = self.raw_unpadded_flattened_data[self.nonzero_indices]

        display_data = raw_nonzero[alpha == 1]
        if display_data.shape[0] > 0:
            # divide each row with the row sum to create percentages for each voxel
            row_sums = display_data.sum(axis=1)
            percentage_per_voxel = np.divide(
                display_data,
                row_sums[:, None],
                out=np.zeros_like(display_data),
                where=row_sums[:, None] != 0,
            )
            means = (
                100.0
                * percentage_per_voxel.sum(axis=0)
                / percentage_per_voxel.shape[0]
            )
        else:
            means = np.zeros(display_data.shape[1])

        labels = self.state.component_labels
        displayed_voxel_means = {k: v for k, v in zip(labels, means.tolist())}
        self.state.displayed_voxel_means = displayed_voxel_means
        self.server.controller.figure_update(_bar_plot(displayed_voxel_means))

    @change("data_channels")
    @change("normalize_ranges")
    def on_data_change(self, data_channels, **_):
        if not self.state.array_modified:
            # No updates were actually made. Just return
            return
        
        print("data_channels - changed")
        self.state.component_labels = [
            item.get("label")
            for item in data_channels.values()
            if item.get("enabled")
        ]

        arrays = []

        # Set a voxel to be zero in all channels if one channel
        # is outside the focus range.
        set_to_zero = np.zeros(self.data_shape, dtype=bool)
        for key, item in data_channels.items():
            if item.get("enabled"):
                array = self.arrays_raw[key]

                focus_range = item["focus_range"]
                set_to_zero[array < focus_range[0]] = True
                set_to_zero[array > focus_range[1]] = True

                arrays.append(array)

        # Update rest of pipeline
        data = np.stack(arrays, axis=3)

        if self.state.normalize_ranges:
            # Set any invalid voxels to zero before normalizing
            data[set_to_zero] = 0

        if self.normalize_channels:
            # Normalize all channels separately
            for i in range(data.shape[-1]):
                data[:, :, :, i] = _normalize_data(data[:, :, :, i])
        else:
            # Normalize them all together
            data = _normalize_data(data)

        if not self.state.normalize_ranges:
            # The invalid voxels are set to zero after normalizing instead
            data[set_to_zero] = 0

        # Store the data in a flattened form. It is easier to work with.
        flattened_data = data.reshape(np.prod(self.data_shape), len(arrays))
        self.nonzero_indices = ~np.all(np.isclose(flattened_data, 0), axis=1)

        # Only store nonzero data. We will reconstruct the zeros later.
        self.nonzero_data = flattened_data[self.nonzero_indices]

        if self.use_supervoxels:
            self.filtered_kmeans_rgb_data = self.flattened_final_colored_volume[self.nonzero_indices]

        # Trigger an update of the data
        self.update_gbc()

    @change("w_rendering_shadow", "w_rendering_bg")
    def on_rendering_settings(
        self, w_rendering_shadow, w_rendering_bg, **kwargs
    ):
        self.volume_view.volume_property.SetShade(
            1 if w_rendering_shadow else 0
        )
        if w_rendering_bg:
            self.volume_view.renderer.SetBackground(1, 1, 1)
        else:
            self.volume_view.renderer.SetBackground(0, 0, 0)
        self.ctrl.view_update()

    @change("slice_axis")
    def update_max_slice_index(self, slice_axis, **kwargs):
        if slice_axis == "x":
            self.state.max_slice_index = self.data_shape[0] - 1
        elif slice_axis == "y":
            self.state.max_slice_index = self.data_shape[1] - 1
        else:
            self.state.max_slice_index = self.data_shape[2] - 1

        self.state.slice_index = 0  # reset index on axis change if needed

    @change("slice_index")
    @change("show_groups")
    def update_slice(self, slice_index, **kwargs):
        if 'visualize-slice' not in self.state.show_groups:
            self.state.w_clip_x = [0, 1]
            self.state.w_clip_y = [0, 1]
            self.state.w_clip_z = [0, 1]
            if hasattr(self, '_initial_update_slice_call'):
                del self._initial_update_slice_call

        else:
            first_call = not hasattr(self, '_initial_update_slice_call')

            if first_call:
                self._initial_update_slice_call = True
                return

            fractional_slice_index = slice_index / self.state.max_slice_index

            if self.state.slice_axis == "x":
                self.state.w_clip_x = [fractional_slice_index, fractional_slice_index + 0.01]
                self.state.w_clip_y = [0, 1]
                self.state.w_clip_z = [0, 1]
            elif self.state.slice_axis == "y":
                self.state.w_clip_y = [fractional_slice_index, fractional_slice_index + 0.01]
                self.state.w_clip_x = [0, 1]
                self.state.w_clip_z = [0, 1]
            else:
                self.state.w_clip_z = [fractional_slice_index, fractional_slice_index + 0.01]
                self.state.w_clip_x = [0, 1]
                self.state.w_clip_y = [0, 1]

    @change("cluster_method")
    @change("cluster_count")
    @change("use_autoencoder")
    @change("normalize_channels")
    @change("dataset")
    def update_cluster_data(self, **kwargs):
        # Get values from arguments
        cluster_method = kwargs.get('cluster_method', None)
        cluster_count = kwargs.get('cluster_count', None)
        use_autoencoder = kwargs.get('use_autoencoder', None)
        normalize_channels = kwargs.get('normalize_channels', None)
        dataset = kwargs.get('dataset', None)

        print("Cluster method ", cluster_method)
        print("Cluster count ", cluster_count)
        print("Use autoencoder ", use_autoencoder)
        print("Normalize channels ", normalize_channels)
        print("Dataset ", dataset)

        print("-------------------------------------------------------------")

        if cluster_count is not None:
            self.num_clusters = int(cluster_count)
        if cluster_method is not None:
            self.cluster_method = cluster_method
        if use_autoencoder is not None:
            self.use_autoencoder = use_autoencoder
        if normalize_channels is not None:
            self.normalize_channels = normalize_channels
        if dataset is not None:
            self.dataset = dataset

        if dataset is not None:
            # self.ui = self._build_ui()
            self.load_data()

    @property
    def state(self):
        return self.server.state

    @property
    def ctrl(self):
        return self.server.controller

    @property
    def render_window(self):
        return self.volume_view.render_window

    @property
    def lens_enabled(self):
        return "lens" in self.state.show_groups

    @property
    def voxel_means_enabled(self):
        return ("voxel-means" in self.state.show_groups) or (
            "voxel-means-plot" in self.state.show_groups
        )

    @life_cycle.server_ready
    def initial_reset_camera(self, **kwargs):
        self.volume_view.renderer.ResetCameraClippingRange()
        self.volume_view.renderer.ResetCamera()

    @property
    def clip_ranges(self):
        return [
            self.state.w_clip_x,
            self.state.w_clip_y,
            self.state.w_clip_z,
        ]

    def compute_alpha(self, clusterChanged):
        gbc_data = self.gbc_data
        if gbc_data is None:
            # Can't do anything
            return None

        clip_mask = np.zeros(self.data_shape, dtype=bool)
        slices = []
        for i, (min_clip, max_clip) in enumerate(self.clip_ranges):
            min_idx = int(np.round(self.data_shape[i] * min_clip))
            max_idx = int(np.round(self.data_shape[i] * max_clip))
            slices.append(np.s_[min_idx:max_idx])

        clip_mask[slices[0], slices[1], slices[2]] = True

        # ------------------------------------------------------------------------------------------------

        if self.use_supervoxels:
            if not clusterChanged and self.indexArray is not None:
                # If the cluster array is not changed, we can use the indexArray to find indexes we have to set as 0 in clip_mask
                clip_mask[self.indexArray] = 0
            else:
                print("--------------------------------------------------")
                # Filter out only selected clusters
                cluster_array = self.state.cluster_array

                selected_clusters = [i for i, v in cluster_array.items() if v]

                print("Selected clusters ", selected_clusters)
                print("Selected supervoxels ", self.state.selected_supervoxels)

                cluster_mask = np.zeros(self.data_shape, dtype=bool)
                supervoxel_mask = np.zeros(self.data_shape, dtype=bool)
                self.indexArray = np.zeros(self.data_shape, dtype=bool)
                
                if len(selected_clusters) > 0:
                    # Convert selected_clusters and selected_supervoxels to sets of ints
                    selected_cluster_ids = set(map(int, selected_clusters))

                    # Create mask for cluster condition
                    cluster_ids = self.final_cluster_labels.astype(int)
                    cluster_mask = ~np.isin(cluster_ids, list(selected_cluster_ids))

                if self.state.selected_supervoxels is not None:
                    selected_supervoxels = np.array(self.state.selected_supervoxels, dtype=self.segments_info.dtype)

                    # Create mask for supervoxel mismatch
                    supervoxel_mask = ~np.isin(self.segments_info, selected_supervoxels)

                # Combine masks: True where either condition is met
                combined_mask = cluster_mask | supervoxel_mask

                # Apply the mask
                clip_mask[combined_mask] = 0
                self.indexArray[combined_mask] = True

        # ------------------------------------------------------------------------------------------------

        # Reshape into the flat form and remove any zero index data
        clip_flattened = clip_mask.reshape(np.prod(self.data_shape))
        # If we perform any other operations, we can logical_and them
        alpha = clip_flattened[self.nonzero_indices]

        if not self.lens_enabled:
            # Only apply clipping
            return alpha

        # These are in unit circle coordinates
        r = self.state.w_lradius
        x, y = self.state.lens_center

        lens_alpha = _compute_alpha(np.array([x, y]), r, gbc_data)
        if self.state.w_linvert:
            lens_alpha = np.invert(lens_alpha)

        # Combine the lens alpha with the current alpha
        return np.logical_and(alpha, lens_alpha)

    def _build_ui(self):
        self.state.setdefault('lens_center', [0, 0])
        self.state.setdefault("array_modified", '')
        self.state.setdefault('normalize_ranges', False)
        self.state.setdefault("unselected_opacity_multiplier", 0.1)

        server = self.server
        ctrl = self.ctrl

        @ctrl.set("get_search")
        def get_search(search):
            print("******************************************************************")
            print("******************************************************************")
            print("Page Load Happened!!")
            print("******************************************************************")
            print("******************************************************************")
            # Reset parameters on load
            self.indexArray = None
            self.num_clusters = 5
            self.cluster_method = "kmeans"
            self.use_autoencoder = False
            self.normalize_channels = False

            if self.use_supervoxels:
                for i in range(self.num_clusters):
                    self.clusterArray[str(i)] = False
                self.state.cluster_array = self.clusterArray

            for key in self.state.data_channels.keys():
                array = self.arrays_raw[key]
                min_val = np.nanmin(array)
                max_val = np.nanmax(array)
                self.state.data_channels[key]['focus_range'] = [min_val, max_val]

            self.state.selected_supervoxels = None
            self.state.dataset = "CeCoFeGd"
            self.state.normalize_channels = False
            self.state.dirty("cluster_array")
            # ----------------------------------------------------
            if not search or search == "?":
                print("No search params")
                self.state.selected_supervoxels = None
                return
            # -----------------------------------------------------
            print(search)
            query_params = search[1:].split("&")

            for param in query_params:
                key, value = param.split("=")
                if key == "cluster_method":
                    print("1")
                    self.state.cluster_method = value
                elif key == "cluster_count":
                    print("2")
                    self.state.cluster_count = int(value)
                elif key == "use_autoencoder":
                    print("3")
                    self.state.use_autoencoder = value
                elif key == "selected_supervoxel":
                    print("4")
                    # Parase coomma separeted values and put them in a list
                    value_list = list(map(int, value.split("%2C")))
                    self.state.selected_supervoxels = value_list
                elif key == "normalize_channels":
                    print("5")
                    self.state.normalize_channels = value
                elif key == "dataset":
                    print("6")
                    self.state.dataset = value

        self.state.trame__title = "MultivariateView"
        self.state.trame__favicon = ASSETS.favicon

        with VAppLayout(server, full_height=True) as layout:
            client.Style('html { overflow-y: hidden; }')

            client.ClientTriggers(mounted=(ctrl.get_search, "[window.location.search]"))

            with vtk.VtkRemoteView(
                self.render_window, interactive_ratio=1
            ) as html_view:
                ctrl.reset_camera = html_view.reset_camera
                ctrl.view_update = html_view.update

                with v.VCard(
                    classes=(
                        "{ 'ma-4': 1, 'rounded-xl': !show_control_panel }",
                    ),
                    style="z-index: 1; position: absolute; top: 0.2rem; left: 0.2rem; max-height: calc(100vh - 2.4rem); overflow: auto;",
                ):
                    with v.VToolbar(
                        density="compact",
                        style="position: sticky; top: 0;",
                    ):
                        v.VProgressLinear(
                            color="primary",
                            indeterminate=("trame__busy",),
                            v_show="trame__busy",
                            absolute=True,
                            style="bottom: 0; top: none;",
                        )
                        v.VBtn(
                            icon="mdi-cogs",
                            click="show_control_panel = !show_control_panel",
                            density="compact",
                            classes="mx-3",
                        )
                        v.VSpacer()

                        with v.VBtnToggle(
                            v_show=("show_control_panel", True),
                            v_model=("show_groups", []),
                            # base_color="grey-darken-1",
                            # color="grey-darken-4",
                            variant="outlined",
                            density="conpact",
                            multiple=True,
                            divided=True,
                            classes="mr-4",
                        ):
                            v.VBtn(icon="mdi-database", value="tune-data", v_if="data_channels && Object.keys(data_channels).length")
                            v.VBtn(icon="mdi-magnify", value="lens")
                            v.VBtn(icon="mdi-palette", value="color")
                            v.VBtn(icon="mdi-eye-settings-outline",value="rendering")
                            v.VBtn(icon="mdi-chart-histogram", value="sampling")
                            v.VBtn(icon="mdi-crop", value="clip")
                            v.VBtn(icon="mdi-sigma", value="voxel-means")
                            v.VBtn(icon="mdi-align-vertical-bottom", value="voxel-means-plot")
                            v.VBtn(icon="mdi-table", value="table")
                            v.VBtn(icon="mdi-scatter-plot", value="filter-cluster")
                            v.VBtn(icon="mdi-video-2d", value="visualize-slice")

                        v.VSpacer()

                        if self.server.hot_reload:
                            v.VBtn(
                                v_show=("show_control_panel", True),
                                icon="mdi-refresh",
                                click=self.ctrl.on_server_reload,
                                density="compact",
                            )

                        v.VBtn(
                            icon="mdi-crop-free",
                            density="compact",
                            classes="mr-3",
                            click=ctrl.reset_camera,
                        )

                    # Main widget
                    radvolviz.NdColorMap(
                        brush_mode=1 if self.use_supervoxels else 0,
                        v_show="show_control_panel",
                        component_labels=('component_labels', []),
                        unrotated_bin_data=('unrotated_bin_data', []),
                        unrotated_component_coords=(
                            'unrotated_component_coords',
                            [],
                        ),
                        size=400,
                        rotation=('w_rotation', 0),
                        sample_size=('w_sample_size', 1100),
                        number_of_bins=('w_bins', 6),
                        show_lens=("show_groups.includes('lens')",),
                        lens_radius=('w_lradius', 0.5),
                        lens='lens_center = $event',
                        # style="position: sticky; top: 3rem; z-index: 1; background: white;",
                    )

                    # -------------------------------------------------------------------------------------

                    # Lense control
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('lens')",
                        classes="py-1",
                    ):
                        v.VSlider(
                            v_model='w_lradius',
                            min=0.001,
                            max=1.0,
                            step=0.001,
                            density='compact',
                            prepend_icon="mdi-radius-outline",
                            messages="Adjust lens size",
                        )
                        v.VSwitch(
                            label="Invert lens",
                            v_model=('w_linvert', False),
                            messages="Invert lens",
                            density="compact",
                            hide_details=True,
                            inset=True,
                            color="green",
                            classes="ml-2",
                        )

                    # --------------------------------------------------------------------------------------

                    # Color / Rotation management
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('color')",
                        classes="py-1",
                    ):
                        v.VSlider(
                            v_model='w_rotation',
                            min=0,
                            max=360,
                            step=5,
                            density='compact',
                            prepend_icon="mdi-rotate-360",
                            messages="Rotate color wheel",
                        )

                    # ---------------------------------------------------------------------------------------

                    # Rendering settings
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('rendering')",
                        classes="py-1",
                    ):
                        with v.VCol():
                            v.VSwitch(
                                label="Use shadow",
                                v_model=('w_rendering_shadow', True),
                                density='compact',
                            )
                            v.VSwitch(
                                label="Use white background",
                                v_model=('w_rendering_bg', False),
                                density='compact',
                            )

                    # --------------------------------------------------------------------------------------

                    # Data sampling
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('sampling')",
                        classes="py-1",
                    ):
                        v.VSlider(
                            v_model='w_sample_size',
                            min=100,
                            max=10000,
                            step=500,
                            density='compact',
                            prepend_icon="mdi-blur-radial",
                            messages="Adjust sampling size",
                        )
                        v.VSlider(
                            v_model='w_bins',
                            min=1,
                            max=10,
                            step=1,
                            density='compact',
                            prepend_icon="mdi-chart-scatter-plot-hexbin",
                            messages="Number of bins for the sampling algorithm",
                        )

                    # ---------------------------------------------------------------------------------------

                    # Cropping
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('clip')",
                        classes="py-1 pr-4",
                    ):
                        v.VLabel("Crop dataset", classes="text-body-2 ml-1")
                        v.VDivider(classes="mr-n4")
                        v.VRangeSlider(
                            label='X',
                            v_model=('w_clip_x', [0, 1]),
                            min=0.0,
                            max=1.0,
                            step=0.001,
                            density='compact',
                            hide_details=True,
                        )
                        v.VRangeSlider(
                            label='Y',
                            v_model=('w_clip_y', [0, 1]),
                            min=0.0,
                            max=1.0,
                            step=0.001,
                            density='compact',
                            hide_details=True,
                        )
                        v.VRangeSlider(
                            label='Z',
                            v_model=('w_clip_z', [0, 1]),
                            min=0.0,
                            max=1.0,
                            step=0.001,
                            density='compact',
                            hide_details=True,
                        )

                    # ----------------------------------------------------------------------------------------

                    # Data tuning
                    with v.VCard(
                        v_if="data_channels",
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('tune-data')",
                        classes="py-1",
                    ):
                        v.VLabel(
                            "Data pre-processing", classes="text-body-2 ml-1"
                        )
                        v.VSwitch(
                            v_model=("use_log_histogram", True),
                            density='compact',
                            hide_details=True,
                            inset=True,
                            color="green",
                            classes="ml-2",
                            label="log scale (histograms)",
                            true_icon="mdi-check",
                            false_icon="mdi-close",
                        )
                        v.VDivider(classes="mr-n4")
                        v.VSwitch(
                            v_model=('normalize_ranges', True),
                            density='compact',
                            hide_details=True,
                            inset=True,
                            color="green",
                            classes="ml-2",
                            label="Rescale channels after range edits",
                            true_icon="mdi-check",
                            false_icon="mdi-close",
                        )
                        v.VDivider(classes="mr-n4")
                        with v.VRow(
                            v_for=("data, name in data_channels"),
                            key="name",
                            classes="mx-0 my-1",
                        ):
                            with v.VCol(
                                cols="1", align_self="center pa-0 ma-0"
                            ):
                                html.Div(
                                    "{{ name }}",  # : Scale({{ data.scale }}) Clamp({{ data.clamp[0] }}, {{ data.clamp[1] }})
                                    classes="text-body-2 text-center text-truncate",
                                    style="transform: rotate(-90deg) translateY(calc(-100% - 0.2rem));  width: 5.5rem;",
                                )
                            with v.VCol(
                                classes="border-s-lg",
                                style=(
                                    "`border-color: ${data.color} !important;`",
                                ),
                            ):
                                with v.VRow(classes="mx-0"):
                                    v.VTextField(
                                        model_value=("data.label",),
                                        density='compact',
                                        hide_details=True,
                                        prepend_icon="mdi-tag-outline",
                                        variant="outlined",
                                        update_modelValue="data_channels[name].label = $event; array_modified='';flushState('data_channels')",
                                    )
                                    v.VSwitch(
                                        model_value=("data.enabled",),
                                        density='compact',
                                        hide_details=True,
                                        inset=True,
                                        color="green",
                                        classes="ml-2",
                                        true_icon="mdi-check",
                                        false_icon="mdi-close",
                                        update_modelValue="data_channels[name].enabled = $event; array_modified=''; flushState('data_channels')",
                                    )
                                with html.Div(
                                    style="height: 4rem;",
                                    classes="align-baseline d-flex mt-5 ml-12 mr-2 mb-n3",
                                ):
                                    html.Div(
                                        v_for="v, idx in data.histogram",
                                        key="idx",
                                        style=(
                                            "`height: ${v}%; width: 0.5%;`",
                                        ),
                                        classes="d-flex bg-blue",
                                    )
                                v.VRangeSlider(
                                    model_value=('data.focus_range',),
                                    min=("data.data_range[0]",),
                                    max=("data.data_range[1]",),
                                    step=(
                                        "(data.data_range[1] - data.data_range[0]) / 255",
                                    ),
                                    density='compact',
                                    hide_details=True,
                                    prepend_icon="mdi-magnify",
                                    update_modelValue="data_channels[name].focus_range = $event; array_modified=name; flushState('data_channels')",
                                )

                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('voxel-means')",
                        classes="py-1",
                    ):
                        with v.VTable(density="compact"):
                            with html.Tbody():
                                with html.Tr(
                                    v_for="v, k in displayed_voxel_means",
                                    key="k",
                                ):
                                    html.Td("{{ k }}", classes="text-caption")
                                    html.Td(
                                        "{{ v.toFixed(2) }}%",
                                        classes="text-caption",
                                        style="text-align: right; padding-right: 10rem;",
                                    )

                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('voxel-means-plot')",
                        classes="py-1",
                    ):
                        with html.Div(style="width: 100%; height: 10rem;"):
                            figure = plotly.Figure(
                                display_logo=False,
                                display_mode_bar="true",
                            )
                            self.server.controller.figure_update = (
                                figure.update
                            )

                    if self.label_map is not None:
                        # Table (phase selection)
                        with v.VCard(
                            flat=True,
                            v_show="show_control_panel && show_groups.includes('table')",
                            classes="py-1 pr-4",
                            v_if="table_content",
                        ):
                            v.VLabel("Label Map", classes="text-body-2 ml-1")
                            v.VDivider(classes="mr-n4")
                            v.VSlider(
                                v_model='unselected_opacity_multiplier',
                                min=0,
                                max=1,
                                step=0.01,
                                density='compact',
                                prepend_icon="mdi-chart-scatter-plot-hexbin",
                                messages="Unselected Voxels Opacity Multiplier",
                            )
                            v.VDataTable(
                                headers=("table_headers", []),
                                items=("table_content", None),
                                density="compact",
                                item_value="id",
                                item_selectable=True,
                                select_strategy="single",  # all / single
                                show_select=True,
                                v_model=("table_selection", []),
                                hide_default_footer=True,
                            )

                    # ----------------------------------------------------------------------------------------

                    # Filter clusters
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('filter-cluster')",
                        classes="py-1",
                    ):
                        v.VLabel("Choose the Clusters to visualize", classes="text-body-2 ml-1")
                        v.VDivider(classes="mr-n4")
                        
                        with v.VRow(
                            v_for=("data, name in cluster_array"),
                            key="name",
                            classes="mx-0 my-1",
                        ):
                            v.VSwitch(
                                v_model=("cluster_array[name]", False),
                                label=("`Cluster ${name}`", None),
                                density="compact",
                                hide_details=True,
                                inset=True,
                                color="green",
                                classes="ml-2",
                                update_modelValue="cluster_array[name] = $event; array_modified=name; flushState('cluster_array')"
                            )

                    # -----------------------------------------------------------------------------------------

                    # Visualize slice
                    with v.VCard(
                        flat=True,
                        v_show="show_control_panel && show_groups.includes('visualize-slice')",
                        classes="py-1",
                    ):
                        v.VLabel("Visualize a slice", classes="text-body-2 ml-1")
                        v.VDivider(classes="mr-n4")

                        # Axis selection (x, y, z)
                        with v.VRow(classes="mx-0 my-1"):
                            with v.VRadioGroup(
                                v_model=("slice_axis", "x"),  # default value
                                row=True,
                                classes="ml-2",
                            ):
                                v.VRadio(label="X", value="x")
                                v.VRadio(label="Y", value="y")
                                v.VRadio(label="Z", value="z")

                        v.VDivider(classes="mr-n4")

                        with v.VRow(classes="mx-0 my-1", align="center"):
                            # Current slice label
                            v.VLabel("Slice Index:", classes="mr-2")
                            v.VLabel("{{ slice_index }}", classes="font-weight-bold")

                        v.VSlider(
                            v_model=("slice_index", 0),
                            min=("min_slice_index", 0),
                            max=("max_slice_index", 100),  # Dynamically updated based on axis
                            step=1,
                            hide_details=True,
                            class_="mx-2",
                            prepend_icon="mdi-magnify",
                        )

                        # Min and Max labels
                        with v.VRow(classes="mx-2", justify="space-between", style="margin-top: 10px;"):
                            v.VLabel("Min: {{ min_slice_index }}")
                            v.VLabel("Max: {{ max_slice_index }}")
                        
                        with v.VRow(classes="mx-0 my-1", align="center", style="margin-top: 10px;"):
                            # External site button
                            v.VBtn(
                                "Visualize in 2D",
                                color="primary",
                                class_="mx-2 mt-2",
                                style="margin-top: 20px !important; margin-left: 30% !important;",
                                click="window.open(`http://127.0.0.1:8050?axis=${slice_axis}&index=${slice_index}&data=${data_path}`, '_blank')"
                            )
                
            # print(layout)
            return layout

    # Function to generate supervoxels
    def generate_supervoxel(self, data, save_data=False):
        print("Starting SLIC segmentation...")

        start_time = time.time()

        # Call the custom SLIC function
        num_supervoxels = 500
        labels, cluster_centers = custom_slic(data, num_supervoxels)

        print("SLIC segmentation complete.")
        print("Supervoxel generation time:", time.time() - start_time)

        print("Number of supervoxels:", len(cluster_centers))
        print("Range of labels:", np.min(labels), np.max(labels))
        print("Shape of labels:", labels.shape)

        # Save the labels
        if save_data:
            np.save(data_dict[self.dataset]["segment_path"], labels)

            print("Labels saved.")

        return labels, cluster_centers

@numba.njit(cache=True, nogil=True)
def _compute_alpha(center, radius, gbc_data):
    # Compute distance formula to lens center
    distances = np.sqrt(((gbc_data - center) ** 2).sum(axis=1))

    # Any distances less than the radius are within the lens
    return distances < radius


@numba.njit(cache=True, nogil=True)
def _remove_padding_uniform(data: np.ndarray) -> np.ndarray:
    num_channels = data.shape[-1]
    zero_data = np.isclose(data, 0).sum(axis=3) == num_channels

    # This is the number to crop
    n = 0
    indices = np.array([n, -n - 1])
    while (
        zero_data[indices].all()
        & zero_data[:, indices].all()
        & zero_data[:, :, indices].all()
    ):
        n += 1
        indices = np.array([n, -n - 1])

    if n != 0:
        data = data[n : -n - 1, n : -n - 1, n : -n - 1]

    return data


@numba.njit(cache=True, nogil=True)
def _normalize_data(data: np.ndarray, new_min: float = 0, new_max: float = 1):
    max_val = data.max()
    min_val = data.min()

    return (new_max - new_min) * (data.astype(np.float64) - min_val) / (
        max_val - min_val
    ) + new_min


def _bar_plot(key_values):
    return go.Figure(
        data=go.Bar(x=list(key_values.keys()), y=list(key_values.values()))
    ).update_layout(yaxis_title="%", margin=dict(l=10, r=10, t=25, b=10))

# -------------------------------------------------------------------------------

def initialize_cluster_centers(data, num_superpixels, grid_spacing):
    """
    Initialize cluster centers based on a uniform 3D grid, ignoring zero-value voxels.
    """
    depth, rows, cols, _ = data.shape
    centers = []
    for d in range(grid_spacing // 2, depth, grid_spacing):
        for r in range(grid_spacing // 2, rows, grid_spacing):
            for c in range(grid_spacing // 2, cols, grid_spacing):
                if not np.all(data[d, r, c] == 0):  # Ignore zero-value voxels
                    centers.append([d, r, c] + list(data[d, r, c]))
    return np.array(centers, dtype=np.float32)

@numba.njit
def calculate_distance(voxel, center, spatial_weight, feature_weight):
    """
    Compute the combined distance between a voxel and a cluster center in 3D.
    """
    spatial_dist = np.sqrt((voxel[0] - center[0])**2 + (voxel[1] - center[1])**2 + (voxel[2] - center[2])**2)
    feature_dist = np.sqrt(np.sum((voxel[3:] - center[3:])**2))
    return np.sqrt((spatial_dist / spatial_weight)**2 + (feature_dist / feature_weight)**2)

@numba.njit
def assign_voxels_to_clusters(data, cluster_centers, labels, distances, spatial_weight, feature_weight, grid_spacing):
    """
    Assign each voxel to the nearest cluster center within its search window, ignoring zero-value voxels.
    """
    depth, rows, cols, _ = data.shape
    for idx, center in enumerate(cluster_centers):
        d, r, c = int(center[0]), int(center[1]), int(center[2])
        
        # Search window around the cluster center
        d_min = max(d - grid_spacing, 0)
        d_max = min(d + grid_spacing + 1, depth)
        r_min = max(r - grid_spacing, 0)
        r_max = min(r + grid_spacing + 1, rows)
        c_min = max(c - grid_spacing, 0)
        c_max = min(c + grid_spacing + 1, cols)

        for dd in range(d_min, d_max):
            for rr in range(r_min, r_max):
                for cc in range(c_min, c_max):
                    if np.all(data[dd, rr, cc] == 0):
                        continue  # Ignore zero-value voxels
                    
                    voxel = np.array([dd, rr, cc] + list(data[dd, rr, cc]))
                    distance = calculate_distance(voxel, center, spatial_weight, feature_weight)
                    if distance < distances[dd, rr, cc]:
                        distances[dd, rr, cc] = distance
                        labels[dd, rr, cc] = idx

@numba.njit
def update_cluster_centers(data, labels, cluster_centers):
    """
    Compute new cluster centers based on the average of assigned voxels.
    """
    num_clusters = len(cluster_centers)
    num_features = data.shape[3]
    
    new_centers = np.zeros_like(cluster_centers)
    counts = np.zeros(num_clusters, dtype=np.int32)

    depth, rows, cols = labels.shape
    for d in range(depth):
        for r in range(rows):
            for c in range(cols):
                label = labels[d, r, c]
                if label == -1:
                    continue  # Skip unassigned voxels
                new_centers[label][:3] += np.array([d, r, c], dtype=np.float32)
                new_centers[label][3:] += data[d, r, c]
                counts[label] += 1

    for i in range(num_clusters):
        if counts[i] > 0:
            new_centers[i] /= counts[i]

    return new_centers

def custom_slic(data, num_superpixels, spatial_weight=5, max_iter=20):
    """
    Perform optimized SLIC segmentation on 4D volumetric data,
    ensuring zero-value voxels are ignored in segmentation and grouped into a single supervoxel at the end.
    """
    depth, rows, cols, num_features = data.shape
    grid_spacing = int(np.cbrt((depth * rows * cols) / num_superpixels))
    feature_weight = np.std(data.reshape(-1, num_features), axis=0).mean()

    # Initialize cluster centers
    cluster_centers = initialize_cluster_centers(data, num_superpixels, grid_spacing)
    labels = -1 * np.ones((depth, rows, cols), dtype=np.int32)
    distances = np.full((depth, rows, cols), np.inf, dtype=np.float32)

    # Iterate until convergence
    for _ in range(max_iter):
        assign_voxels_to_clusters(data, cluster_centers, labels, distances, spatial_weight, feature_weight, grid_spacing)
        cluster_centers = update_cluster_centers(data, labels, cluster_centers)
    
    # Assign all zero-value voxels to a single supervoxel label
    zero_voxel_label = (np.max(labels) + 1)
    for d in range(depth):
        for r in range(rows):
            for c in range(cols):
                if np.all(np.abs(data[d, r, c]) < 1e-15):
                    labels[d, r, c] = zero_voxel_label

    # Ensure all voxels are assigned to a cluster
    unassigned = np.where(labels == -1)
    if unassigned[0].size > 0:
        tree = cKDTree(cluster_centers[:, :3])
        points = np.column_stack(unassigned)
        _, nearest_labels = tree.query(points)
        labels[unassigned] = nearest_labels

    return labels, cluster_centers

# -------------------------------------------------------------------------------