import os
import time
import imageio
import numpy as np
import trimesh
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
from autolab_core import RigidTransform
from pyrender import Scene, Mesh, Viewer, Node, MetallicRoughnessMaterial, TextAlign

class Visualizer3D(object):
    """
    Class with static PyOpenGL-based 3D visualization tools.
    Should be thought of as a namespace rather than a class.

    """
    _scene = Scene(bg_color=np.ones(3))
    _size = np.array([640, 480])
    _kwargs = {}
    _save_directory = None
    CENTER = TextAlign.CENTER
    CENTER_LEFT = TextAlign.CENTER_LEFT
    CENTER_RIGHT = TextAlign.CENTER_RIGHT
    BOTTOM_LEFT = TextAlign.BOTTOM_LEFT
    BOTTOM_RIGHT = TextAlign.BOTTOM_RIGHT
    BOTTOM_CENTER = TextAlign.BOTTOM_CENTER
    TOP_LEFT = TextAlign.TOP_LEFT
    TOP_RIGHT = TextAlign.TOP_RIGHT
    TOP_CENTER = TextAlign.TOP_CENTER


    @staticmethod
    def figure(bg_color=(1.0, 1.0, 1.0), size=(1000, 1000)):
        """Create a viewing window.

        Parameters
        ----------
        bg_color : (3,) float
            The background RGB color.
        size : (2,) int
            The width and height of the window in pixels.
        """
        Visualizer3D._scene = Scene(bg_color=np.array(bg_color))
        Visualizer3D._size = size


    @staticmethod
    def show(asynch=False, animate=False, **kwargs):
        """Display the current figure and enable interaction.

        Parameters
        ----------
        asynch : bool
            Whether to run Viewer in separate thread
        animate : bool
            Whether or not to animate the scene.
        kwargs : dict
            Other keyword arguments for the Viewer instance.
        """
        Visualizer3D._kwargs.update({'rotate': animate,
                                     'run_in_thread': asynch})
        Visualizer3D._kwargs.update(kwargs)

        viewer = Visualizer3D._run(Visualizer3D._kwargs)


    @staticmethod
    def render(n_frames=1, **kwargs):
        """Render frames from the viewer.

        Parameters
        ----------
        n_frames : int
            Number of frames to render. If more than one, the scene will animate.
        kwargs : dict
            Other keyword arguments for the Viewer instance.

        Returns
        -------
        list of perception.ColorImage
            A list of ColorImages rendered from the viewer.
        """
        Visualizer3D._kwargs.update({'run_in_thread': True,
                                     'record': True,
                                     'rotate': (n_frames > 1)})
        Visualizer3D._kwargs.update(kwargs)
        viewer = Visualizer3D._run(Visualizer3D._kwargs)
        while len(viewer._saved_frames) < n_frames:
            pass
        viewer.close_external()

        if n_frames > 1:
            return viewer._saved_frames
        else: 
            return [viewer._saved_frames[0]]


    @staticmethod
    def save(filename, n_frames=1, **kwargs):
        """Save frames from the viewer out to a file.

        Parameters
        ----------
        filename : str
            The filename in which to save the output image. If more than one frame,
            should have extension .gif.
        n_frames : int
            Number of frames to render. If more than one, the scene will animate.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        frames = Visualizer3D.render(n_frames=n_frames, **kwargs)
        framerate = kwargs['refresh_rate'] if 'refresh_rate' in kwargs else 30.0
        if n_frames > 1:
            imageio.mimwrite(filename, frames,
                             fps=framerate,
                             palettesize=128, subrectangles=True)
        else:
            imageio.imwrite(filename, frames[0])


    @staticmethod
    def _run(kwargs):
        """Internal method that runs the viewer

        Parameters
        ----------
        kwargs : dict
            Other keyword arguments for the Viewer instance.
        """
        if 'use_raymond_lighting' not in kwargs:
            kwargs['use_raymond_lighting'] = True
        viewer = Viewer(Visualizer3D._scene, viewport_size=Visualizer3D._size, **kwargs)
        if viewer.viewer_flags['save_directory']:
            Visualizer3D._save_directory = viewer.viewer_flags['save_directory']
        Visualizer3D._kwargs = {}
        return viewer


    @staticmethod
    def save_loop(filename, framerate=30.0, time=3.0, **kwargs):
        """Off-screen save a GIF of one rotation about the scene.

        Parameters
        ----------
        filename : str
            The filename in which to save the output image (should have extension .gif)
        framerate : int
            The frame rate at which to animate motion.
        time : float
            The number of seconds for one rotation.
        kwargs : dict
            Other keyword arguments for the Viewer instance.
        """
        n_frames = int(framerate * time)
        Visualizer3D.save(filename, n_frames=n_frames, rotate_rate=(2.0 * np.pi / time), refresh_rate=framerate, **kwargs)


    @staticmethod
    def get_object_keys():
        """Return the visualizer's object keys.

        Returns
        -------
        list of str
            The keys for the visualizer's objects.
        """
        return [n.name for n in Visualizer3D._scene.mesh_nodes]


    @staticmethod
    def get_object(name):
        """Return a Node corresponding to the given name.

        Returns
        -------
        pyrender.Node
            The corresponding Node.
        """
        return next(iter(Visualizer3D._scene.get_nodes(name=name)))


    @staticmethod
    def points(points, name=None, T_points_world=None, color=None, material=None, n_cuts=20, scale=0.01):
        """Scatter a point cloud in pose T_points_world.

        Parameters
        ----------
        points : (n,3) float
            The point set to visualize.
        name : str
            A name for the object to be added.
        T_points_world : autolab_core.RigidTransform
            Pose of points, specified as a transformation from point frame to world frame.
        color : (3,) or (n,3) float
            Color of whole cloud or per-point colors
        material:
            Material of mesh
        n_cuts : int
            Number of longitude/latitude lines on sphere points.
        scale : float
            Radius of each point.
        """
        n = Visualizer3D._create_node_from_points(points, name=name, tube_radius=scale,
                            pose=T_points_world, color=color, material=material, n_divs=n_cuts)
        Visualizer3D._scene.add_node(n)
        return n


    @staticmethod
    def mesh(mesh, name=None, T_mesh_world=None, style='surface',
             color=(0.5,0.5,0.5), material=None, smooth=False):
        """Visualize a 3D triangular mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to visualize.
        name : str
            A name for the object to be added.
        T_mesh_world : autolab_core.RigidTransform
            The pose of the mesh, specified as a transformation from mesh frame to world frame.
        style : str
            Triangular mesh style, either 'surface' or 'wireframe'.
        color : 3-tuple
            Color tuple.
        material:
            Material of mesh
        smooth : bool
            If true, the mesh is smoothed before rendering.
        """
        if not isinstance(mesh, trimesh.Trimesh):
            raise ValueError('Must provide a trimesh.Trimesh object')

        n = Visualizer3D._create_node_from_mesh(mesh, name=name, pose=T_mesh_world, color=color,
                                                material=material, poses=None, wireframe=(style=='wireframe'), smooth=smooth)
        Visualizer3D._scene.add_node(n)
        return n


    @staticmethod
    def mesh_stable_pose(mesh, T_obj_table,
                         T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                         style='surface', smooth=False, color=(0.5,0.5,0.5), material=None,
                         dim=0.15, plot_table=True, plot_com=False, name=None):
        """Visualize a mesh in a stable pose.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to visualize.
        T_obj_table : autolab_core.RigidTransform
            Pose of object relative to table.
        T_table_world : autolab_core.RigidTransform
            Pose of table relative to world.
        style : str
            Triangular mesh style, either 'surface' or 'wireframe'.
        smooth : bool
            If true, the mesh is smoothed before rendering.
        color : 3-tuple
            Color tuple.
        material:
            Material of mesh
        dim : float
            The side-length for the table.
        plot_table : bool
            If true, a table is visualized as well.
        plot_com : bool
            If true, a ball is visualized at the object's center of mass.
        name : str
            A name for the object to be added.

        Returns
        -------
        autolab_core.RigidTransform
            The pose of the mesh in world frame.
        """
        T_obj_table = T_obj_table.as_frames('obj', 'table')
        T_obj_world = T_table_world * T_obj_table

        Visualizer3D.mesh(mesh, T_mesh_world=T_obj_world, style=style, smooth=smooth, color=color, material=material, name=name)
        if plot_table:
            Visualizer3D.table(T_table_world, dim=dim)
        if plot_com:
            Visualizer3D.points(Point(np.array(mesh.center_mass), 'obj'), T_points_world=T_obj_world, scale=0.01)
        return T_obj_world


    @staticmethod
    def plot3d(points, tube_radius=None, name=None, pose=None, color=(0.5, 0.5, 0.5), material=None, n_components=30, smooth=True):
        """Plot a 3d curve through a set of points using tubes.

        Parameters
        ----------
        points : (n,3) float
            A series of 3D points that define a curve in space.
        tube_radius : float
            Radius of tube representing curve.
        name : str
            A name for the object to be added.
        pose : autolab_core.RigidTransform
            Pose of object relative to world.
        color : (3,) float
            The color of the tube.
        material:
            Material of mesh
        n_components : int
            The number of edges in each polygon representing the tube.
        smooth : bool
            If true, the mesh is smoothed before rendering.
        """
        # Generate circular polygon
        vec = np.array([0.0, 1.0]) * tube_radius
        angle = 2 * np.pi / n_components
        rotmat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        perim = []
        for _ in range(n_components):
            perim.append(vec)
            vec = np.dot(rotmat, vec)
        poly = Polygon(perim)

        # Sweep it along the path
        mesh = trimesh.creation.sweep_polygon(poly, points)
        return Visualizer3D.mesh(mesh, name=name, T_mesh_world=pose, color=color, material=material, smooth=smooth)


    @staticmethod
    def arrow(start_point, direction, tube_radius=0.005, color=(0.5, 0.5, 0.5), material=None, n_components=30, smooth=True):
        """Plot an arrow with start and end points.

        Parameters
        ----------
        start_point : (3,) float
            Origin point for the arrow
        direction : (3,) float
            Vector defining the arrow
        tube_radius : float
            Radius of plotted x,y,z axes.
        color : (3,) float
            The color of the tube.
        material:
            Material of mesh
        n_components : int
            The number of edges in each polygon representing the tube.
        smooth : bool
            If true, the mesh is smoothed before rendering.
        """
        end_point = start_point + direction
        arrow_head = Visualizer3D._create_arrow_head(length=np.linalg.norm(direction), tube_radius=tube_radius)
        arrow_head_rot = trimesh.geometry.align_vectors(np.array([0,0,1]), direction)
        arrow_head_tf = np.matmul(trimesh.transformations.translation_matrix(end_point), arrow_head_rot)
        
        vec = np.array([start_point, end_point])
        Visualizer3D.plot3d(vec, tube_radius=tube_radius, color=color)
        Visualizer3D.mesh(arrow_head, T_mesh_world=arrow_head_tf, color=color, material=material, smooth=smooth)


    @staticmethod
    def pose(T_frame_world=None, length=0.1, tube_radius=0.005, center_scale=0.01):
        """Plot a 3D pose as a set of axes (x red, y green, z blue).

        Parameters
        ----------
        T_frame_world : autolab_core.RigidTransform
            The pose relative to world coordinates.
        length : float
            Length of plotted x,y,z axes.
        tube_radius : float
            Radius of plotted x,y,z axes.
        center_scale : float
            Radius of the pose's origin ball.
        """
        if T_frame_world is None:
            R = np.eye(3)
            t = np.zeros(3)
        else:
            R = T_frame_world.rotation
            t = T_frame_world.translation

        Visualizer3D.points(t, color=(1,1,1), scale=center_scale)
        Visualizer3D.arrow(t, length * R[:,0], tube_radius=tube_radius, color=(1,0,0))
        Visualizer3D.arrow(t, length * R[:,1], tube_radius=tube_radius, color=(0,1,0))
        Visualizer3D.arrow(t, length * R[:,2], tube_radius=tube_radius, color=(0,0,1))


    @staticmethod
    def table(T_table_world=RigidTransform(from_frame='table', to_frame='world'), dim=0.16, color=(0.3,0.3,0.3)):
        """Plot a table mesh in 3D.

        Parameters
        ----------
        T_table_world : autolab_core.RigidTransform
            Pose of table relative to world.
        dim : float
            The side-length for the table.
        color : 3-tuple
            Color tuple.
        """

        table_mesh = trimesh.creation.box(extents=(dim, dim, dim / 10))
        table_mesh.apply_translation(-np.array([0,0, dim / 20]))
        table_mesh.apply_transform(T_table_world.matrix)
        Visualizer3D.mesh(table_mesh, style='surface', smooth=True, color=color)


    @staticmethod
    def caption(text, location=TextAlign.TOP_RIGHT, font_name='OpenSans-Regular', font_pt=20, color=None, scale=1.0):
        """Displays text on the visualization.

        Parameters
        ----------
        text : str
            The text to be displayed
        location : int
            Enum of location for the text
        font_name : str
            Valid font to be used
        font_pt : int
            Size of font to be used
        color : 3-tuple
            Color tuple.
        scale : float
            Scale of text
        """
        caption_dict = {'text': text,
                        'location': location,
                        'font_name': font_name,
                        'font_pt': font_pt,
                        'color': color,
                        'scale': scale
                       }
        if 'caption' in Visualizer3D._kwargs:
            Visualizer3D._kwargs['caption'].append(caption_dict)
        else:
            Visualizer3D._kwargs['caption'] = [caption_dict]


    @staticmethod
    def _create_node_from_mesh(mesh, name=None, pose=None, color=None, material=None, poses=None, wireframe=False, smooth=True):
        """Helper method that creates a pyrender.Node from a trimesh.Trimesh"""
        # Create default pose
        if pose is None:
            pose = np.eye(4)
        elif isinstance(pose, RigidTransform):
            pose = pose.matrix

        # Create vertex colors if needed
        if color is not None:
            color = np.asanyarray(color, dtype=np.float)
            if color.ndim == 1 or len(color) != len(mesh.vertices):
                color = np.repeat(color[np.newaxis,:], len(mesh.vertices), axis=0)
            mesh.visual.vertex_colors = color

        if material is None and mesh.visual.kind != 'texture':
            if color is not None:
                material = None
            else:
                material = MetallicRoughnessMaterial(
                    baseColorFactor=np.array([1.0, 1.0, 1.0, 1.0]),
                    metallicFactor=0.2,
                    roughnessFactor=0.8
                )

        m = Mesh.from_trimesh(mesh, material=material, poses=poses, wireframe=wireframe, smooth=smooth)
        return Node(mesh=m, name=name, matrix=pose)


    @staticmethod
    def _create_node_from_points(points, name=None, pose=None, color=None, material=None, tube_radius=None, n_divs=20):
        """Helper method that creates a pyrender.Node from an array of points"""
        points = np.asanyarray(points)
        if points.ndim == 1:
            points = np.array([points])

        if pose is None:
            pose = np.eye(4)
        else:
            pose = pose.matrix

        # Create vertex colors if needed
        if color is not None:
            color = np.asanyarray(color, dtype=np.float)
            if color.ndim == 1 or len(color) != len(points):
                color = np.repeat(color[np.newaxis,:], len(points), axis=0)

        if tube_radius is not None:
            poses = None
            mesh = trimesh.creation.uv_sphere(tube_radius, [n_divs, n_divs])
            if color is not None:
                mesh.visual.vertex_colors = color[0]
            poses = np.tile(np.eye(4), (len(points), 1)).reshape(len(points),4,4)
            poses[:, :3, 3::4] = points[:,:,None]
            m = Mesh.from_trimesh(mesh, material=material, poses=poses)
        else:
            m = Mesh.from_points(points, colors=color)

        return Node(mesh=m, name=name, matrix=pose)

    @staticmethod
    def _create_arrow_head(length=0.1, tube_radius=0.005, n_components=30):
        
        radius = tube_radius * 1.5
        height = length * 0.1

        # create a 2D pie out of wedges
        theta = np.linspace(0, np.pi * 2, n_components)
        vertices = np.column_stack((np.sin(theta),
                                    np.cos(theta), 
                                    np.zeros(len(theta)))) * radius
        
        # the single vertex at the center of the circle
        # we're overwriting the duplicated start/end vertex
        # plus add vertex at tip of cone
        vertices[0] = [0, 0, 0]
        vertices = np.append(vertices, [[0, 0, height]], axis=0)

        # whangle indexes into a triangulation of the pie wedges
        index = np.arange(1, len(vertices)).reshape((-1, 1))
        index[-1] = 1
        faces_2d = np.tile(index, (1, 2)).reshape(-1)[1:-1].reshape((-1, 2))
        faces = np.column_stack((np.zeros(len(faces_2d), dtype=np.int), faces_2d))

        # add triangles connecting to vertex above
        faces = np.append(faces, np.column_stack(((len(faces_2d) + 1) * np.ones(len(faces_2d), dtype=np.int), faces_2d))[:,::-1], axis=0)

        arrow_head = trimesh.Trimesh(faces=faces, vertices=vertices)
        return arrow_head
