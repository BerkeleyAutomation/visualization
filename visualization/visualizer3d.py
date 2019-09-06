import uuid
import matplotlib.pyplot as plt
import numpy as np
import threading
import trimesh
import time
from shapely.geometry import Polygon

from autolab_core import RigidTransform
from pyrender import Scene, Mesh, Viewer, Node, MetallicRoughnessMaterial, TextAlign

class Visualizer3D(object):
    """
    Dex-Net extension of the base PyOpenGL-based 3D visualization tools
    """
    _scene = Scene(bg_color=np.ones(3), ambient_light=np.ones(3)*0.02)
    _size = np.array([1000, 1000])
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
        Visualizer3D._scene = Scene(bg_color=np.array(bg_color), ambient_light=np.ones(3)*0.02)
        Visualizer3D._size = size

    @staticmethod
    def grasp(grasp, gripper_color=None, tooltip_color=None):
        """Show a grasp by visualizing the gripper in its pose.
        """
        Visualizer3D.mesh(grasp.gripper.geometry, pose=grasp.pose, color=gripper_color)
        f or tooltip, pose in zip(grasp.gripper.tooltips, grasp.gripper.tooltip_poses):
            Visualizer3D.mesh(tooltip.geometry, pose=grasp.pose.dot(pose), color=tooltip_color)

    @staticmethod
    def show(asynch=False, animate=False, axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Display the current figure and enable interaction.

        Parameters
        ----------
        asych : bool
            Whether to run Viewer in separate thread
        animate : bool
            Whether or not to animate the scene.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after showing the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        if 'use_raymond_lighting' not in kwargs:
            kwargs['use_raymond_lighting'] = True
        if 'save_directory' not in kwargs:
            kwargs['save_directory'] = Visualizer3D._save_directory

        Visualizer3d._kwargs.update({'rotate_axis': axis,
                                     'rotate': animate,
                                     'run_in_thread': asynch,
                                     'clf': clf})
        Visualizer3D._kwargs.update(kwargs)

        viewer = Visualizer3D._run(Visualizer3D._kwargs)

    @staticmethod
    def render(record_time=1,rate=(np.pi/3.0), axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Render frames from the viewer.

        Parameters
        ----------
        record_time : float
            Number of seconds to record frames. Defaults to 1 second.
        rate : float
            Rate of rotation in radians per second. Defaults to Pi/3.0.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after rendering the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.

        Returns
        -------
        list of perception.ColorImage
            A list of ColorImages rendered from the viewer.
        """
        Visualizer3D._kwargs.update(kwargs)
        Visualizer3d._kwargs.update({'rotate_axis': axis,
                                    'run_in_thread': True,
                                    'clf': clf,
                                    'record': True,
                                    'rotate': True,
                                    'rotate_rate': rate})

        viewer = Visualizer3D._run(kwargs=Visualizer3D._kwargs)
        time.sleep(record_time)
        viewer.close_external()

        return viewer._saved_frames

    @staticmethod
    def save(filename, record_time=1,rate=(np.pi/3.0), axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Save frames from the viewer out to a file.

        Parameters
        ----------
        filename : str
            The filename in which to save the output image. If more than one frame,
            should have extension .gif.
        record_time : float
            Number of seconds to record frames. Defaults to 1 second.
        rate : float
            Rate of rotation in radians per second. Defaults to PI/3.0.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after rendering the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        n_frames = record_time * rate
        gif = os.path.splitext(filename)[1] == '.gif'
        if n_frames >1 and not gif:
            raise ValueError('Expected .gif file for multiple-frame save.')

        Visualizer3D._kwargs.update(kwargs)
        Visualizer3D._kwargs.update({'rotate_axis': axis,
                                    'run_in_thread': True,
                                    'clf': clf,
                                    'record': True,
                                    'rotate': True,
                                    'rotate_rate': rate})

        viewer = Visualizer3D._run(kwargs=Visualizer3D._kwargs)
        time.sleep(record_time)
        viewer.close_external()

        if gif:
            viewer.save_gif(filename)
        else:
            imageio.imwrite(filename, v.saved_frames[0].data)

    @staticmethod
    def _run_viewer(kwargs):
        return Viewer(Visualizer3D._scene, viewport_size=Visualizer3D._size, **kwargs)

    @staticmethod
    def _run(kwargs):
        viewer = Visualizer3D._run_viewer(kwargs)
        if viewer.viewer_flags['save_directory']:
            Visualizer3D._save_directory = viewer.viewer_flags['save_directory']
        if clf:
            Visualizer3D.clf()
        Visualizer3D._kwargs = {}

    @staticmethod
    def save_loop(filename, framerate=30, time=3.0, axis=np.array([0.,0.,1.]), clf=True, **kwargs):
        """Off-screen save a GIF of one rotation about the scene.

        Parameters
        ----------
        filename : str
            The filename in which to save the output image (should have extension .gif)
        framerate : int
            The frame rate at which to animate motion.
        time : float
            The number of seconds for one rotation.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after rendering the figure.
        kwargs : dict
            Other keyword arguments for the SceneViewer instance.
        """
        n_frames = framerate * time
        Visualizer3D.save(filename, record_time=time, rate=(2.0 * np.pi /n_frames),
                                    axis=axis, clf=clf)

    @staticmethod
    def clf():
        Visualizer3D._scene = Scene(bg_color=Visualizer3D._scene.background_color)
        Visualizer3D._scene.ambient_light = np.ones(3)

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
        return Visualizer3D._scene.get_nodes(name=name)

    @staticmethod
    def mesh(mesh, name=None, T_mesh_world=None, style='surface',
                color=(0.5,0.5,0.5), material=None, smooth=False):
        """Visualize a 3D triangular mesh.

        Parameters
        ----------
        mesh : trimesh.Trimesh
            The mesh to visualize.
        T_mesh_world : autolab_core.RigidTransform
            The pose of the mesh, specified as a transformation from mesh frame to world frame.
        style : str
            Triangular mesh style, either 'surface' or 'wireframe'.
        smooth : bool
            If true, the mesh is smoothed before rendering.
        color : 3-tuple
            Color tuple.
        name : str
            A name for the object to be added.
        """
        n = Visualizer3D._create_node_from_mesh(mesh, name=name, pose=pose, color=color,
                                                material=material, poses=None, wireframe=(style=='wireframe'), smooth=smooth)
        Visualizer3D._scene.add_node(n)
        return n

    @staticmethod
    def mesh_stable_pose(mesh, T_obj_table,
                         T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                         style='wireframe', smooth=False, color=(0.5,0.5,0.5),
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

        Visualizer3D.mesh(mesh, T_obj_world, style=style, smooth=smooth, color=color, name=name)
        if plot_table:
            Visualizer3D.table(T_table_world, dim=dim)
        if plot_com:
            Visualizer3D.points(Point(np.array(mesh.center_mass), 'obj'), T_obj_world, scale=0.01)
        return T_obj_world

    @staticmethod
    def points(points, name=None, T_points_world=None, color=None, material=None, n_cuts=20, scale=0.01):
        """Scatter a point cloud in pose T_points_world.

        Parameters
        ----------
        points : (n,3) float
            The point set to visualize.
        T_points_world : autolab_core.RigidTransform
            Pose of points, specified as a transformation from point frame to world frame.
        color : (3,) or (n,3) float
            Color of whole cloud or per-point colors
        material:
            Material of mesh
        scale : float
            Radius of each point.
        n_cuts : int
            Number of longitude/latitude lines on sphere points.
        name : str
            A name for the object to be added.
        """
        n = Visualizer3D._create_node_from_points(points, name=name, tube_radius=scale,
                            pose=T_points_world, color=color, material=material, n_divs=n_cuts)
        Visualizer3D._scene.add_node(n)
        return n

    @staticmethod
    def plot3d(points, tube_radius=None, name=None, pose=None, color=(0.5, 0.5, 0.5), material=None, n_components=30, smooth=True):
        """Plot a 3d curve through a set of points using tubes.

        Parameters
        ----------
        points : (n,3) float
            A series of 3D points that define a curve in space.
        color : (3,) float
            The color of the tube.
        tube_radius : float
            Radius of tube representing curve.
        n_components : int
            The number of edges in each polygon representing the tube.
        name : str
            A name for the object to be added.
        """
        # Generate circular polygon
        vec = np.array([0.0, 1.0]) * tube_radius
        angle = 2 * np.pi / n_divs
        rotmat = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        perim = []
        for _ in range(n_divs):
            perim.append(vec)
            vec = np.dot(rotmat, vec)
        poly = Polygon(perim)

        # Sweep it along the path
        mesh = trimesh.creation.sweep_polygon(poly, points)

        return Visualizer3D.mesh(mesh, name=name, pose=pose, color=color, material=material, smooth=smooth)

    @staticmethod
    def pose(T_frame_world=None, alpha=0.1, tube_radius=0.005, center_scale=0.01):
        """Plot a 3D pose as a set of axes (x red, y green, z blue).

        Parameters
        ----------
        T_frame_world : autolab_core.RigidTransform
            The pose relative to world coordinates.
        alpha : float
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
        x = np.array([t, t + alpha * R[:,0]])
        y = np.array([t, t + alpha * R[:,1]])
        z = np.array([t, t + alpha * R[:,2]])

        Visualizer3D.points(t, color=(1,1,1), scale=center_scale)
        Visualizer3D.plot(x, tube_radius, color=(1,0,0))
        Visualizer3D.plot(y, tube_radius, color=(0,1,0))
        Visualizer3D.plot(z, tube_radius, color=(0,0,1))

    @staticmethod
    def table(T_table_world=RigidTransform(from_frame='table', to_frame='world'), dim=0.16, color=(0,0,0)):
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

        table_vertices = np.array([[ dim,  dim, 0],
                                   [ dim, -dim, 0],
                                   [-dim,  dim, 0],
                                   [-dim, -dim, 0]]).astype('float')
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        table_mesh = trimesh.Trimesh(table_vertices, table_tris)
        table_mesh.apply_transform(T_table_world.matrix)
        Visualizer3D.mesh(table_mesh, style='surface', smooth=True, color=color)

    @staticmethod
    def caption(text, location=TextAlign.TOP_RIGHT, font_name='OpenSans-Regular', font_pt=20, color=None, scale=1.0):
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
                    #baseColorFactor=np.random.uniform(0.0, 1.0, size=3),
                    baseColorFactor=np.array([0.3, 0.3, 0.3, 1.0]),
                    metallicFactor=0.2,
                    roughnessFactor=0.8
                )

        m = Mesh.from_trimesh(mesh, material=material, poses=poses, wireframe=wireframe, smooth=smooth)
        return Node(mesh=m, name=name, matrix=pose)

    @staticmethod
    def _create_node_from_points(points, name=None, pose=None, color=None, material=None, tube_radius=None, n_divs=20):
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
    def _q_to_c(q):
        return plt.get_cmap('hsv')(0.3 * q)[:-1]
