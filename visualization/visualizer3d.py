"""
Common 3D visualizations
Author: Matthew Matl and Jeff Mahler
"""
import uuid

import numpy as np
import trimesh
from trimesh import Trimesh

from autolab_core import RigidTransform, BagOfPoints, Point
from meshrender import Scene, SceneObject, InstancedSceneObject, AmbientLight, SceneViewer, MaterialProperties

class Visualizer3D:
    """
    Class containing static methods for visualization.
    The interface is styled after pyplot.
    Should be thought of as a namespace rather than a class.
    """
    _scene = Scene(background_color=np.array([1.0, 1.0, 1.0]))
    _init_size = np.array([640,480])
    _init_kwargs = None


    @staticmethod
    def figure(bgcolor=(1,1,1), size=(1000,1000), **kwargs):
        """ Creates a figure.

        Parameters
        ----------
        bgcolor : (3,) float
           Color of the background with values in [0,1].
        size : (2,) int
           Width and height of the figure in pixels.
        kwargs : list
           keyword args for scene viewer.
        """
        Visualizer3D._scene = Scene(background_color=np.array(bgcolor))
        Visualizer3D._scene.ambient_light = AmbientLight(color=[1.0, 1.0, 1.0], strength=1.0)
        Visualizer3D._init_size = np.array(size)
        Visualizer3D._init_kwargs = kwargs


    @staticmethod
    def show(animate=False, az=0.05, rate=30, axis=[0,0,1], clf=True):
        """ Displays a figure and enables interaction.

        Parameters
        ----------
        animate : bool
            Whether or not to animate the scene.
        az : float (optional)
            The azimuth to rotate for each animation timestep.
        rate : float (optional)
            The frame rate at which to animate motion.
        axis : (3,) float or None
            If present, the animation will rotate about the given axis in world coordinates.
            Otherwise, the animation will rotate in azimuth.
        clf : bool
            If true, the Visualizer is cleared after showing the figure.
        """
        SceneViewer(Visualizer3D._scene,
                    size=Visualizer3D._init_size,
                    raymond_lighting=True,
                    bad_normals=True,
                    animate=animate,
                    animate_az=az,
                    animate_rate=rate,
                    animate_axis=axis)

        if clf:
            Visualizer3D.clf()


    @staticmethod
    def clf():
        """ Clear the current figure
        """
        Visualizer3D._scene = Scene(background_color=Visualizer3D._scene.background_color)
        Visualizer3D._scene.ambient_light = AmbientLight(color=[1.0, 1.0, 1.0], strength=1.0)


    @staticmethod
    def close(*args, **kwargs):
        """ Close the current figure
        """
        pass


    @staticmethod
    def points(points, T_points_world=None, color=(0,1,0), scale=0.01, subsample=None, random=False):
        """ Scatters a point cloud in pose T_points_world.

        Parameters
        ----------
        points : :obj:`autolab_core.BagOfPoints`
            point set to visualize
        T_points_world : :obj:`autolab_core.RigidTransform`
            pose of points, specified as a transformation from point frame to world frame
        color : 3-tuple
            color tuple
        scale : float
            scale of each point
        subsample : int
            parameter of subsampling to display fewer points
        """
        if not isinstance(points, BagOfPoints) and points.dim == 3:
            raise ValueError('Data type %s not supported' %(type(points)))

        if subsample is not None:
            points = points.subsample(subsample, random=random)

        # transform into world frame
        if points.frame != 'world':
            if T_points_world is None:
                T_points_world = RigidTransform(from_frame=points.frame, to_frame='world')
            points_world = T_points_world * points
        else:
            points_world = points

        point_data = points_world.data
        if len(point_data.shape) == 1:
            point_data = point_data[:,np.newaxis]
        point_data = point_data.T

        mp = MaterialProperties(
            color = np.array(color),
            k_a = 0.5,
            k_d = 0.3,
            k_s = 0.0,
            alpha = 10.0,
            smooth=True
        )

        # For each point, create a sphere of the specified color and size.
        sphere = trimesh.creation.uv_sphere(scale, [20, 20])
        poses = []
        for point in point_data:
            poses.append(RigidTransform(translation=point, from_frame='obj', to_frame='world'))
        obj = InstancedSceneObject(sphere, poses, material=mp)
        name = str(uuid.uuid4())
        Visualizer3D._scene.add_object(name, obj)

    @staticmethod
    def mesh(mesh, T_mesh_world=RigidTransform(from_frame='obj', to_frame='world'),
             style='surface', smooth=False, color=(0.5,0.5,0.5)):
        """ Visualizes a 3D triangular mesh.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            mesh to visualize
        T_mesh_world : :obj:`autolab_core.RigidTransform`
            pose of mesh, specified as a transformation from mesh frame to world frame
        style : :obj:`str`
            triangular mesh style, see Mayavi docs
        color : 3-tuple
            color tuple
        opacity : float
            how opaque to render the surface
        """
        if not isinstance(mesh, Trimesh):
            raise ValueError('Must provide a trimesh.Trimesh object')

        mp = MaterialProperties(
            color = np.array(color),
            k_a = 0.5,
            k_d = 0.3,
            k_s = 0.1,
            alpha = 10.0,
            smooth=smooth,
            wireframe=(style == 'wireframe')
        )

        obj = SceneObject(mesh, T_mesh_world, mp)
        name = str(uuid.uuid4())
        Visualizer3D._scene.add_object(name, obj)


    @staticmethod
    def mesh_stable_pose(mesh, T_obj_table,
                         T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                         style='wireframe', smooth=False, color=(0.5,0.5,0.5),
                         dim=0.15, plot_table=True, plot_com=False):
        """ Visualizes a 3D triangular mesh.

        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            mesh to visualize
        stable_pose : :obj:`meshpy.StablePose`
            stable pose to visualize
        T_table_world : :obj:`autolab_core.RigidTransform`
            pose of table, specified as a transformation from mesh frame to world frame
        style : :obj:`str`
            triangular mesh style, see Mayavi docs
        color : 3-tuple
            color tuple
        opacity : float
            how opaque to render the surface
        dim : float
            the dimension of the table
        plot_table : bool
            whether or not to plot the table
        plot_com : bool
            whether or not to plot the mesh center of mass

        Returns
        -------
        :obj:`autolab_core.RigidTransform`
            pose of the mesh in world frame
        """
        T_obj_table = T_obj_table.as_frames('obj', 'table')
        T_obj_world = T_table_world * T_obj_table

        Visualizer3D.mesh(mesh, T_obj_world, style=style, smooth=smooth, color=color)
        if plot_table:
            Visualizer3D.table(T_table_world, dim=dim)
        if plot_com:
            Visualizer3D.points(Point(np.array(mesh.center_mass), 'obj'), T_obj_world, scale=0.01)
        return T_obj_world

    @staticmethod
    def pose(T_frame_world, alpha=0.1, tube_radius=0.005, center_scale=0.01,
             show_frame=False):
        """ Plots a pose with frame label.

        Parameters
        ----------
        T_frame_world : :obj:`autolab_core.RigidTransform`
            pose specified as a transformation from the poses frame to the world frame
        alpha : float
            length of plotted x,y,z axes
        tube_radius : float
            radius of plotted x,y,z axes
        center_scale : float
            scale of the pose's origin
        show_frame : bool
            whether to show the frame name in text
        """
        R = T_frame_world.rotation
        t = T_frame_world.translation

        x_axis_tf = np.array([t, t + alpha * R[:,0]])
        y_axis_tf = np.array([t, t + alpha * R[:,1]])
        z_axis_tf = np.array([t, t + alpha * R[:,2]])

        center = Point(t, 'obj')
        Visualizer3D.points(center, color=(1,1,1), scale=center_scale)

        Visualizer3D.plot3d(x_axis_tf, color=(1,0,0), tube_radius=tube_radius)
        Visualizer3D.plot3d(y_axis_tf, color=(0,1,0), tube_radius=tube_radius)
        Visualizer3D.plot3d(z_axis_tf, color=(0,0,1), tube_radius=tube_radius)

    @staticmethod
    def table(T_table_world=RigidTransform(from_frame='table', to_frame='world'), dim=0.16, color=(0,0,0)):
        """ Plots a table of dimension dim in pose T_table_world.

        Parameters
        ----------
        T_table_world : :obj:`autolab_core.RigidTransform`
            pose of the table in world frame
        dim : float
            the dimensions of the table
        color : 3-tuple
            color of table
        """

        table_vertices = np.array([[ dim,  dim, 0],
                                   [ dim, -dim, 0],
                                   [-dim,  dim, 0],
                                   [-dim, -dim, 0]]).astype('float')
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        table_mesh = Trimesh(table_vertices, table_tris)
        table_mesh.apply_transform(T_table_world.matrix)
        Visualizer3D.mesh(table_mesh, style='surface', smooth=True, color=color)

    @staticmethod
    def plot3d(points, color=(0.5, 0.5, 0.5), tube_radius=0.005):
        """Plot a 3d curve through a set of points using tubes.

        Parameters
        ----------
        points : (n,3) float
            A series of 3D points that define a curve in space.
        color : (3,) float
            The color of the tube.
        tube_radius : float
            Radius of tube representing curve.
        Note
        ----
        TODO for this -- change to instanced scene object, need to have similarity TF that
        can scale anisotropically.
        """
        mp = MaterialProperties(
            color = np.array(color),
            k_a = 0.5,
            k_d = 0.3,
            k_s = 0.0,
            alpha = 10.0,
            smooth=True
        )
        for i in range(len(points) - 1):
            p0 = points[i]
            p1 = points[i+1]

            length = np.linalg.norm(p1 - p0)
            center = p0 + (p1 - p0) / 2.0
            z = (p1 - p0) / length
            x = np.array([z[1], -z[0], 0])
            xl = np.linalg.norm(x)
            if xl == 0:
                x = np.array([1.0, 0.0, 0.0])
            else:
                x = x / xl
            y = np.cross(z, x)
            y = y / np.linalg.norm(y)

            R = np.array([x, y, z])

            M = np.eye(4)
            M[:3,:3] = R.T
            M[:3,3] = center

            # Generate a cylinder between p0 and p1
            cyl = trimesh.creation.cylinder(radius=tube_radius, height=length, transform=M)

            # For each point, create a sphere of the specified color and size.
            obj = SceneObject(cyl, material=mp)
            name = str(uuid.uuid4())
            Visualizer3D._scene.add_object(name, obj)
