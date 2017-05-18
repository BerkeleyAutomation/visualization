"""
Common 3D visualizations
Author: Jeff Mahler
"""
import numpy as np

try:
    import mayavi.mlab as mlab
except ImportError:
    print "Could not import Mayavi! Visualizer will not be working."

from core import RigidTransform
from core import BagOfPoints, Point, PointCloud, RgbPointCloud, NormalCloud
from meshpy import Mesh3D, Sdf3D, StablePose

class Visualizer3D:
    """
    Class containing static methods for visualization.
    The interface is styled after pyplot.
    Should be thought of as a namespace rather than a class.
    """

    @staticmethod
    def figure(bgcolor=(1,1,1), size=(1000,1000), *args, **kwargs):
        """ Creates a figure.

        Parameters
        ----------
        bgcolor : 3-tuple
           color of the background with values in [0,1] e.g. (1,1,1) = white
        size : 2-tuple
           size of the view window in pixels
        args : list
           args of mayavi figure
        kwargs : list
           keyword args of mayavi figure
        """
        return mlab.figure(bgcolor=bgcolor, size=size, *args, **kwargs)
    
    @staticmethod
    def show():
        """ Displays a figure and enables interaction.
        """
        mlab.show()

    @staticmethod
    def clf():
        """ Clear the current figure
        """
        mlab.clf()

    @staticmethod
    def points(points, T_points_world=None, color=(0,1,0), scale=0.01, subsample=None, random=False):
        """ Scatters a point cloud in pose T_points_world.
        
        Parameters
        ----------
        points : :obj:`core.BagOfPoints`
            point set to visualize
        T_points_world : :obj:`core.RigidTransform`
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
        mv_obj = mlab.points3d(point_data[0,:], point_data[1,:], point_data[2,:],
                              color=color, scale_factor=scale)
        return mv_obj

    @staticmethod
    def normals(normals, points, T_normals_world=None, color=(0,1,0), scale=0.01, length=1.0, subsample=None):
        """ Scatters a point cloud with surface normals in pose T_normals_world.
        
        Parameters
        ----------
        normals : :obj:`core.NormalCloud`
            normal cloud to visualize
        points : :obj:`core.BagOfPoints`
            set of points where the normals are rooted (must have the same number of elements as normals)
        T_normals_world : :obj:`core.RigidTransform`
            pose of points and surface normals, specified as a transformation from normals frame to world frame
        color : 3-tuple
            color tuple
        scale : float
            scale of each point
        length : float
            length of each rendered surface normal
        subsample : int
            parameter of subsampling to display fewer points
        """
        if not isinstance(normals, NormalCloud):
            raise ValueError('Normals data type %s not supported' %(type(normals)))
        if points.num_points != normals.num_points:
            raise ValueError('Points and normals must be the same size')

        if not isinstance(points, BagOfPoints) and points.dim == 3:
            raise ValueError('Data type %s not supported' %(type(points)))
        if normals.frame != points.frame:
            raise ValueError('Points and normals must have the same shape')            
        if subsample is not None:
            normals = normals.subsample(subsample)
            points = points.subsample(subsample)
     
        # transform into world frame
        if normals.frame != 'world':
            if T_normals_world is None:
                T_normals_world = RigidTransform(from_frame=normals.frame, to_frame='world')
            normals_world = T_normals_world * normals
            points_world = T_normals_world * points
        else:
            normals_world = normals
            points_world = points

        # plot
        normal_data = length * normals_world.data
        point_data = points_world.data
        mv_obj = mlab.quiver3d(point_data[0,:], point_data[1,:], point_data[2,:],
                               normal_data[0,:], normal_data[1,:], normal_data[2,:],
                               color=color, scale_factor=scale)
        return mv_obj

    @staticmethod
    def surface(points, triangles, T_points_world=None,
                color=(0,1,0), style='surface'):
        """ Triangulates a surface specified as a set of points and triangles.
        
        Parameters
        ----------
        points : :obj:`core.BagOfPoints`
            point set to visualize
        triangles : Nx3 :obj:`numpy.ndarray`
            set of N triangles specified as triplets of integer indices in the points array
        T_points_world : :obj:`core.RigidTransform`
            pose of points, specified as a transformation from point frame to world frame
        color : 3-tuple
            color tuple
        style : :obj:`str`
            triangular mesh style, see Mayavi docs
        """
        if not isinstance(points, BagOfPoints) and points.dim == 3:
            ValueError('Data type %s not supported' %(type(points)))
        if T_points_world is None:
            T_points_world = RigidTransform(from_frame=points.frame, to_frame='world')

        # transform into world frame
        points_world = T_points_world * points
        point_data = points_world.data
        mv_obj = mlab.triangular_mesh(point_data[0,:], point_data[1,:], point_data[2,:],
                                      triangles, representation=style, color=color)
        return mv_obj

    @staticmethod
    def mesh(mesh, T_mesh_world=RigidTransform(from_frame='obj', to_frame='world'),
             style='wireframe', color=(0.5,0.5,0.5), opacity=1.0):
        """ Visualizes a 3D triangular mesh.
        
        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            mesh to visualize
        T_mesh_world : :obj:`core.RigidTransform`
            pose of mesh, specified as a transformation from mesh frame to world frame
        style : :obj:`str`
            triangular mesh style, see Mayavi docs
        color : 3-tuple
            color tuple
        opacity : float
            how opaque to render the surface
        """
        if not isinstance(mesh, Mesh3D):
            raise ValueError('Must provide a meshpy.Mesh3D object')
        vertex_cloud = PointCloud(mesh.vertices.T, frame=T_mesh_world.from_frame)
        vertex_cloud_tf = T_mesh_world * vertex_cloud
        vertices = vertex_cloud_tf.data.T
        surface = mlab.triangular_mesh(vertices[:,0],
                                       vertices[:,1],
                                       vertices[:,2],
                                       mesh.triangles,
                                       representation=style,
                                       color=color, opacity=opacity)
        return surface

    @staticmethod
    def mesh_stable_pose(mesh, stable_pose,
                         T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                         style='wireframe', color=(0.5,0.5,0.5),
                         opacity=1.0, dim=0.15, plot_table=True):
        """ Visualizes a 3D triangular mesh.
        
        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            mesh to visualize
        stable_pose : :obj:`meshpy.StablePose`
            stable pose to visualize
        T_table_world : :obj:`core.RigidTransform`
            pose of table, specified as a transformation from mesh frame to world frame
        style : :obj:`str`
            triangular mesh style, see Mayavi docs
        color : 3-tuple
            color tuple
        opacity : float
            how opaque to render the surface
        dim : float
            the dimension of the table

        Returns
        -------
        :obj:`core.RigidTransform`
            pose of the mesh in world frame
        """
        if not isinstance(stable_pose, StablePose):
            raise ValueError('Must provide a meshpy.StablePose object')
        T_stp_obj = RigidTransform(rotation=stable_pose.r, from_frame='obj', to_frame='stp')
        T_obj_table = mesh.get_T_surface_obj(T_stp_obj).as_frames('obj', 'table')
        T_obj_world = T_table_world * T_obj_table

        Visualizer3D.mesh(mesh, T_obj_world, style=style, color=color, opacity=opacity)
        if plot_table:
            Visualizer3D.table(T_table_world, dim=dim)
        return T_obj_world

    @staticmethod
    def mesh_table(mesh, T_obj_table,
                   T_table_world=RigidTransform(from_frame='table', to_frame='world'),
                   style='wireframe', color=(0.5,0.5,0.5),
                   opacity=1.0, dim=0.15, plot_table=True):
        """ Visualizes a 3D triangular mesh.
        
        Parameters
        ----------
        mesh : :obj:`meshpy.Mesh3D`
            mesh to visualize
        T_mesh_table : :obj:`core.RigidTransform`
            pose of mesh wrt table (rotation only)
        T_table_world : :obj:`core.RigidTransform`
            pose of table, specified as a transformation from mesh frame to world frame
        style : :obj:`str`
            triangular mesh style, see Mayavi docs
        color : 3-tuple
            color tuple
        opacity : float
            how opaque to render the surface
        dim : float
            the dimension of the table

        Returns
        -------
        :obj:`core.RigidTransform`
            pose of the mesh in world frame
        """
        if not isinstance(T_obj_table, RigidTransform):
            raise ValueError('Must provide a core.RigidTransform object')
        T_obj_table = mesh.get_T_surface_obj(T_obj_table).as_frames('obj', 'table')
        T_obj_world = T_table_world * T_obj_table

        Visualizer3D.mesh(mesh, T_obj_world, style=style, color=color, opacity=opacity)
        if plot_table:
            Visualizer3D.table(T_table_world, dim=dim)
        return T_obj_world

    @staticmethod
    def sdf(sdf, T_sdf_world=RigidTransform(from_frame='obj', to_frame='world'),
             color=(0.5,0.5,0.5), scale=0.01, subsample=1, random=False):
        """ Visualizes points on the surface of a 3D signed distance field (SDF)
        
        Parameters
        ----------
        sdf : :obj:`meshpy.Sdf3D`
            sdf to visualize
        T_sdf_world : :obj:`core.RigidTransform`
            pose of sdf, specified as a transformation from sdf frame to world frame
        color : 3-tuple
            color tuple
        scale : float
            scale of the points
        subsample : int
            rate to subsample the point cloud
        random : bool
            whether or not to subsample points randomly
        """
        if not isinstance(sdf, Sdf3D):
            raise ValueError('Must provide a meshpy.Sdf3D object')
        points, _ = sdf.surface_points(grid_basis=False) 
        point_cloud = PointCloud(points.T, frame=T_sdf_world.from_frame)
        return Visualizer3D.points(point_cloud, T_points_world=T_sdf_world,
                                   color=color, scale=scale, subsample=subsample,
                                   random=random)


    @staticmethod
    def pose(T_frame_world, alpha=0.5, tube_radius=0.005, center_scale=0.01,
             show_frame=False):
        """ Plots a pose with frame label.
        
        Parameters
        ----------
        T_frame_world : :obj:`core.RigidTransform`
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
                
        mlab.points3d(t[0], t[1], t[2], color=(1,1,1), scale_factor=center_scale)
            
        mlab.plot3d(x_axis_tf[:,0], x_axis_tf[:,1], x_axis_tf[:,2], color=(1,0,0), tube_radius=tube_radius)
        mlab.plot3d(y_axis_tf[:,0], y_axis_tf[:,1], y_axis_tf[:,2], color=(0,1,0), tube_radius=tube_radius)
        mlab.plot3d(z_axis_tf[:,0], z_axis_tf[:,1], z_axis_tf[:,2], color=(0,0,1), tube_radius=tube_radius)

        if show_frame:
            mlab.text3d(t[0], t[1], t[2], ' %s' %T_frame_world.from_frame.upper(), scale=0.01, color=(0,0,0))

    @staticmethod
    def table(T_table_world=RigidTransform(from_frame='table', to_frame='world'), dim=0.16, color=(0,0,0)):
        """ Plots a table of dimension dim in pose T_table_world.
        
        Parameters
        ----------
        T_table_world : :obj:`core.RigidTransform`
            pose of the table in world frame
        dim : float
            the dimensions of the table
        color : 3-tuple
            color of table
        """
        table_vertices = np.array([[ dim,  dim, 0],
                                   [ dim, -dim, 0],
                                   [-dim,  dim, 0],
                                   [-dim, -dim, 0]]).T.astype('float')
        points_table = PointCloud(table_vertices, frame=T_table_world.from_frame)
        points_world = T_table_world * points_table
        table_tris = np.array([[0, 1, 2], [1, 2, 3]])
        Visualizer3D.surface(points_world, table_tris, color=color)

    @staticmethod
    def view(azimuth=None, elevation=None, distance=None, focalpoint=None):
        """ Set the camera viewpoint.
        
        Parameters
        ----------
        azimuth : float
            azimuth of the camera in spherical coordinates
        elevation : float
            elevation of the camera in spherical coordinates
        distance : float
            distance of camera to the focalpoint
        focalpoint : 3-tuple of float
            point to center the camera on
        """
        mlab.view(azimuth=azimuth, elevation=elevation, distance=distance, focalpoint=focalpoint)

    @staticmethod
    def set_offscreen(toggle):
        """ Set offscreen rendering mode.

        Parameters
        ----------
        toggle : bool
            whether or not to render offscreen
        """
        mlab.options.offscreen = toggle

    @staticmethod
    def savefig(*args, **kwargs):
        """ Save a given figure.

        Parameters
        ----------
        filename : :obj:`str`
            filename to save image to
        """
        mlab.savefig(*args, **kwargs)

    @staticmethod
    def axes():
        """ Plot the x,y,z axes. """
        mlab.axes()
