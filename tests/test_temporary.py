"""
Placeholder test cases for meshrender -- hard to test it in CI
as doing so requires an X display.
"""
import unittest

class PointsTest(unittest.TestCase):

    def test_temporary(self):
        pass

# class Test3DVis(unittest.TestCase):

#     import trimesh
#     from visualization import Visualizer3D as vis3d
#     import numpy as np
#     from autolab_core import RigidTransform

#     def test_3d_vis(self):

#         m = trimesh.load('strawberry.obj')
#         m_name = 'Strawberry'
#         vis3d.figure()
#         vis3d.mesh(m, name=m_name)
#         assert vis3d.get_object(name=m_name).name == m_name
#         points = 0.01*np.random.randn(10,3)
#         vis3d.points(points, color=(1,0,0))
#         vis3d.save('test.jpg')
#         vis3d.save_loop('test.gif')

#         stp = m.compute_stable_poses(threshold=0.05)[0][0]
#         R, t = RigidTransform.rotation_and_translation_from_matrix(stp)
#         vis3d.figure()
#         vis3d.mesh_stable_pose(m, RigidTransform(rotation=R, translation=t), name=m_name)
#         vis3d.pose()
#         vis3d.get_object_keys()
#         vis3d.caption('{}'.format(m_name), location=vis3d.TOP_CENTER)
#         vis3d.show()

if __name__ == '__main__':
    unittest.main()
