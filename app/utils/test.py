import numpy as np
import unittest
from adapthresh import sparseOccupancy_AND_XOR, \
    access_partial_Occupancy_ijk
from denoising import __cluster_inCube__, __mark_overlappingLabels__, \
    denoise_crossCubes
from image import preprocess_patches
from scene import initializeCubes
from viewPairSelection import __argmaxN_viewPairs__


class utils_test(unittest.TestCase):

    def test_sparseOccupancy_AND_XOR(self):
        ijk1 = np.array([[1, 0], [2, 3], [222, 666], [0, 0]])
        ijk2 = np.array([[11, 10], [2, 3], [22, 66], [0, 0], [7, 17]])
        self.assertEqual(sparseOccupancy_AND_XOR(ijk1, ijk2), (2, 5))

    def test_access_partial_Occupancy_ijk(self):
        Occ_ijk = np.array([[1, 5, 2], [5, 2, 0], [0, 1, 5], [2, 1, 1],
                           [4, 5, 5]])
        gt = Occ_ijk[2:3] - np.array([[0, 0, 3]])
        result = access_partial_Occupancy_ijk(Occ_ijk, (-1, 0, 1),
                D_cube=6)
        self.assertTrue((gt == result).all())
        self.assertTrue(gt.shape == result.shape)

    def test_cluster_inCube(self):
        vxl_ijk_list = [np.array([
            [1, 0, 0],
            [2, 2, 2],
            [3, 3, 3],
            [1, 0, 1],
            [2, 3, 3],
            [0, 3, 3],
            [1, 2, 2],
            ]), np.array([[0, 2, 3], [0, 1, 0], [0, 0, 0], [0, 3, 3]]),
                np.array([[0, 2, 3], [0, 1, 0], [0, 2, 3]]),
                np.array([[0, 2, 3], [0, 1, 3], [0, 0, 0], [0, 3, 3],
                         [3, 3, 3]], dtype=np.uint8)]
        vxl_mask_list = [np.array([
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            ], dtype=np.bool), np.array([1, 1, 0, 1], dtype=np.bool),
                np.array([0, 0, 0], dtype=np.bool), np.array([1, 1, 1,
                    1, 1], dtype=np.bool)]
        res = __cluster_inCube__(vxl_ijk_list, vxl_mask_list)
        des = ([np.array([
            2,
            0,
            4,
            2,
            4,
            1,
            3,
            ], dtype=np.uint32), np.array([2, 1, 0, 2],
                    dtype=np.uint32), np.array([0., 0., 0.]),
                np.array([2, 2, 1, 2, 3], dtype=np.uint32)], [4, 2, 0,
                    3])
        for i in range(len(des[0])):
            self.assertTrue((res[0][i] == des[0][i]).all())
        self.assertTrue(res[1] == des[1])

    def test_mark_overlappingLabels(self):
        cube_ijk_np = np.array([[1, 6, 8], [2, 6, 8], [2, 7, 8], [2, 5,
                               8]], dtype=np.uint8)
        vxl_ijk_list = [np.array([
            [1, 0, 0],
            [2, 2, 2],
            [3, 2, 3],
            [3, 3, 3],
            [1, 0, 1],
            [2, 3, 3],
            [3, 0, 3],
            ], dtype=np.uint8), np.array([
            [0, 2, 3],
            [0, 1, 3],
            [0, 0, 0],
            [0, 3, 3],
            [1, 0, 3],
            [3, 3, 0],
            ], dtype=np.uint8), np.array([[0, 2, 3], [0, 1, 3], [0, 0,
                    0], [0, 3, 3]], dtype=np.uint8), np.array([[0, 2,
                    3], [0, 1, 3], [0, 0, 0], [0, 3, 3], [3, 3, 3]],
                    dtype=np.uint8)]
        vxl_mask_list = [np.array([
            1,
            0,
            0,
            1,
            1,
            1,
            1,
            ], dtype=np.bool), np.array([
            1,
            1,
            0,
            1,
            1,
            1,
            ], dtype=np.bool), np.array([0, 0, 0, 0], dtype=np.bool),
                np.array([1, 1, 1, 1, 1], dtype=np.bool)]
        res = __mark_overlappingLabels__(cube_ijk_np, vxl_ijk_list,
                vxl_mask_list, D_cube=4)
        des = ([[2, 3], [1, 2], [], [2]], [np.array([
            1,
            0,
            0,
            2,
            1,
            2,
            3,
            ], dtype=np.uint32), np.array([
            1,
            1,
            0,
            1,
            2,
            3,
            ], dtype=np.uint32), np.array([0., 0., 0., 0.]),
                np.array([2, 2, 1, 2, 3], dtype=np.uint32)])
        self.assertTrue(res[0] == des[0])
        for i in range(len(des[1])):
            self.assertTrue((res[1][i] == des[1][i]).all())

    def test_denoise_crossCubes(self):
        cube_ijk_np = np.array([[1, 6, 8], [2, 6, 8], [2, 7, 8], [2, 5,
                               8]], dtype=np.uint8)
        vxl_ijk_list = [np.array([[1, 0, 0], [2, 2, 2], [3, 3, 3], [1,
                        0, 1], [2, 3, 3]], dtype=np.uint8),
                        np.array([[0, 2, 3], [0, 1, 3], [0, 0, 0], [0,
                        3, 3], [3, 3, 0]], dtype=np.uint8),
                        np.array([[0, 2, 3], [0, 1, 3], [0, 0, 0], [0,
                        3, 3]], dtype=np.uint8), np.array([[0, 2, 3],
                        [0, 1, 3], [0, 0, 0], [0, 3, 3], [3, 3, 3]],
                        dtype=np.uint8)]
        vxl_mask_list = [np.array([1, 0, 1, 1, 1], dtype=np.bool),
                         np.array([1, 1, 0, 1, 1], dtype=np.bool),
                         np.array([0, 0, 0, 0], dtype=np.bool),
                         np.array([1, 1, 1, 1, 1], dtype=np.bool)]
        res = denoise_crossCubes(cube_ijk_np, vxl_ijk_list,
                                 vxl_mask_list, D_cube=4)
        des = [np.array([False, False, True, False, True], dtype=bool),
               np.array([True, True, False, True, False], dtype=bool),
               np.array([False, False, False, False], dtype=bool),
               np.array([True, True, False, True, False], dtype=bool)]
        for i in range(len(des[0])):
            self.assertTrue((res[0][i] == des[0][i]).all())

    def test_preprocess_patches(self):
        res = preprocess_patches(np.zeros((2, 2, 5, 3)),
                                 mean_BGR=np.array([1, 2, 3]))
        des = [[[[-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.]],
               [[-2., -2., -2., -2., -2.], [-2., -2., -2., -2., -2.]],
               [[-3., -3., -3., -3., -3.], [-3., -3., -3., -3., -3.]]],
               [[[-1., -1., -1., -1., -1.], [-1., -1., -1., -1., -1.]],
               [[-2., -2., -2., -2., -2.], [-2., -2., -2., -2., -2.]],
               [[-3., -3., -3., -3., -3.], [-3., -3., -3., -3., -3.]]]]
        self.assertTrue((res == des).all())

    def test_initializeCubes(self):
        (cubes_param_np, _) = initializeCubes(resol=1, cube_D=22,
                cube_Dcenter=10, cube_overlapping_ratio=0.5,
                BB=np.array([[3, 88], [-11, 99], [-110, -11]]))
        res = (cubes_param_np['xyz'])[18:22]
        des = [[-3., -17., -26.], [-3., -17., -21.], [-3., -17., -16.],
               [-3., -17., -11.]]

        for i in range(len(des)):
            self.assertTrue((res[i] == des[i]).all())


unittest.main(argv=['first-arg-is-ignored'], exit=False)
