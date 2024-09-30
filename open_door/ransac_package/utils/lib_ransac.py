'''
This code is partially copied and modified from:
https://scipy-cookbook.readthedocs.io/items/RANSAC.html 
'''

import numpy as np
import scipy
import scipy.linalg
import time


def random_partition(n, N):
    # get n random indices out of [0, 1, 2, ..., N]
    indices = np.random.permutation(N)
    return indices[:n], indices[n:]


class PlaneModel(object):

    def __init__(self, feature_dimension=3):
        ''' For a plane in 3D world, feature_dimension=3. 
        Plane model with weights w: 
            w[0] + w[1]*x + w[2]*y + w[3]*z = 0
        '''
        self._feature_dimension = 3

    def fit_plane(self, points):
        ''' Fit a plane to the data points.
        Return:
            w: shape=(4, 0). Plane model:
                w[0] + w[1]*x + w[2]*y + w[3]*z = 0
        Algorithm: Compute PCA by svd algorithm.
            The least axis of PCA is the plane normal direction.
        Details:
            U, S, W = svd(Xc), where Xc is X subtracted by X's average.
            if Xc=3*N, U[:, -1], last col is the plane norm
            if Xc=N*3, W[-1, :], last row is the plane norm
            Besides, S are square root of eigen values
        '''
        points = self._check_data(points)
        X = points
        X_mean = np.mean(X, axis=0)  # Squash each column to compute mean.
        Xc = X - X_mean[np.newaxis, :]
        U, S, W = np.linalg.svd(Xc)
        plane_normal = W[-1, :]

        '''
        Compute the bias:
            The fitted plane is this: w[1]*(x-xm)+w[2]*(x-ym)+w[3]*(x-zm)=0
            Change it back to the original: w[1]x+w[2]y+w[3]z+(-w[1]xm-w[2]ym-w[3]zm)=0
                --> w[0]=-w[1]xm-w[2]ym-w[3]zm
        '''
        w_0 = np.dot(X_mean, -plane_normal)
        w_1 = plane_normal
        w = np.concatenate(([w_0], w_1))
        return w

    def get_error(self, points, w):
        ''' Compute the distance between each data point and plane.
        Arguments:
            points: shape=(N, 3).
            w: Plane weights. shape=(4, ).
        Return:
            dists: shape=(N, )
        '''
        points = self._check_data(points)
        dists = np.abs(w[0] + points.dot(w[1:]))/np.linalg.norm(w[1:])
        return dists

    def _check_data(self, points):
        ''' Make sure the data shape is (N, 3). '''
        if not self._feature_dimension in points.shape:
            raise ValueError("Wrong input data shape")
        if points.shape[0] == self._feature_dimension:
            points = points.T
        return points

    
class RansacPlane(object):
    def __init__(self):
        pass

    def fit(self,
            points,  # 3xN or Nx3 points of xyz positions.
            model,  # The PlaneModel.
            # n_pts_fit_model: Number of points to sample from source pcd.
            #   This should be the minumum points number to fit a plane model.
            n_pts_fit_model,
            n_min_pts_inlier,  # Min number of points for a valid plane.
            max_iter,
            # dist_thresh: A point is considered as inlier if its distance to the plane is smaller than this.
            dist_thresh,
            # is_print_iter: Print in each iteration when a better model is found.
            is_print_iter=False,
            is_print_res=True,  # Print final results.
            ):
        '''
        Return:
            is_succeed {bool}
            best_w {1D array, size=4}: weight of the detected plane.
                Plane model: w[0] + w[1]*x + w[2]*y + w[3]*z = 0.
            best_res_inliers {1D array}: Indices of the points in the source point cloud
                which are part of the detected plane.
        '''
        FAILURE_RETURN = False, None, None

        # -- Check input
        if points.shape[1] != 3:  # shape: (3, N) --> (N, 3)
            points = points.T
        if len(points) < n_min_pts_inlier:
            return FAILURE_RETURN

        # -- Init variables
        N = points.shape[0]  # Number of data points.
        t0 = time.time()  # Timer
        n_pts_fit_model += 1

        # Variables to store the best model.
        best_w = None
        best_res_num_inliers = -1
        best_res_inliers = []

        # -- Start random sampling and fitting.
        for i in range(max_iter):

            # -- Step 1: Sample some data to fit a model_A.
            maybe_idxs = self._sample_indices(n_pts_fit_model, N)
            maybe_w, maybe_error, all_error = self._fit_model(
                maybe_idxs, points, model)
            n_pts_inlier = np.count_nonzero(all_error < dist_thresh)
            if n_pts_inlier >= n_min_pts_inlier:  # A good model is detected.

                # -- Step 2: Use the inliers of model_A to fit a model_B.

                # Let's use part of the inliers to fit the model again
                also_idxs = np.arange(N)[all_error < dist_thresh]
                # Limit the number of also_idxs to avoid using too many points to compute the model
                np.random.shuffle(also_idxs)
                also_idxs = also_idxs[:n_min_pts_inlier]

                # Fit the model
                also_w, also_error, all_error = self._fit_model(
                    also_idxs, points, model)

                # -- Step 3: Copmare model_B with best model,
                # and decide whether we use model_B or not.

                # Select a criteria for evaluating the model.
                # Here we use the number of points.
                best_inliers = np.arange(N)[all_error < dist_thresh]
                n_pts_inlier = len(best_inliers)
                if n_pts_inlier > best_res_num_inliers:
                    if is_print_iter:
                        print("A better model is found "
                              "in {}th iter: number of inliers = {}".format(
                                  i, n_pts_inlier))
                    best_res_num_inliers = n_pts_inlier
                    best_w = also_w
                    best_res_inliers = best_inliers

        # -- Check if a good model is found.
        if best_w is None:
            return FAILURE_RETURN
        else:
            is_succeed = True

        # -- Print time cost.
        n_inliers = len(best_res_inliers)
        if is_print_res:
            print("RANSAC performance report:")
            print("    Source data points = {}".format(N))
            print("    Inlier data points = {}".format(n_inliers))
            print("    Iterations = {}".format(max_iter))
            print("    Time cost = {:.3} seconds".format(time.time()-t0))
            print("    Plane model: w[0] + w[1]*x + w[2]*y + w[3]*z = 0")
            print("    Weights: w = {}".format(best_w))

        # -- Return result.
        return is_succeed, best_w, best_res_inliers

    def _sample_indices(self, n_pts_fit_model, N):
        rand_indices = np.random.permutation(N)
        return rand_indices[:n_pts_fit_model]

    def _fit_model(self, maybe_idxs, data, model):
        maybe_data = data[maybe_idxs]
        maybe_w = model.fit_plane(maybe_data)
        maybe_error = model.get_error(maybe_data, maybe_w)
        all_error = model.get_error(data, maybe_w)
        return maybe_w, maybe_error, all_error
