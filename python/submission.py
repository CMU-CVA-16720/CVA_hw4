"""
Homework4.
Replace 'pass' by your implementation.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

import util
import helper
from findM2 import findM2

'''
Q2.1: Eight Point Algorithm
    Input:  pts1, Nx2 Matrix, contains (x,y)
            pts2, Nx2 Matrix, constains (x,y)
            M, a scalar parameter computed as max (imwidth, imheight)
    Output: F, the fundamental matrix
'''
def eightpoint(pts1, pts2, M):
    # Correspondence equation: prT * F * pl = 0
    # xl*xr*f11 + xr*yl*f12 + xr*f13 + xl*yr*f21 + yl*yr*f22 + yr*f23 + xl*f31 + yl*f32 + f33 = 0
    # Solve for f = [f11, f12,... f32, f33]T
    # Get parameters
    N = pts1.shape[0]
    # Compute normalization matrix
    T = np.zeros((3,3))
    T[[0,1],[0,1]] = 1/M
    T[2,2] = 1
    # Set up matrix A, where Af=0, and A = num_pts x 9
    # Each row: xl*xr + xr*yl + xr + xl*yr + yl*yr + yr + xl + yl + 1
    A = np.ones((N,9))
    for i in range(0,N):
        # Get coordinates
        xl, yl = pts1[i,:]
        xr, yr = pts2[i,:]
        # Normalize coord
        xl,yl,_ = T@[xl,yl,1]
        xr,yr,_ = T@[xr,yr,1]
        # Update A
        A[i,0] = xl*xr
        A[i,1] = xr*yl
        A[i,2] = xr
        A[i,3] = xl*yr
        A[i,4] = yl*yr
        A[i,5] = yr
        A[i,6] = xl
        A[i,7] = yl
    # Compute f_norm using A
    u, s, vT = np.linalg.svd(A)
    f = vT[-1]
    # Reshape, refine, unscale f
    F = np.reshape(f,(3,3))
    F = util.refineF(F,pts1/M,pts2/M)
    F = np.transpose(T) @ F @ T
    return F



'''
Q3.1: Compute the essential matrix E.
    Input:  F, fundamental matrix
            K1, internal camera calibration matrix of camera 1
            K2, internal camera calibration matrix of camera 2
    Output: E, the essential matrix
'''
def essentialMatrix(F, K1, K2):
    # F = Kr^(-T) E Kl^(-1), so E = KrT F Kl
    Kl = K1
    Kr = K2
    return (Kr.T @ F @ Kl)


'''
Q3.2: Triangulate a set of 2D coordinates in the image to a set of 3D points.
    Input:  C1, the 3x4 camera matrix
            pts1, the Nx2 matrix with the 2D image coordinates per row
            C2, the 3x4 camera matrix
            pts2, the Nx2 matrix with the 2D image coordinates per row
    Output: P, the Nx3 matrix with the corresponding 3D points per row
            err, the reprojection error.
'''
def triangulate(C1, pts1, C2, pts2):
    N = pts1.shape[0]
    P = np.zeros((N,3))
    # Set up matrix A, where Aw=0, and A = (2 * 2 * num_pts) x 4
    # x1i = C1 @ w1i -> cross(x1i, C1 @ w1i) =0 -> A1i @ wi = 0; two independent equations:
    # (Cl[3,1]*yl - Cl[2,1])*X + (Cl[3,2]*yl - Cl[2,2])*Y + (Cl[3,3]*yl - Cl[2,3])*Z + (Cl[3,4]*yl - Cl[2,4]) = 0
    # (Cl[1,1] - Cl[3,1]*xl)*X + (Cl[1,2] - Cl[3,2]*xl)*Y + (Cl[1,3] - Cl[3,3]*xl)*Z + (Cl[1,4] - Cl[3,4]*xl) = 0
    # (Cr[3,1]*yr - Cr[2,1])*X + (Cr[3,2]*yr - Cr[2,2])*Y + (Cr[3,3]*yr - Cr[2,3])*Z + (Cr[3,4]*yr - Cr[2,4]) = 0
    # (Cr[1,1] - Cr[3,1]*xr)*X + (Cr[1,2] - Cr[3,2]*xr)*Y + (Cr[1,3] - Cr[3,3]*xr)*Z + (Cr[1,4] - Cr[3,4]*xr) = 0
    Cl = C1
    Cr = C2
    for i in range(0,N):
        # Get coordinates
        xl, yl = pts1[i,:]
        xr, yr = pts2[i,:]
        # Compute A
        A = np.zeros((4,4))
        A[0,:] = np.array([Cl[2,0]*yl - Cl[1,0], Cl[2,1]*yl - Cl[1,1], Cl[2,2]*yl - Cl[1,2], Cl[2,3]*yl - Cl[1,3]])
        A[1,:] = np.array([Cl[0,0] - Cl[2,0]*xl, Cl[0,1] - Cl[2,1]*xl, Cl[0,2] - Cl[2,2]*xl, Cl[0,3] - Cl[2,3]*xl])
        A[2,:] = np.array([Cr[2,0]*yr - Cr[1,0], Cr[2,1]*yr - Cr[1,1], Cr[2,2]*yr - Cr[1,2], Cr[2,3]*yr - Cr[1,3]])
        A[3,:] = np.array([Cr[0,0] - Cr[2,0]*xr, Cr[0,1] - Cr[2,1]*xr, Cr[0,2] - Cr[2,2]*xr, Cr[0,3] - Cr[2,3]*xr])
        # Compute w
        u, s, vT = np.linalg.svd(A)
        w = vT[-1]
        # Convert to non-homogeneous coord, then save
        w /= w[-1]
        P[i,:] = w[0:3]
    # Compute error
    err = 0
    for i in range(0,N):
        # Get x1 and x2
        x1 = pts1[i,:]
        x2 = pts2[i,:]
        # Compute x1_hat, x2_hat
        x1_hat = C1@np.append(P[i,:],1)
        x1_hat /= x1_hat[-1]
        x1_hat = x1_hat[0:2]
        x2_hat = C2@np.append(P[i,:],1)
        x2_hat /= x2_hat[-1]
        x2_hat = x2_hat[0:2]
        # Update error
        err += np.sum(np.square(x1-x1_hat))
        err += np.sum(np.square(x2-x2_hat))
    return P, err


'''
Q4.1: 3D visualization of the temple images.
    Input:  im1, the first image
            im2, the second image
            F, the fundamental matrix
            x1, x-coordinates of a pixel on im1
            y1, y-coordinates of a pixel on im1
    Output: x2, x-coordinates of the pixel on im2
            y2, y-coordinates of the pixel on im2

'''
def epipolarCorrespondence(im1, im2, F, x1, y1, window = 10):
    # Compute unit epipolar line
    l = F@np.array([x1, y1, 1])
    s = np.sqrt(l[0]**2+l[1]**2)
    l = l/s
    # Compute x coord to search
    ye = im2.shape[0]-1
    ys = 0
    xs = -(l[1] * ys + l[2])/l[0]
    xe = -(l[1] * ye + l[2])/l[0]
    x = lambda y: int(-(l[1] * y + l[2])/l[0])
    # Grab pixels around (x1,y1)
    template = im1[y1-window:y1+window+1,x1-window:x1+window+1,:]
    template_g = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    template_g = template_g/np.sum(template_g)
    # Check along epipolar line; assume pr is within k pixels of pl
    k = 40
    y_start = max(window, y1-k)
    y_end = min(ye-window, y1+k)
    error_best = float('inf')
    x2 = 0
    y2 = 0
    for y_search in range(y_start, y_end+1):
        # Compute x coordinate
        x_search = x(y_search)
        # Get relevant slice
        cur_slice = im2[y_search-window:y_search+window+1,
            x_search-window:x_search+window+1,:]
        cur_slice_g = cv2.cvtColor(cur_slice, cv2.COLOR_BGR2GRAY)
        cur_slice_g = cur_slice_g/np.sum(cur_slice_g)
        # Compute error
        error = template_g - cur_slice_g
        error = np.linalg.norm(error)
        # Update as necesary
        if(error < error_best):
            error_best = error
            x2 = x_search
            y2 = y_search
    return x2, y2

'''
Q5.1: Extra Credit RANSAC method.
    Input:  pts1, Nx2 Matrix
            pts2, Nx2 Matrix
            M, a scaler parameter
    Output: F, the fundamental matrix
            inliers, Nx1 bool vector set to true for inliers
'''
def ransacF(pts1, pts2, M, nIters=1000, tol=0.42):
    F_best = np.zeros((3,3))
    inlier_best = 0
    inliers_best = np.zeros(pts1.shape[0])
    for i in range(0,nIters):
        print('Loop: {}/{}'.format(i, nIters))
        # Randomly pick 8 pairs of points
        sample_indx = np.random.choice(pts1.shape[0],8,replace=False)
        pts1_sample = pts1[sample_indx,:]
        pts2_sample = pts2[sample_indx,:]
        # Compute F using 8 points
        F_cur = eightpoint(pts1_sample, pts2_sample, M)
        # Compute inliers
        inlier_cur = 0
        inliers_cur = np.zeros(pts1.shape[0])
        for j in range(0,pts1.shape[0]):
            # Points
            pl = np.append(pts1[j,:],1)
            pr = np.append(pts2[j,:],1)
            # Compute line
            #line = np.transpose(F_cur)@pr
            line = F_cur@pl
            # Compute distance
            #dist = abs(np.transpose(pr)@line/math.sqrt(line[0]**2+line[1]**2))
            dist = abs(np.transpose(pr)@line)
            if(dist < tol):
                inlier_cur += 1
                inliers_cur[j] = True
        # Update best as necessary
        if(inlier_cur > inlier_best):
            inlier_best = inlier_cur
            inliers_best = inliers_cur
            F_best = F_cur
            print('** Inlier update: {}/{} ({}%)'.format(inlier_best,pts1.shape[0], inlier_best*100/pts1.shape[0]))
    return [F_best, inliers_best]

'''
Q5.2:Extra Credit  Rodrigues formula.
    Input:  r, a 3x1 vector
    Output: R, a rotation matrix
'''
def rodrigues(r):
    theta = np.linalg.norm(r)
    r /= theta
    K = np.array([
        [0,-r[2,0],r[1,0]],
        [r[2,0],0,-r[0,0]],
        [-r[1,0],r[0,0],0]
    ])
    R = np.identity(3)+math.sin(theta)*K+(1-math.cos(theta))*(K@K)
    return R

'''
Q5.2:Extra Credit  Inverse Rodrigues formula.
    Input:  R, a rotation matrix
    Output: r, a 3x1 vector
'''
def invRodrigues(R):
    theta = math.acos((np.trace(R)-1)/2)
    w = 1/(2*math.sin(theta))*np.array([
        [R[2,1]-R[1,2]],
        [R[0,2]-R[2,0]],
        [R[1,0]-R[0,1]]
    ])
    r = theta * w
    return r

'''
Q5.3: Extra Credit Rodrigues residual.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2, the intrinsics of camera 2
            p2, the 2D coordinates of points in image 2
            x, the flattened concatenationg of P, r2, and t2.
    Output: residuals, 4N x 1 vector, the difference between original and estimated projections
'''
def rodriguesResidual(K1, M1, p1, K2, p2, x):
    # Replace pass by your implementation
    pass

'''
Q5.3 Extra Credit  Bundle adjustment.
    Input:  K1, the intrinsics of camera 1
            M1, the extrinsics of camera 1
            p1, the 2D coordinates of points in image 1
            K2,  the intrinsics of camera 2
            M2_init, the initial extrinsics of camera 1
            p2, the 2D coordinates of points in image 2
            P_init, the initial 3D coordinates of points
    Output: M2, the optimized extrinsics of camera 1
            P2, the optimized 3D coordinates of points
'''
def bundleAdjustment(K1, M1, p1, K2, M2_init, p2, P_init):
    # Replace pass by your implementation
    pass


if __name__ == "__main__":
    # # 2.1 - 8 point correspond:
    # Get points
    some_corresp = np.load('../data/some_corresp.npz')
    pts1 = some_corresp['pts1']
    pts2 = some_corresp['pts2']
    # Get dimensions
    img1 = cv2.imread('../data/im1.png')
    img2 = cv2.imread('../data/im2.png')
    imheight, imwidth, _ = img1.shape
    M = np.max([imwidth, imheight])
    print('Image: H = {}, W = {}'.format(imheight, imwidth))
    # Compute fundamental matrix
#    F = eightpoint(pts1, pts2, M)
    # Visualize results
#    helper.displayEpipolarF(img1, img2, F)
    # Save results
#    np.savez_compressed('q2_1.npz',
#        F=F,
#        M=M
#    )

    # # 3.1. - Essential matrix
    # Get intrinsics
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    # Get F
    q2_1 = np.load('q2_1.npz')
    F = q2_1['F']
    print('Fundamental matrix:\n{}'.format(F))
    # Compute E
#    E = essentialMatrix(F, K1, K2)
    # Save results
#    np.savez_compressed('q3_1.npz',
#       E=E,
#       F=F
#    )

    # # 3.2. - Triangulate
    # Get E
    q3_1 = np.load('q3_1.npz')
    E = q3_1['E']
    print('Essential matrix:\n{}'.format(E))
    # Compute M1, C1
    M1 = np.zeros((3,4))
    M1[:,0:3] = np.identity(3)
    C1 = K1 @ M1
    # Compute and get M2, C2
    findM2()
    q3_3 = np.load('q3_3.npz')
    M2 = q3_3['M2']
    C2 = q3_3['C2']
    # Compute coordinates
    [P, err] = triangulate(C1, pts1, C2, pts2)
    # Graph results
#    print('Reprojection error: {}'.format(err))
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')
#    ax.scatter(P[:,0], P[:,1], P[:,2])
#    plt.show()

    # # 4.1. - Correspondence
    # GUI test; used to generate q4_1.npz
#    helper.epipolarMatchGUI(img1, img2, F)

    # # 4.2. - Point cloud
    # get templeCoords
    templeCoords = np.load('../data/templeCoords.npz')
    img1_x = templeCoords['x1']
    img1_y = templeCoords['y1']
    # Compute templeCoords for img2
    img2_x = np.zeros(img1_x.shape)
    img2_y = np.zeros(img1_y.shape)
    for i in range(0, img1_x.shape[0]):
        [img2_x[i,0], img2_y[i,0]] = epipolarCorrespondence(img1, img2, F, img1_x[i,0], img1_y[i,0])
        # if(np.linalg.norm([np.array([img2_x[i,0]-img1_x[i,0], img2_y[i,0]-img1_y[i,0]])]) > 20):
        #     print('Correspondence: ({},{}) -> ({},{}); delta = {}'.format(
        #         img1_x[i,0],img1_y[i,0],img2_x[i,0],img2_y[i,0],
        #         np.linalg.norm([np.array([img2_x[i,0]-img1_x[i,0], img2_y[i,0]-img1_y[i,0]])])))
    # Compute 3D coordinates using triangulate
    [P2, err2] = triangulate(C1, np.append(img1_x,img1_y,axis=1), C2, np.append(img2_x,img2_y,axis=1))
    print('Reprojection error 2: {}'.format(err2))
    # Display results
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')
#    ax.scatter(P2[:,0], P2[:,1], P2[:,2])
#    plt.show()
    # Save results
    np.savez_compressed('q4_2.npz',
        F=F,
        M1=M1,
        M2=M2,
        C1=C1,
        C2=C2
    )

    # # 5.1. F RANSAC
    some_corresp_noisy = np.load('../data/some_corresp_noisy.npz')
    pts1_noisy = some_corresp_noisy['pts1']
    pts2_noisy = some_corresp_noisy['pts2']
    #[Fransac, inliers] = ransacF(pts1_noisy, pts2_noisy, M, 10)
    [Fransac, inliers] = ransacF(pts1_noisy, pts2_noisy, M, 100, 2e-3)
    Fnoisy = eightpoint(pts1_noisy, pts2_noisy, M)
    print('Fnoisy:\n{}'.format(Fnoisy/Fnoisy[-1,-1]))
    print('Num inliers: {}\nFransac:\n{}'.format(np.count_nonzero(inliers), Fransac/Fransac[-1,-1]))
    helper.displayEpipolarF(img1, img2, Fransac)


    # # 5.2. Rodrigues & Inv(Rodrigues)
    # Test cases
    rot_mag = 2.3
    R = np.array([[math.cos(rot_mag),-math.sin(rot_mag),0], [math.sin(rot_mag),math.cos(rot_mag),0], [0,0,1]]) # Z
    #R = R@np.array([[math.cos(rot_mag),0,math.sin(rot_mag)], [0,1,0], [-math.sin(rot_mag),0,math.cos(rot_mag)]]) # Y
    R = R@np.array([[1,0,0], [0,math.cos(rot_mag),-math.sin(rot_mag)], [0,math.sin(rot_mag),math.cos(rot_mag)]]) # X
    print('R = \n{}'.format(R))
    w = invRodrigues(R)
    print('w = \n{}'.format(w))
    R = rodrigues(w)
    print('R = \n{}'.format(R))
    
    


    

