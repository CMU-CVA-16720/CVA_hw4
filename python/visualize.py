'''
Q4.2:
    1. Integrating everything together.
    2. Loads necessary files from ../data/ and visualizes 3D reconstruction using scatter
'''

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import scipy.optimize

import util
import helper
from findM2 import *
from submission import *

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
    F = eightpoint(pts1, pts2, M)
    # Visualize results
#    helper.displayEpipolarF(img1, img2, F)
    # Save results
    np.savez_compressed('q2_1.npz',
        F=F,
        M=M
    )

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
    E = essentialMatrix(F, K1, K2)
    # Save results
    np.savez_compressed('q3_1.npz',
        E=E,
        F=F
    )

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
    print('Reprojection error: {}'.format(err))
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
    # Compute 3D coordinates using triangulate
    [P2, err2] = triangulate(C1, np.append(img1_x,img1_y,axis=1), C2, np.append(img2_x,img2_y,axis=1))
    print('Reprojection error 2: {}'.format(err2))
    # Display results
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(P2[:,0], P2[:,1], P2[:,2])
    plt.show()
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
    # [Fransac, inliers] = ransacF(pts1_noisy, pts2_noisy, M)
    # Fnoisy = eightpoint(pts1_noisy, pts2_noisy, M)
    # np.savez_compressed('q5_1.npz',
    #     Fnoisy=Fnoisy,
    #     Fransac=Fransac,
    #     inliers=inliers
    # )
    q5_1 = np.load('q5_1.npz')
    Fnoisy = q5_1['Fnoisy']
    Fransac = q5_1['Fransac']
    inliers = q5_1['inliers']
    print('Fnoisy:\n{}'.format(Fnoisy/Fnoisy[-1,-1]))
    print('Num inliers: {}\nFransac:\n{}'.format(np.count_nonzero(inliers), Fransac/Fransac[-1,-1]))
#    helper.displayEpipolarF(img1, img2, Fransac)
#    helper.displayEpipolarF(img1, img2, Fnoisy)


    # # 5.2. Rodrigues & Inv(Rodrigues)
    # Test cases
#    rot_mag = 2.3
#    R = np.array([[math.cos(rot_mag),-math.sin(rot_mag),0], [math.sin(rot_mag),math.cos(rot_mag),0], [0,0,1]]) # Z
#    R = R@np.array([[math.cos(rot_mag),0,math.sin(rot_mag)], [0,1,0], [-math.sin(rot_mag),0,math.cos(rot_mag)]]) # Y
#    R = R@np.array([[1,0,0], [0,math.cos(rot_mag),-math.sin(rot_mag)], [0,math.sin(rot_mag),math.cos(rot_mag)]]) # X
#    print('R = \n{}'.format(R))
#    w = invRodrigues(R)
#    print('w = \n{}'.format(w))
#    R = rodrigues(w)
#    print('R = \n{}'.format(R))

    # # 5.3. Bundle adjustment
    # Perform initial triangulation using inliers
    pts1_in = pts1_noisy[inliers,:]
    pts2_in = pts2_noisy[inliers,:]
    Eransac = essentialMatrix(Fransac, K1, K2)
    M2_init, _, _ = findM2_EC(pts1_in, pts2_in, K1, K2, Eransac)
    [w_init, err_init] = triangulate(C1, pts1_in, K2@M2_init, pts2_in)
    [M2_opt, w_opt] = bundleAdjustment(K1, M1, pts1_in, K2, M2_init, pts2_in, w_init)
    # Compute and compare results
    r2_init = invRodrigues(M2_init[:,0:3])
    t2_init = M2_init[:,3]
    x_init = np.concatenate([w_init.flatten(),r2_init.flatten(),t2_init.flatten()])
    residuals_init = rodriguesResidual(K1, M1, pts1_in, K2, pts2_in, x_init)
    r2_opt = invRodrigues(M2_opt[:,0:3])
    t2_opt = M2_opt[:,3]
    x_opt = np.concatenate([w_opt.flatten(),r2_opt.flatten(),t2_opt.flatten()])
    residuals_opt = rodriguesResidual(K1, M1, pts1_in, K2, pts2_in, x_opt)
    print("Init error = {}".format(np.sum(residuals_init**2)))
    print("Opt  error = {}".format(np.sum(residuals_opt**2)))
    # Plot results
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')
#    ax.scatter(w_init[:,0], w_init[:,1], w_init[:,2])
#    plt.show()
#    fig = plt.figure()
#    ax = fig.add_subplot(projection='3d')
#    ax.scatter(w_opt[:,0], w_opt[:,1], w_opt[:,2])
#    plt.show()
    # Save results
    np.savez_compressed('q5_3.npz',
        M2_init=M2_init,
        w_init=w_init,
        M2_opt=M2_opt,
        w_opt=w_opt
    )
    q5_3 = np.load('q5_3.npz')
    M2_init=q5_3['M2_init']
    w_init=q5_3['w_init']
    M2_opt=q5_3['M2_opt']
    w_opt=q5_3['w_opt']
    print('M2 init:\n{}'.format(M2_init))
    print('M2 opt:\n{}'.format(M2_opt))
