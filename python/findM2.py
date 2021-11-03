'''
Q3.3:
    1. Load point correspondences
    2. Obtain the correct M2
    3. Save the correct M2, C2, and P to q3_3.npz
'''

import numpy as np
import helper
import submission

def findM2():
    # Get points
    some_corresp = np.load('../data/some_corresp.npz')
    pts1 = some_corresp['pts1']
    pts2 = some_corresp['pts2']
    # Get intrinsics
    intrinsics = np.load('../data/intrinsics.npz')
    K1 = intrinsics['K1']
    K2 = intrinsics['K2']
    # Get E
    q3_1 = np.load('q3_1.npz')
    E = q3_1['E']
    # Compute M2 possibilities
    M2s = helper.camera2(E)
    # Compute 3D coordinates using all three possibilities
    M1 = np.zeros((3,4))
    M1[:,0:3] = np.identity(3)
    C1 = K1 @ M1
    P_array = np.zeros((pts1.shape[0],3*4))
    for i in range(0,4):
        C2 = K2 @ M2s[:,:,i]
        [P, err] = submission.triangulate(C1, pts1, C2, pts2)
        P_array[:,3*i:3*i+3] = P
    # Correct M2 should have all positive Z values, since object is in front of cameras
    # Therefore pick M2 with most positive Z values
    M2_indx = np.argmax(np.count_nonzero(P_array[:,2::3]>0,axis=0))
    M2 = M2s[:,:,M2_indx]
    C2 = K2 @ M2s[:,:,M2_indx]
    P = P_array[:,M2_indx*i:3*M2_indx+3]
    # Save M2, C2, P
    np.savez_compressed('q3_3.npz',
       M2=M2,
       C2=C2,
       P=P
    )

if __name__ == "__main__":
    findM2()
