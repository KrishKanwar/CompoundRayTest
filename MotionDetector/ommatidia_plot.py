import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('Projection2D/data_extraction_test_data.csv', delimiter=',')
pts = data[1:787, 0:3]

ommatid_data = np.genfromtxt('Projection2D/ommatid_data.csv', delimiter=',')

fig = plt.figure(figsize = (10,10))
ax = plt.axes(projection='3d')
ax.grid()



#ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c = 'r', s = 50)
ax.set_title('3D Scatter Plot')


color_list1 = [445, 390]
color_list2 = [447, 387]
color_list3 = [419, 415]

#418,446,448,420,391,388,416
for i in range(786):

    if(i == 417):
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], c = 'b', s = 50)
    elif(i in color_list1):
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], c = 'g', s = 50)
    elif(i in color_list2):
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], c = 'c', s = 50)
    elif(i in color_list3):
            ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], c = 'y', s = 50)
    else:
        ax.scatter(pts[i, 0], pts[i, 1], pts[i, 2], c = 'r', s = 50)

#ax.scatter(pts[417,0], pts[417,1], pts[417,2], c = 'b', s = 50)
#ax.scatter(pts[417,0], pts[,1], pts[,2], c = 'b', s = 50)
#ax.scatter(pts[,0], pts[,1], pts[,2], c = 'b', s = 50)
#ax.scatter(pts[,0], pts[,1], pts[,2], c = 'b', s = 50)
#ax.scatter(pts[,0], pts[,1], pts[,2], c = 'b', s = 50)
#ax.scatter(pts[,0], pts[,1], pts[,2], c = 'b', s = 50)
#ax.scatter(pts[,0], pts[,1], pts[,2], c = 'b', s = 50)

# Set axes label
ax.set_xlabel('x', labelpad=20)
ax.set_ylabel('y', labelpad=20)
ax.set_zlabel('z', labelpad=20)

plt.show()