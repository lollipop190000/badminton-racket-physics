"""Exact square/rectangular string-lattice solver for the badminton racket face.
Provides surface() used by the force-map notebook. Part 1 of the study."""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
import time

def li_add(li1,li2): return [li1[i]+li2[i] for i in range(len(li1))]
def li_to_li_mul(li1,li2): return sum([li1[i]*li2[i] for i in range(len(li1))])
def flip(li): return [-i for i in li]
def li_mul(li,n): return [i*n for i in li]
def analyze(last,solution):
    analyzed = [last]
    for sol in solution[::-1]:
        x = li_to_li_mul(analyzed,sol[:-1])+sol[-1]
        analyzed.insert(0,x)
    return analyzed
def solve(N,L_bounary,R_bounary,solution=[]):
    if N==1:
        return analyze(R_bounary[0]/L_bounary[0][0],solution)
    else:
        indexs = 0
        for i in range(N):
            if L_bounary[i][0]!=0: indexs=i;break
        update = list(map(lambda x:x/L_bounary[indexs][0],flip(L_bounary[indexs][1:])))
        n_L_bounary,n_R_bounary = L_bounary[:indexs]+L_bounary[indexs+1:],R_bounary[:indexs]+R_bounary[indexs+1:]
        n_R_bounary = li_add(n_R_bounary,li_mul(map(lambda x:x[0],n_L_bounary),-R_bounary[indexs]/L_bounary[indexs][0]))
        for i in range(len(n_L_bounary)): n_L_bounary[i] = li_add(n_L_bounary[i][1:],li_mul(update,n_L_bounary[i][0]))
        n_s = update+ [R_bounary[indexs]/L_bounary[indexs][0]]
        solution += [n_s]
        return solve(N-1,n_L_bounary,n_R_bounary,solution)
    
def form(N,x,y): return N*y+x
def to_form(N,li,var):
    li = list(map(lambda x:form(N,*x),li))
    var = list(var)
    ans =[0]*(N**2)
    for i in li: ans[i] = var.pop(0)
    return ans
def pprint(data):
    for i in data:
        print(*i,sep="\t")

def surface(N,X,Y,DEPTH,TOT_DEPTH):
    data = [[0]*N for i in range(N)]
    time_start = time.time()
    x,y=X,Y
    data[y][x] = DEPTH
    L_bounary = []
    R_bounary = []
    for j in range(N):
        for i in range(N):
            if (i,j) != (x,y):
                if i==0 and j==0: L_bounary.append(to_form(N,[(0,0),(1,0),(0,1)],(4,-1,-1)))
                elif i==0 and j==N-1: L_bounary.append(to_form(N,[(0,N-1),(1,N-1),(0,N-2)],(4,-1,-1)))
                elif i==N-1 and j==0: L_bounary.append(to_form(N,[(N-1,0),(N-2,0),(N-1,1)],(4,-1,-1)))
                elif i==N-1 and j==N-1: L_bounary.append(to_form(N,[(N-1,N-1),(N-2,N-1),(N-1,N-2)],(4,-1,-1)))
                elif i==0: L_bounary.append(to_form(N,[(0,j),(0,j+1),(0,j-1),(1,j)],(4,-1,-1,-1)))
                elif i==N-1: L_bounary.append(to_form(N,[(N-1,j),(N-1,j+1),(N-1,j-1),(N-2,j)],(4,-1,-1,-1)))
                elif j==0: L_bounary.append(to_form(N,[(i,0),(i+1,0),(i-1,0),(i,1)],(4,-1,-1,-1)))
                elif j==N-1: L_bounary.append(to_form(N,[(i,N-1),(i+1,N-1),(i-1,N-1),(i,N-2)],(4,-1,-1,-1)))
                else: L_bounary.append(to_form(N,[(i,j),(i+1,j),(i-1,j),(i,j+1),(i,j-1)],(4,-1,-1,-1,-1)))
                R_bounary.append(0)
            else:
                L_bounary.append(to_form(N,[(i,j)],(1,)))
                R_bounary.append(DEPTH)
    solved = solve(N**2,L_bounary,R_bounary,[])
    for i in range(N):
        for j in range(N): data[j][i] = solved[form(N,i,j)]
    time_end = time.time()
    print(f'time:{time_end-time_start:.4f}s')
    return data

def plot_surface(data, N, L, DEPTH, TOT_DEPTH):
    ax = plt.axes(projection='3d')
    x = np.linspace(0, L, N)
    y = np.linspace(0, L, N)
    if TOT_DEPTH>DEPTH: z = ax.set_zlim(0,TOT_DEPTH)
    else: z = ax.set_zlim(0,DEPTH)
    X,Y = np.meshgrid(x,y)
    Z = np.array(data)
    ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title('Surface')
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('DEPTH')
    plt.show()
