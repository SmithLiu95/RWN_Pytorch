def calc_D_between_scalar_matrix_within_R(x,i,j,R,n):
    mini=max(i-R,0)
    maxi=min(i+R+1,n)
    minj=max(j-R,0)
    maxj=min(j+R+1,n)
    xx=x[mini:maxi,minj:maxj].copy()
    xx.data=abs(xx.data-x[i,j])
    return xx

def get_flatten_indices(data,n):
    indices_y=[]
    value=[]
    for x in data:
        indices=x.indices
        indptr=x.indptr
        for id,i in enumerate(indptr[:-1]):
            indices_y+=list(id*n+indices[i:indptr[id+1]])
        #print(x.data)
        value+=list(x.data)
    return indices_y,value